from pathlib import Path
from typing import Callable, List, Literal, Dict, Tuple, Optional, Union

import cirq
import numpy as np
from qm import QuantumMachinesManager
from qm.jobs.running_qm_job import RunningQmJob
from qm.qua import *

from qualang_tools.bakery.bakery import Baking
from .RBBaker import RBBaker
from .RBResult import RBResult
from .gates import GateGenerator, gate_db, tableau_from_cirq
from .simple_tableau import SimpleTableau
from .util import run_in_thread, pbar
from .verification.command_registry import (
    CommandRegistry,
    decorate_single_qubit_generator_with_command_recording,
    decorate_two_qubit_gate_generator_with_command_recording,
)
from .verification.sequence_tracker import SequenceTracker

from exp.QMMeasurement import QMMeasurement
from exp.RO_macros import multiRO_declare, multiRO_measurement, multiRO_pre_save
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)
import exp.config_par as gc
import xarray as xr


class TwoQubitRb:
    _buffer_length = 4096

    def __init__(
        self,
        config: dict,
        single_qubit_gate_generator: Callable[[Baking, int, float, float, float], None],
        two_qubit_gate_generators: Dict[Literal["sqr_iSWAP", "CNOT", "CZ"], Callable[[Baking, int, int], None]],
        prep_func: Callable[[], None],
        measure_func: Callable[[], Tuple],
        verify_generation: bool = False,
        interleaving_gate: Optional[List[cirq.GateOperation]] = None,
    ):
        """
        A class for running two qubit randomized benchmarking experiments.

        This class is used to generate the experiment configuration and run the experiment.
        The experiment is run by calling the run method.

        Gate generation is performed using the `Baking`[https://github.com/qua-platform/py-qua-tools/blob/main/qualang_tools/bakery/README.md] class.
        This class adds to QUA the ability to generate arbitrary waveforms ("baked waveforms") using syntax similar to QUA.

        Args:
            config: A QUA configuration containing the configuration for the experiment.

            single_qubit_gate_generator: A callable used to generate a single qubit gate using a signature similar to `phasedXZ`[https://quantumai.google/reference/python/cirq/PhasedXZGate].
                This is done using the baking object (see above).
                Note that this allows us to execute every type of single qubit gate.
                Callable arguments:
                    baking: The baking object.
                     x: The x rotation exponent.
                    z: The z rotation exponent.
                    a: the axis phase exponent.

            two_qubit_gate_generators: A dictionary mapping one or more two qubit gate names to callables used to generate those gates.
                This is done using the baking object (see above).
                Callable arguments:
                    baking: The baking object.
                    qubit1: The first qubit number.
                    qubit2: The second qubit number.
                This callable should generate a two qubit gate.

            prep_func: A callable used to reset the qubits to the |00> state. This function does not use the baking object, and is a proper QUA code macro.
                Callable arguments: None

            measure_func: A callable used to measure the qubits. This function does not use the baking object, and is a proper QUA code macro.
                Callable[[], Tuple[_Expression, _Expression]]: A tuple containing the measured values of the two qubits as Qua expressions.
                The expression must evaluate to a boolean value. False means |0>, True means |1>. The MSB is the first qubit.

            verify_generation: A boolean indicating whether to verify the generated sequences. Not be used in production, as it is very slow.

            interleaving_gate: Interleaved gate represented as list of cirq GateOperation
        """
        for i, qe in config["elements"].items():
            if "operations" not in qe:
                qe["operations"] = {}

        self._command_registry = CommandRegistry()
        self._sequence_tracker = SequenceTracker(command_registry=self._command_registry)

        single_qubit_gate_generator = decorate_single_qubit_generator_with_command_recording(
            single_qubit_gate_generator, self._command_registry
        )
        two_qubit_gate_generators = decorate_two_qubit_gate_generator_with_command_recording(
            two_qubit_gate_generators, self._command_registry
        )
        self._rb_baker = RBBaker(
            config, single_qubit_gate_generator, two_qubit_gate_generators, interleaving_gate, self._command_registry
        )

        self._interleaving_gate = interleaving_gate
        self._interleaving_tableau = tableau_from_cirq(interleaving_gate) if interleaving_gate is not None else None
        self._config = self._rb_baker.bake()
        self._symplectic_generator = GateGenerator(set(two_qubit_gate_generators.keys()))
        self._prep_func = prep_func
        self._measure_func = measure_func
        self._verify_generation = verify_generation

    def convert_sequence_to_cirq(self, sequence: List[int]) -> List[cirq.GateOperation]:
        gates = []
        for cmd_id in sequence:
            gates.extend(self._rb_baker.gates_from_cmd_id(cmd_id))
        return gates

    def _verify_rb_sequence(self, gate_ids, final_tableau: SimpleTableau):
        if final_tableau != SimpleTableau(np.eye(4), [0, 0, 0, 0]):
            raise RuntimeError("Verification of RB sequence failed")
        gates = []
        for gate_id in gate_ids:
            if gate_id == gate_db.get_interleaving_gate():
                gates.extend(self._interleaving_gate)
            else:
                gates.extend(self._symplectic_generator.generate(gate_id))

        unitary = cirq.Circuit(gates).unitary()
        fixed_phase_unitary = np.conj(np.trace(unitary) / 4) * unitary
        if np.linalg.norm(fixed_phase_unitary - np.eye(4)) > 1e-12:
            raise RuntimeError("Verification of RB sequence failed")

    def _gen_rb_sequence(self, depth):
        gate_ids = []
        tableau = SimpleTableau(np.eye(4), [0, 0, 0, 0])
        for i in range(depth):
            symplectic = gate_db.rand_symplectic()
            pauli = gate_db.rand_pauli()
            gate_ids.append(symplectic)
            gate_ids.append(pauli)

            tableau = tableau.then(gate_db.get_tableau(symplectic)).then(gate_db.get_tableau(pauli))

            if self._interleaving_tableau is not None:
                gate_ids.append(gate_db.get_interleaving_gate())
                tableau = tableau.then(self._interleaving_tableau)

        inv_tableau = tableau.inverse()
        inv_id = gate_db.find_symplectic_gate_id_by_tableau_g(inv_tableau)
        after_inv_tableau = tableau.then(gate_db.get_tableau(inv_id))

        pauli = gate_db.find_pauli_gate_id_by_tableau_alpha(after_inv_tableau)

        gate_ids.append(inv_id)
        gate_ids.append(pauli)

        if self._verify_generation:
            final_tableau = after_inv_tableau.then(gate_db.get_tableau(pauli))
            self._verify_rb_sequence(gate_ids, final_tableau)

        return gate_ids

    def _gen_qua_program(
        self,
        sequence_depths: list[int],
        num_repeats: int,
        num_averages: int,
    ):
        with program() as prog:
            sequence_depth = declare(int)
            repeat = declare(int)
            n_avg = declare(int)
            state = declare(int)
            length = declare(int)
            progress = declare(int)
            progress_os = declare_stream()
            state_os = declare_stream()
            gates_len_is = declare_input_stream(int, name="__gates_len_is__", size=1)
            gates_is = {
                qe: declare_input_stream(int, name=f"{qe}_is", size=self._buffer_length)
                for qe in self._rb_baker.all_elements
            }

            assign(progress, 0)
            with for_each_(sequence_depth, sequence_depths):
                with for_(repeat, 0, repeat < num_repeats, repeat + 1):
                    assign(progress, progress + 1)
                    save(progress, progress_os)
                    advance_input_stream(gates_len_is)
                    for gate_is in gates_is.values():
                        advance_input_stream(gate_is)
                    assign(length, gates_len_is[0])
                    with for_(n_avg, 0, n_avg < num_averages, n_avg + 1):
                        self._prep_func()
                        self._rb_baker.run(gates_is, length)
                        out1, out2 = self._measure_func()
                        assign(state, (Cast.to_int(out2) << 1) + Cast.to_int(out1))
                        save(state, state_os)

            with stream_processing():
                state_os.buffer(len(sequence_depths), num_repeats, num_averages).save("state")
                progress_os.save("progress")
        return prog

    def _decode_sequence_for_element(self, element: str, seq: list):
        seq = [self._rb_baker.decode(i, element) for i in seq]
        if len(seq) > self._buffer_length:
            RuntimeError("Buffer is too small")
        return seq + [0] * (self._buffer_length - len(seq))

    @run_in_thread
    def _insert_all_input_stream(
        self,
        job: RunningQmJob,
        sequence_depths: List[int],
        num_repeats: int,
        callback: Optional[Callable[[List[int]], None]] = None,
    ):
        for sequence_depth in sequence_depths:
            for repeat in range(num_repeats):
                sequence = self._gen_rb_sequence(sequence_depth)
                if self._sequence_tracker is not None:
                    self._sequence_tracker.make_sequence(sequence)
                job.insert_input_stream("__gates_len_is__", len(sequence))
                for qe in self._rb_baker.all_elements:
                    job.insert_input_stream(f"{qe}_is", self._decode_sequence_for_element(qe, sequence))

                if callback is not None:
                    callback(sequence)

    def run(
        self,
        qmm: QuantumMachinesManager,
        circuit_depths: List[int],
        num_circuits_per_depth: int,
        num_shots_per_circuit: int,
        **kwargs,
    ):
        """
        Runs the randomized benchmarking experiment. The experiment is sweep over Clifford circuits with varying depths.
        For every depth, we generate a number of random circuits and run them. The number of different circuits is determined by
        the num_circuits_per_depth parameter. The number of shots per individual circuit is determined by the num_averages parameter.

        Args:
            qmm (QuantumMachinesManager): The Quantum Machines Manager object which is used to run the experiment.
            circuit_depths (List[int]): A list of the number of Cliffords per circuit (not including inverse).
            num_circuits_per_depth (int): The number of different circuit randomizations per depth.
            num_shots_per_circuit (int): The number of shots per particular circuit.

        """

        prog = self._gen_qua_program(circuit_depths, num_circuits_per_depth, num_shots_per_circuit)

        qm = qmm.open_qm(self._config)
        job = qm.execute(prog)

        gen_sequence_callback = kwargs["gen_sequence_callback"] if "gen_sequence_callback" in kwargs else None
        self._insert_all_input_stream(job, circuit_depths, num_circuits_per_depth, gen_sequence_callback)

        full_progress = len(circuit_depths) * num_circuits_per_depth
        pbar(job.result_handles, full_progress, "progress")
        job.result_handles.wait_for_all_values()

        return RBResult(
            circuit_depths=circuit_depths,
            num_repeats=num_circuits_per_depth,
            num_averages=num_shots_per_circuit,
            state=job.result_handles.get("state").fetch_all(),
        )

    def print_command_mapping(self):
        """
        Prints the mapping of Command ID index, which is understood by the
        input stream, into single-qubit and two-qubit gates.
        """
        self._command_registry.print_commands()

    def print_sequences(self):
        """
        Prints a break-down of all gates/commands which were played in
        each random sequence.
        """
        self._sequence_tracker.print_sequences()

    def save_command_mapping_to_file(self, path: Union[str, Path]):
        """
        Saves a text file containing the mapping of Command ID index, which
        is understood by the input stream, into single-qubit and two-qubit gates.
        """
        self._command_registry.save_to_file(path)

    def save_sequences_to_file(self, path: Union[str, Path]):
        """
        Save a text file of the break-down of all gates/commands which
        were played in each random sequence
        """
        self._sequence_tracker.save_to_file(path)

    def verify_sequences(self):
        """
        Simulates the application of all random sequences on the |00> state
        to ensure that they recover the qubit to |00> correctly.

        Note: You should only call this function *after* self.run().
        """
        self._sequence_tracker.verify_sequences()


class TwoQubitRb_AS( QMMeasurement ):

    def __init__( self, config, qmm: QuantumMachinesManager ):
        super().__init__( config, qmm )

        self.ro_elements = ["q1_ro", "q2_ro"]

        self.preprocess = "ave"
        self.initializer = None

        self.sequence_depths: list[int] = [1, 2, 3],
        self.num_repeats: int = 2,

    def _get_qua_program(self):
        with program() as qua_prog:
            iqdata_stream = multiRO_declare( self.ro_elements )
            n = declare(int)  
            n_st = declare_stream()
            sequence_depth = declare(int)
            repeat = declare(int)
            length = declare(int)
            gates_len_is = declare_input_stream(int, name="__gates_len_is__", size=1)
            gates_is = {
                qe: declare_input_stream(int, name=f"{qe}_is", size=self._buffer_length)
                for qe in self._rb_baker.all_elements
            }

            with for_each_(sequence_depth, self.sequence_depths):
                with for_(repeat, 0, repeat < self.num_repeats, repeat + 1):
                    advance_input_stream(gates_len_is)
                    for gate_is in gates_is.values():
                        advance_input_stream(gate_is)
                    assign(length, gates_len_is[0])
                    with for_(n, 0, n < self.shot_num, n + 1):
                        # Initialization
                        if self.initializer is None:
                            wait(1*u.us, self.ro_elements)
                        else:
                            try:
                                self.initializer[0](*self.initializer[1])
                            except:
                                print("initializer didn't work!")
                                wait(1*u.us, self.ro_elements)

                        self._rb_baker.run(gates_is, length)
                        # measurement
                        multiRO_measurement( iqdata_stream, self.ro_elements, weights='rotated_'  )
                    save(n, n_st)

            with stream_processing():
                n_st.save("iteration")
                multiRO_pre_save( iqdata_stream, self.ro_elements, (len(self.sequence_depths), self.num_repeats))
        return qua_prog

    def _get_fetch_data_list( self ):
        ro_ch_name = []
        for r_name in self.ro_elements:
            ro_ch_name.append(f"{r_name}_I")
            ro_ch_name.append(f"{r_name}_Q")

        data_list = ro_ch_name + ["iteration"]   
        return data_list
    
    def _data_formation( self ):
        coords = { 
            "mixer":np.array(["I","Q"]), 
            "sequence_depth":self.sequence_depths,
            "num_repeats": self.num_repeats,
            }
        match self.preprocess:
            case "shot":
                dims_order = ["mixer","shot","sequence_depth","num_repeats"]
                coords["shot"] = np.arange(self.shot_num)
            case _:
                dims_order = ["mixer","sequence_depth","num_repeats"]

        output_data = {}
        for r_idx, r_name in enumerate(self.ro_elements):
            data_array = np.array([ self.fetch_data[r_idx*2], self.fetch_data[r_idx*2+1]])
            output_data[r_name] = ( dims_order, np.squeeze(data_array))

        dataset = xr.Dataset( output_data, coords=coords )

        # dataset = dataset.transpose("mixer", "prepare_state", "frequency", "amp_ratio")

        self._attribute_config()
        dataset.attrs["ro_LO"] = self.ref_ro_LO
        dataset.attrs["ro_IF"] = self.ref_ro_IF
        dataset.attrs["xy_LO"] = self.ref_xy_LO
        dataset.attrs["xy_IF"] = self.ref_xy_IF
        dataset.attrs["z_offset"] = self.z_offset

        dataset.attrs["z_amp_const"] = self.z_amp
        return dataset

    def _attribute_config( self ):
        self.ref_ro_IF = []
        self.ref_ro_LO = []
        for r in self.ro_elements:
            self.ref_ro_IF.append(gc.get_IF(r, self.config))
            self.ref_ro_LO.append(gc.get_LO(r, self.config))

        self.ref_xy_IF = []
        self. ref_xy_LO = []
        for xy in self.xy_elements:
            self.ref_xy_IF.append(gc.get_IF(xy, self.config))
            self.ref_xy_LO.append(gc.get_LO(xy, self.config))

        self.z_offset = []
        self.z_amp = []
        for z in self.z_elements:
            self.z_offset.append( gc.get_offset(z, self.config ))
            self.z_amp.append(gc.get_const_wf(z, self.config ))

 
    def _lin_z_array( self ):
        return np.arange( -self.z_modify_range, self.z_modify_range, self.z_resolution )

    def _get_pi_length( self ):
        pi_length = self.config["pulses"][f"{self.detector_qubit}_xy_x180_pulse"]["length"]
        return pi_length
