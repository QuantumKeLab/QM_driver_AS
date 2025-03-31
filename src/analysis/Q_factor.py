from xarray import DataArray
import numpy as np
import matplotlib.pyplot as plt
from qcat.analysis.resonator.photon_dep.res_data import ResonatorData
from numpy import array, arange, real, imag, arctan2

def QM_CS_ana(ds: DataArray) -> list:
    """
    分析 dataset 並產生每個 q 的圖形，
    圖形中直接標示出 Qi, Qc, Ql 與 fr，
    並將 (file_name, fig) 存入 output_fig list 後返回。

    注意：這邊假設 ds 的維度為 (q_idx, mixer, frequency)，
    而 mixer 的值分別為 "I" 與 "Q"。
    """
    output_fig = []  # 用來存放 (file_name, fig) tuple
    q_vars = [var for var in ds.data_vars if var.startswith("q")]

    for idx, q in enumerate(q_vars):
        ds_q = ds[q]  # 直接選取對應的 DataArray
        # 如果 q 名稱的最後一段不是 "freq" 則進行處理
        if str(q).split("_")[-1] != "freq":
            # 使用 mixer 座標來取得 I 與 Q 分量
            S21_I = np.array(ds_q.sel(mixer="I"))
            S21_Q = np.array(ds_q.sel(mixer="Q"))
            S21 = S21_I + 1j * S21_Q

            # 計算頻率，從第 5 個點開始並做 LO 與 IF 補正
            freq = np.array(ds_q["frequency"])[5:] * 1e6 + ds_q.attrs['ro_LO'][idx] + ds_q.attrs['ro_IF'][idx]

            # 選取 S21 從第 5 個點開始的部分作為擬合資料
            zdata = S21[5:]

            # 利用 ResonatorData 進行擬合 (請確保此物件已正確定義與引入)
            res_er = ResonatorData(freq=freq, zdata=zdata)
            result, data2plot, fit2plot = res_er.fit()

            # 提取擬合參數
            Qi = result.get("Qi_dia_corr", None)   # 可依需求選擇 "Qi_no_corr"
            Qc = result.get("absQc", None)
            Ql = result.get("Ql", None)
            fr = result.get("fr", None)

            # 定義檔名 (以 q 的名稱命名)
            file_name = f"{q}_fit.png"

            # 建立圖形 (2x2 子圖)
            fig, ax = plt.subplots(2, 2, figsize=(12, 12))

            # 子圖1：幅值曲線與擬合結果
            ax0 = ax[0][0]
            ax0.grid()
            ax0.plot(freq, result['A'] * np.abs(data2plot), label='data')
            ax0.plot(freq, result['A'] * np.abs(fit2plot), c="red", label='fitting')
            ax0.vlines(result['fr'], result['A'] * min(data2plot), result['A'] * max(data2plot), linestyles="--")
            ax0.set_title(f"{q} cavity @ {round(float(result['fr'])*1e-9, 5)} GHz")
            # 用科學記號、4個有效數字格式化數值
            ax0.text(0.05, 0.95, 
                     f"Qi: {Qi:.4g}\nQc: {Qc:.4g}\nQl: {Ql:.4g}\nfr: {fr:.4g}",
                     transform=ax0.transAxes, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.5))
            ax0.legend()

            # 子圖2：相位曲線
            ax1 = ax[0][1]
            ax1.grid()
            ax1.plot(freq, np.arctan2(np.imag(data2plot), np.real(data2plot)), label='data')
            ax1.plot(freq, np.arctan2(np.imag(fit2plot), np.real(fit2plot)), c="red", label='fitting')
            ax1.set_title("Phase")
            ax1.legend()

            # 子圖3：S21 原始資料 (散點圖)
            ax2 = ax[1][0]
            ax2.grid()
            ax2.scatter(np.real(S21[1:]), np.imag(S21[1:]), label='data')
            ax2.set_title("S21 raw data")
            ax2.legend()

            # 子圖4：S21 擬合後資料
            ax3 = ax[1][1]
            ax3.grid()
            ax3.scatter(result['A'] * np.real(data2plot), result['A'] * np.imag(data2plot), label='data')
            ax3.scatter(result['A'] * np.real(fit2plot), result['A'] * np.imag(fit2plot), label='fit', c='red', s=10)
            ax3.set_title("S21 after fit")
            ax3.legend()

            plt.tight_layout()
            # 不在此處呼叫 plt.show()，而是在迴圈外一次全部顯示
            output_fig.append((file_name, fig))

    # 在所有圖建立完後，一次性顯示
    plt.show()
    return output_fig
