"""Research figures from audited tables; PNG and SVG, no training/evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from analysis.scenario_mix.comprehensive_20261019 import evaluate as E
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

F=E.OUT/"figures"
LABELS={"REAL_ONLY":"Real-only","REAL_REPLAY_PR30_B50":"Real-tail PR30","REAL_REPLAY_PR10_B50":"Real-tail PR10",
        "FULL_B50":"TimeDiff Full","PR30_B50":"TimeDiff PR30","PR10_B50":"TimeDiff PR10"}
LABELS.update({g:g.replace("_"," ") for g in E.LEGACY})
COLORS={"REAL_ONLY":"#616B75","REAL_REPLAY_PR30_B50":"#7B9858","REAL_REPLAY_PR10_B50":"#A88E34","FULL_B50":"#815CA0","PR30_B50":"#2478A9","PR10_B50":"#D77D34"}
STYLES={"REAL_ONLY":"--","REAL_REPLAY_PR30_B50":"--","REAL_REPLAY_PR10_B50":"-.","FULL_B50":":","PR30_B50":"-","PR10_B50":"-"}
for version,palette in [("V2",["#A4BDD2","#7FA3C0","#4E82AC","#35678F","#214A6D"]),("V3",["#D3B59D","#BD9472","#A67950","#895B37","#67412C"])]:
    for i,group in enumerate(g for g in E.LEGACY if g.startswith(version)):
        COLORS[group]=palette[i];STYLES[group]=[":","-.","--","-",(0,(3,1,1,1))][i]
SPLIT_LABELS={"train":"Train 2010–2017","val":"Validation 2018–2019","test":"Test 2020–2025"}
ASSETS=["CASH",*E.tickers()]
ASSET_COLORS=["#D3D8DD","#336A94","#5E8EAE","#94B8CD","#386B5A","#7BA28C","#BACAB8","#B0853E","#D9B578","#93637C","#BC91A5"]
BASE_NOTE="統一修正後 TRF／交易成本 0.25%；本次每組 10 seeds，v2/v3 每組 3 seeds。"


def style():
    font=Path("/System/Library/Fonts/STHeiti Light.ttc")
    if font.exists():
        fm.fontManager.addfont(str(font));plt.rcParams["font.family"]=fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":10,"axes.titlesize":12,"axes.labelsize":10,
        "figure.facecolor":"white","axes.facecolor":"white","axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.color":"#E4E8EB","grid.linewidth":.6,"grid.alpha":.8,"axes.axisbelow":True,"svg.fonttype":"path"})
    F.mkdir(exist_ok=True);(F/"by_group").mkdir(exist_ok=True)


def save(fig,name,note):
    fig.text(.015,.012,note,ha="left",va="bottom",fontsize=9,color="#535D65")
    fig.savefig(F/f"{name}.png",dpi=160,bbox_inches="tight")
    fig.savefig(F/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)


def date_axis(ax,dates):
    start,end=pd.Timestamp(dates.iloc[0]),pd.Timestamp(dates.iloc[-1])
    left,right=mdates.date2num(start),mdates.date2num(end)
    candidate=mdates.AutoDateLocator(minticks=3,maxticks=5).tick_values(start,end)
    middle=[t for t in candidate if left+.12*(right-left)<t<right-.12*(right-left)]
    ax.set_xlim(start,end);ax.set_xticks([left,*middle,right])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    labels=ax.get_xticklabels()
    if labels:
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")


def curve(ax,z,metric,group,band=True,label=None):
    p=z[z.group==group].pivot(index="nominal_timesteps",columns="seed",values=metric)
    x=p.index.to_numpy()/1000
    ax.plot(x,p.mean(axis=1),color=COLORS[group],linestyle=STYLES[group],linewidth=1.8,label=label or LABELS[group])
    if band:ax.fill_between(x,p.quantile(.25,axis=1),p.quantile(.75,axis=1),color=COLORS[group],alpha=.10,linewidth=0)
    ax.set_xlim(25,500);ax.set_xticks([25,100,200,300,400,500])


def checkpoint_overviews(all_rows):
    measures=[("cagr_net_pct","年化淨報酬（%）"),("sharpe","Sharpe"),("mdd_pct","最大回撤幅度（%）")]
    variants=[("01_current_checkpoint_curves",E.CURRENT,"本次六組：各 checkpoint 的連續真實行情績效",False),
              ("02_common_seed_version_curves",["REAL_ONLY","FULL_B50","PR30_B50","PR10_B50","V2_B50","V3_B50"],"共同 seeds 0–2：本次與 v2／v3 B50",True)]
    for name,groups,title,common in variants:
        data=all_rows[all_rows.seed<3] if common else all_rows
        fig,axes=plt.subplots(3,3,figsize=(15,10.5))
        for i,split in enumerate(E.PERIODS):
            z=data[data.split==split]
            for j,(metric,label) in enumerate(measures):
                ax=axes[i,j]
                for group in groups:curve(ax,z,metric,group)
                ax.set_title(f"{SPLIT_LABELS[split]}｜{label}");ax.set_ylabel(label)
                if i==2:ax.set_xlabel("名義訓練步數（千步）")
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.5,.955),frameon=False)
        fig.suptitle(title,fontsize=20,y=1.005)
        fig.subplots_adjust(top=.86,bottom=.10,hspace=.34,wspace=.25)
        save(fig,name,"線＝跨 seeds 平均；色帶＝25–75%，非信賴區間。500k 名義步數＝532,480 實際步數。\nTest 曲線僅供事後診斷，選模只使用 validation Sharpe；各面板縱軸獨立。"+(" 本圖每組均為 3 seeds。" if common else ""))


def selected_all(summary):
    measures=[("cagr_net_pct","年化淨報酬（%）"),("sharpe","Sharpe"),("mdd_pct","最大回撤（%，低較好）"),("es95_daily_pct","每日 ES95（%，低較好）")]
    for split in E.PERIODS:
        fig,axes=plt.subplots(1,4,figsize=(16,9.5),sharey=True)
        z=summary[summary.split==split].set_index("group").loc[E.GROUPS]
        for ax,(metric,label) in zip(axes,measures):
            for i,g in enumerate(E.GROUPS):
                r=z.loc[g];ax.errorbar(r[f"{metric}_mean"],i,xerr=r[f"{metric}_sd"],fmt="o",color=COLORS[g],capsize=3,markersize=6,linewidth=1.4)
            ax.axhline(5.5,color="#A5ADB4",linestyle=":");ax.axhline(10.5,color="#A5ADB4",linestyle=":")
            ax.set_xlabel(label);ax.grid(axis="y",visible=False)
        axes[0].set_yticks(range(len(E.GROUPS)),[LABELS[g] for g in E.GROUPS]);axes[0].invert_yaxis()
        fig.suptitle(f"Sharpe 選模後｜{SPLIT_LABELS[split]} 的全部組別",fontsize=20,y=.985)
        fig.subplots_adjust(left=.16,right=.98,top=.92,bottom=.11,wspace=.25)
        save(fig,f"03_selected_metrics_{split}","點＝平均；線段＝seed 標準差，不是信賴區間。"+BASE_NOTE+"\nv2/v3 的訓練成本、TRF、生成器與執行環境存在差異；它們是歷史參照。")


def load_selected(selected):
    return {(r.group,int(r.seed),r.split):pd.read_csv(ROOT/r.daily_path,parse_dates=["date"]) for r in selected.itertuples()}


def paths(daily):
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex="col")
    for j,split in enumerate(E.PERIODS):
        for g in E.CURRENT:
            ds=[d for (group,seed,s),d in daily.items() if group==g and s==split]
            dates=ds[0].date;v=np.column_stack([d.value for d in ds]);dd=(v/np.maximum.accumulate(v,axis=0)-1)*100
            for ax,array in zip(axes[:,j],[v,dd]):
                ax.plot(dates,np.median(array,axis=1),color=COLORS[g],linestyle=STYLES[g],label=LABELS[g],linewidth=1.8)
                ax.fill_between(dates,np.quantile(array,.25,axis=1),np.quantile(array,.75,axis=1),color=COLORS[g],alpha=.10)
        b=pd.read_csv(E.OUT/"benchmarks"/f"BUY_HOLD_CASH_{split}.csv.gz",parse_dates=["date"])
        axes[0,j].plot(b.date,b.value,color="#20252A",linestyle=":",linewidth=1.2,label="Buy & hold + cash")
        axes[1,j].plot(b.date,(b.value/b.value.cummax()-1)*100,color="#20252A",linestyle=":",linewidth=1.2)
        axes[0,j].set_title(SPLIT_LABELS[split]);axes[0,j].set_ylabel("淨值（初始資本＝1）");axes[1,j].set_ylabel("回撤（%）")
        for ax in axes[:,j]:
            date_axis(ax,dates)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc="upper center",bbox_to_anchor=(.5,.96),frameon=False)
    fig.suptitle("Sharpe 選模後：本次六組的淨值與回撤",fontsize=20,y=1.02)
    fig.subplots_adjust(top=.85,bottom=.13,wspace=.25,hspace=.18)
    save(fig,"04_selected_paths","線＝每日跨 seeds 中位數；色帶＝25–75%；回撤先逐 seed 計算再彙整。各 split 獨立建倉並扣成本。\nTrain、validation 與 test 期間長度不同；累積報酬幅度不宜直接跨 split 比高低。")


def paired():
    p=pd.read_csv(E.OUT/"paired_seed_differences.csv");p=p[p.split=="test"]
    comps=list(p[["treatment","control"]].drop_duplicates().itertuples(index=False,name=None))
    fig,axes=plt.subplots(1,3,figsize=(15,8.5),sharey=True)
    for ax,(metric,label) in zip(axes,[("cagr_net_pct","年化淨報酬差（百分點）"),("sharpe","Sharpe 差"),("mdd_pct","回撤幅度差（百分點）")]):
        for i,(a,b) in enumerate(comps):
            r=p[(p.treatment==a)&(p.control==b)&(p.metric==metric)].iloc[0]
            ax.errorbar(r.mean_difference,i,xerr=[[r.mean_difference-r.ci95_low],[r.ci95_high-r.mean_difference]],fmt="D",color=COLORS[a],capsize=4)
        ax.axvline(0,color="#7E878E",linewidth=1);ax.set_xlabel(label);ax.grid(axis="y",visible=False)
        ax.set_title("負值較好" if metric=="mdd_pct" else "正值較好")
    axes[0].set_yticks(range(len(comps)),[f"{LABELS[a]}\n− {LABELS[b]}" for a,b in comps]);axes[0].invert_yaxis()
    fig.suptitle("Test 2020–2025：相同 seed 的配對差異",fontsize=20,y=.99)
    fig.subplots_adjust(left=.22,right=.98,top=.9,bottom=.13,wspace=.28)
    save(fig,"05_test_paired_differences","菱形＝10 對 seeds 的平均差；線段＝配對 bootstrap 95% 區間（100,000 次）。\n探索性比較，未校正多重比較；區間只反映訓練 seed 變異，不涵蓋市場或生成器不確定性。")


def generalization(summary):
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    families=[E.CURRENT,E.LEGACY[:5],E.LEGACY[5:]]
    for ax,groups,title in zip(axes,families,["本次：10 seeds","v2：3 seeds","v3：3 seeds"]):
        for i,g in enumerate(groups):
            z=summary[summary.group==g].set_index("split").loc[list(E.PERIODS)]
            # Legacy shares root color; use line patterns and markers for B ratios.
            ax.errorbar(range(3),z.sharpe_mean,yerr=z.sharpe_sd,color=COLORS[g],linestyle=STYLES[g] if g in E.CURRENT else ["-","--","-.",":","-"][i],marker=["o","s","^","D","v","P"][i],capsize=2,label=LABELS[g],linewidth=1.5)
        ax.set(xticks=range(3),xticklabels=["Train","Val","Test"],ylabel="Sharpe",title=title)
        ax.legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle("同一個 validation 選中模型：跨期間表現",fontsize=19,y=1.02)
    fig.subplots_adjust(top=.86,bottom=.18,wspace=.25)
    save(fig,"06_generalization","點＝平均、誤差棒＝seed 標準差；三段行情不同，不能把差距全部歸因於過度擬合。\nValidation 同時用於選模，其表現具有選擇偏差；test 未用於選模。")


def training_rewards():
    ep=pd.read_csv(E.OUT/"training_episode_curves.csv")
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for j,(groups,title) in enumerate([(E.CURRENT,"本次"),(E.LEGACY[:5],"v2"),(E.LEGACY[5:],"v3：訓練無成本")]):
        for i,(metric,label) in enumerate([("reward_mean","每 episode reward"),("reward_per_day_mean","每交易日 reward")]):
            for g in groups:curve(axes[i,j],ep,metric,g,band=False)
            axes[i,j].set_title(f"{title}｜{label}");axes[i,j].set_ylabel(label)
            if i==1:axes[i,j].set_xlabel("名義訓練步數（千步）")
        axes[0,j].legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle("實際訓練 episodes：reward 隨步數變化",fontsize=20,y=1.01)
    fig.subplots_adjust(top=.9,bottom=.13,hspace=.28,wspace=.27)
    save(fig,"07_training_episode_rewards","每個 checkpoint 區間先逐 seed 平均，再跨 seeds 平均；reward＝100 × episode log return。\n來源池、episode 長度與 v3 的訓練成本不同，這張圖不等同連續真實 train 回測績效。")


def annual_heatmap():
    z=pd.read_csv(E.OUT/"selected_subperiod_metrics.csv")
    z=z[(z.split=="test")&z.period.astype(str).str.fullmatch(r"\d{4}")]
    fig,axes=plt.subplots(1,2,figsize=(14,10))
    for ax,(metric,label) in zip(axes,[("return_pct","年度累積報酬（%）"),("mdd_pct","年度最大回撤（%）")]):
        data=z.groupby(["group","period"])[metric].mean().unstack().reindex(E.GROUPS)
        image=ax.imshow(data,aspect="auto",cmap="YlGnBu" if metric=="mdd_pct" else "RdBu",vmin=0 if metric=="mdd_pct" else -max(abs(data.min().min()),abs(data.max().max())),vmax=data.max().max() if metric=="mdd_pct" else max(abs(data.min().min()),abs(data.max().max())))
        for i in range(len(data)):
            for j in range(len(data.columns)):
                val=data.iloc[i,j];rgba=image.cmap(image.norm(val));luma=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=9,color="white" if luma<.55 else "#20252A")
        ax.set(xticks=range(len(data.columns)),xticklabels=data.columns,yticks=range(len(data)),yticklabels=[LABELS[g] for g in E.GROUPS],title=label)
        ax.grid(False);fig.colorbar(image,ax=ax,fraction=.03,pad=.02)
    fig.suptitle("選中模型的 test 年度分解",fontsize=20,y=.99)
    fig.subplots_adjust(left=.15,right=.96,top=.92,bottom=.1,wspace=.5)
    save(fig,"08_test_annual_heatmap","格內＝跨 seeds 平均；跨年持股延續，沒有每年重設策略。2020 包含建倉成本，其餘年份以前年末淨值為基準。\n年度最大回撤在各年內重算，不是完整 test 最大回撤的加總。")


def concentration(selected):
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(1,3,figsize=(16,9.5),sharey=True)
    for ax,split in zip(axes,E.PERIODS):
        z=selected[selected.split==split]
        for i,g in enumerate(E.GROUPS):
            v=z[z.group==g]
            largest=v.mean_largest_stock_weight_pct.mean();stock=v.stock_weight_pct.mean()
            ax.plot([largest,stock],[i,i],color="#D0D6DB",linewidth=1)
            ax.errorbar(largest,i,xerr=v.mean_largest_stock_weight_pct.std(ddof=1),fmt="D",color=COLORS[g],capsize=2,markersize=5)
            ax.plot(stock,i,marker="o",markerfacecolor="white",markeredgecolor=COLORS[g],markersize=6)
        ax.set(xlabel="平均權重（%）",title=SPLIT_LABELS[split],xlim=(0,102))
        ax.grid(axis="y",visible=False)
    axes[0].set_yticks(range(len(E.GROUPS)),[LABELS[g] for g in E.GROUPS]);axes[0].invert_yaxis()
    legend=[Line2D([0],[0],marker="D",color="#525B64",linestyle="none",label="最大單股平均 ± seed 標準差"),Line2D([0],[0],marker="o",markerfacecolor="white",markeredgecolor="#525B64",color="none",label="總股票曝險平均")]
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.55,.955),ncol=2,frameon=False)
    fig.suptitle("選中模型：股票曝險與單股集中",fontsize=20,y=1.005)
    fig.subplots_adjust(left=.16,right=.98,top=.88,bottom=.10,wspace=.15)
    save(fig,"09_exposure_concentration","最大單股權重先逐 seed／逐日計算後取平均，避免先平均持股掩蓋不同 seeds 各自集中。\n誤差棒為 seed 標準差，不是信賴區間；本次 10 seeds，v2/v3 各 3 seeds。")


def by_group_curves(all_rows,selected):
    families={
        "curves":[("return_pct","累積淨報酬（%）"),("cagr_net_pct","年化淨報酬（%）"),("sharpe","Sharpe"),("sortino","Sortino"),("calmar","Calmar"),("mdd_pct","最大回撤（%）")],
        "curves_risk":[("volatility_pct","年化波動（%）"),("es95_daily_pct","每日 ES95（%）"),("var95_daily_pct","每日 VaR95（%）"),("annual_turnover","年化換手（倍）"),("fee_factor_drag_pct","累積費用因子拖累（%）"),("stock_weight_pct","平均股票曝險（%）")],
        "curves_allocation":[("mean_largest_stock_weight_pct","平均最大單股（%）"),("max_single_stock_weight_pct","期間最大單股（%）"),("concentrated_over80_pct","單股超過 80% 的日數（%）"),("action_bound_fraction_pct","動作落於邊界比例（%）"),("worst_21d_return_pct","最差 21 日報酬（%）"),("var95_21d_pct","21 日 VaR95（%）"),("es95_21d_pct","21 日 ES95（%）")],
    }
    for group in E.GROUPS:
        for family,measures in families.items():
            fig,axes=plt.subplots(3,len(measures),figsize=(3.7*len(measures),10))
            for i,split in enumerate(E.PERIODS):
                z=all_rows[(all_rows.group==group)&(all_rows.split==split)]
                chosen=selected[(selected.group==group)&(selected.split==split)]
                for j,(metric,label) in enumerate(measures):
                    ax=axes[i,j]
                    for seed,v in z.groupby("seed"):ax.plot(v.nominal_timesteps/1000,v[metric],color=COLORS[group],alpha=.18,linewidth=.7)
                    curve(ax,z,metric,group)
                    ax.scatter(chosen.nominal_timesteps/1000,chosen[metric],color=COLORS[group],s=14,marker="D",zorder=4,clip_on=False)
                    ax.set_title(label)
                    if j==0:ax.set_ylabel(SPLIT_LABELS[split])
                    if i==2:ax.set_xlabel("名義千步")
                    ax.tick_params(labelsize=8)
            fig.suptitle(f"{LABELS[group]}｜每個 checkpoint 的 train／val／test",fontsize=21,y=1.005)
            fig.subplots_adjust(top=.92,bottom=.10,hspace=.36,wspace=.36)
            save(fig,f"by_group/{family}_{group}","細線＝個別 seed，粗線＝平均，色帶＝25–75%，菱形＝validation Sharpe 選中的步數（所有 split 使用同一模型）。\nTest 曲線僅為診斷；各面板獨立縱軸。"+BASE_NOTE)


def generation_quality():
    d=pd.read_csv(E.OUT/"generation_quality_comparison.csv")
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    x=np.arange(len(d));width=.32
    for ax,q in zip(axes,[30,10]):
        a=ax.bar(x-width/2,d[f"target_pr{q}_pct"],width,color="#9BBBD0",label="抽中條件")
        b=ax.bar(x+width/2,d[f"realized_pr{q}_pct"],width,color="#2478A9",label="生成路徑")
        ax.bar_label(a,fmt="%.1f",padding=3,fontsize=9);ax.bar_label(b,fmt="%.1f",padding=3,fontsize=9)
        ax.axhline(q,color="#747D85",linestyle=":",label=f"歷史 PR{q} 比例")
        ax.set(xticks=x,xticklabels=d.dataset,title=f"報酬 ≤ 訓練 PR{q}",ylabel="路徑占比（%）",ylim=(0,65))
        ax.grid(axis="x",visible=False)
    axes[0].legend(frameon=False,fontsize=9)
    ax=axes[2]
    a=ax.bar(x-width/2,d.paired_real_stock_correlation,width,color="#9BBBD0",label="配對真實時期")
    b=ax.bar(x+width/2,d.mean_stock_correlation,width,color="#2478A9",label="生成路徑")
    ax.bar_label(a,fmt="%.3f",padding=3,fontsize=9);ax.bar_label(b,fmt="%.3f",padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=d.dataset,title="股票間平均相關性",ylabel="21 日視窗內兩兩 log return 相關係數",ylim=(0,.5))
    ax.grid(axis="x",visible=False);ax.legend(frameon=False,fontsize=9)
    fig.suptitle("三批合成資料：尾部占比與股票相關性",fontsize=19,y=1.01)
    fig.subplots_adjust(top=.86,bottom=.17,wspace=.24)
    save(fig,"10_generation_tail_comparison","各批 500 次條件抽樣 × 4 條路徑，共 2,000 條；使用相同 TimeDiff checkpoint 與 diffusion seed。\n完整池已包含尾部；來源池的 70／30 配額不等於實際報酬的尾部比例。相關性先逐視窗計算 45 個股票配對平均，再依抽樣權重平均。")


def horizon_risk(all_rows,summary):
    measures=[("worst_21d_return_pct","最差 21 日報酬（%）"),("var95_21d_pct","21 日 VaR95（%）"),("es95_21d_pct","21 日 ES95（%）")]
    fig,axes=plt.subplots(1,3,figsize=(14,9.5),sharey=True)
    z=summary[summary.split=="test"].set_index("group")
    for ax,(metric,label) in zip(axes,measures):
        for i,g in enumerate(E.GROUPS):
            r=z.loc[g];ax.errorbar(r[f"{metric}_mean"],i,xerr=r[f"{metric}_sd"],fmt="o",color=COLORS[g],capsize=3,markersize=6)
        ax.set_xlabel(label);ax.grid(axis="y",visible=False)
    axes[0].set_yticks(range(len(E.GROUPS)),[LABELS[g] for g in E.GROUPS]);axes[0].invert_yaxis()
    fig.suptitle("Test 2020–2025：對齊生成期間的 21 日尾部風險",fontsize=20,y=.99)
    fig.subplots_adjust(left=.18,right=.98,top=.92,bottom=.11,wspace=.25)
    save(fig,"11_selected_horizon21_risk","點＝seed 平均，線段＝seed 標準差；最差報酬越高越好，VaR／ES 損失幅度越低越好。\n採重疊的 21 交易日視窗；重疊視窗不視為獨立樣本。")
    fig,axes=plt.subplots(3,3,figsize=(15,10.5))
    for i,split in enumerate(E.PERIODS):
        for j,(metric,label) in enumerate(measures):
            ax=axes[i,j]
            for group in E.CURRENT:curve(ax,all_rows[all_rows.split==split],metric,group)
            ax.set_title(f"{SPLIT_LABELS[split]}｜{label}");ax.set_ylabel(label)
            if i==2:ax.set_xlabel("名義訓練步數（千步）")
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.5,.955),frameon=False)
    fig.suptitle("本次六組：21 日尾部風險隨訓練步數變化",fontsize=20,y=1.005)
    fig.subplots_adjust(top=.86,bottom=.10,hspace=.34,wspace=.25)
    save(fig,"12_current_horizon21_curves","線＝跨 seeds 平均，色帶＝25–75% 範圍；各面板獨立縱軸。\n21 日視窗位於各 split 內；test 曲線僅為事後診斷，沒有用於選模。")


def by_group_holdings(daily):
    for group in E.GROUPS:
        fig,axes=plt.subplots(2,3,figsize=(17,8),sharex="col",sharey="row")
        for j,split in enumerate(E.PERIODS):
            ds=[d for (g,seed,s),d in daily.items() if g==group and s==split]
            dates=ds[0].date
            weights=np.stack([d[[f"weight_{t}" for t in ASSETS]].to_numpy() for d in ds])
            mean=weights.mean(axis=0)*100
            axes[0,j].stackplot(dates,mean.T,colors=ASSET_COLORS,labels=ASSETS,linewidth=0,rasterized=False)
            axes[0,j].set_title(SPLIT_LABELS[split]);axes[0,j].set_ylim(0,100)
            stock=(1-weights[:,:,0])*100;largest=weights[:,:,1:].max(axis=2)*100
            for array,color,label,line in [(stock,"#2478A9","總股票曝險","-"),(largest,"#D77D34","最大單股權重","--")]:
                axes[1,j].plot(dates,np.median(array,axis=0),color=color,linestyle=line,label=label,linewidth=1.7)
                axes[1,j].fill_between(dates,np.quantile(array,.25,axis=0),np.quantile(array,.75,axis=0),color=color,alpha=.15)
            axes[1,j].set_ylim(0,102)
            for ax in axes[:,j]:
                date_axis(ax,dates)
        axes[0,0].set_ylabel("平均實際持股比例（%）");axes[1,0].set_ylabel("曝險／集中度（%）")
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=11,loc="upper center",bbox_to_anchor=(.5,.94),frameon=False,fontsize=9)
        axes[1,0].legend(frameon=False,fontsize=9,ncol=2)
        fig.suptitle(f"{LABELS[group]}｜Sharpe 選中模型的持股比例",fontsize=21,y=1.005)
        fig.subplots_adjust(top=.86,bottom=.13,wspace=.08,hspace=.15)
        save(fig,f"by_group/holdings_{group}","上排＝每日實際收盤權重的跨 seeds 平均（含現金）；下排＝每日跨 seeds 中位數與 25–75% 範圍。\n下排最大單股權重先逐 seed 計算，避免平均持股掩蓋集中風險；各 split 獨立建倉。")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--overview-only",action="store_true");args=ap.parse_args()
    style();all_rows=pd.read_csv(E.OUT/"all_checkpoint_metrics.csv");selected=pd.read_csv(E.OUT/"selected_metrics.csv");summary=pd.read_csv(E.OUT/"selected_summary.csv")
    daily=load_selected(selected)
    checkpoint_overviews(all_rows);selected_all(summary);paths(daily);paired();generalization(summary);training_rewards();annual_heatmap();concentration(selected);generation_quality();horizon_risk(all_rows,summary)
    if not args.overview_only:by_group_curves(all_rows,selected);by_group_holdings(daily)
    print(f"Rendered {len(list(F.rglob('*.png')))} PNG figures and matching SVG files",flush=True)


if __name__=="__main__":main()
