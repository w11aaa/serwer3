from Library import NP, App, Chart, Plt, BIO, Series, cut
import warnings


Plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
Plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\) DejaVu Sans\.", category=UserWarning)

def DEChart(Data, IsShow):
    DeformationSeries = Series([Item['Deformation'] for Item in Data if Item['Deformation'] != None])
    Bins = NP.arange(0, 1.0 + 0.05, 0.05)
    Labels = [f"{round(Bins[Index], 2)}-{round(Bins[Index+1], 2)}" for Index in range(len(Bins) - 1)]
    RangeSeries = cut(DeformationSeries, Bins, True, Labels, False, 3, True)
    BinCounts = RangeSeries.value_counts().reset_index()
    BinCounts.columns = ["Range", "Frame"]
    if not IsShow:
        (Figure, Axes) = Plt.subplots(figsize=(11, 6.5))
        Axes.bar(BinCounts["Range"], BinCounts["Frame"])
        Axes.set_title("下水道变形范围分布图")
        Axes.set_xlabel("范围")
        Axes.set_ylabel("帧数")
        Plt.xticks(rotation=90)
        Plt.tight_layout()
        ChartBytes = BIO()
        Plt.savefig(ChartBytes, format="JPG")
        ChartBytes.seek(0)
        Plt.close(Figure)
        return ChartBytes
    with App.expander("下水道变形范围分布图"):
        App.plotly_chart(Chart.bar(BinCounts, "Range", "Frame", "Range"))

def SpChart(Data, IsShow):
    ShapeSeries = Series([Item['Shape'] for Item in Data])
    ShapeCounts = ShapeSeries.value_counts().reset_index()
    ShapeCounts.columns = ["Shape", "Frame"]
    if not IsShow:
        (Figure, Axes) = Plt.subplots(figsize=(11.00, 6.50))
        Axes.bar(ShapeCounts["Shape"], ShapeCounts["Frame"])
        Axes.set_title("下水道几何形状分布图")
        Axes.set_xlabel("形状")
        Axes.set_ylabel("帧数")
        Plt.tight_layout()
        ChartBytes = BIO()
        Plt.savefig(ChartBytes, format="JPG")
        ChartBytes.seek(0)
        Plt.close(Figure)
        return ChartBytes
    with App.expander("下水道几何形状分布图"):
        App.plotly_chart(Chart.bar(ShapeCounts, "Shape", "Frame", "Shape"))


def RiskChart(Data, IsShow):
    RiskSeries = Series([
        Item.get("RiskLevel") if isinstance(Item, dict) else None
        for Item in Data
    ]).fillna("未知")
    RiskCounts = RiskSeries.value_counts().reset_index()
    RiskCounts.columns = ["RiskLevel", "Frame"]
    if not IsShow:
        (Figure, Axes) = Plt.subplots(figsize=(11.00, 6.50))
        Axes.bar(RiskCounts["RiskLevel"], RiskCounts["Frame"])
        Axes.set_title("下水道风险等级分布图")
        Axes.set_xlabel("风险等级")
        Axes.set_ylabel("帧数")
        Plt.tight_layout()
        ChartBytes = BIO()
        Plt.savefig(ChartBytes, format="JPG")
        ChartBytes.seek(0)
        Plt.close(Figure)
        return ChartBytes
    with App.expander("下水道风险等级分布图"):
        App.plotly_chart(Chart.bar(RiskCounts, "RiskLevel", "Frame", "RiskLevel"))

def ShowVisuals(Data):
    SpChart(Data, True)
    DEChart(Data, True)
    RiskChart(Data, True)
