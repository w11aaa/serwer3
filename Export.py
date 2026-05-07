from Library import Image, NP, CV2, Zip, XLImage, BIO, Workbook, Alignment, Table, TableStyleInfo, A4, colors, getSampleStyleSheet, SimpleDocTemplate, PDFTable, TableStyle, PDFImage, Paragraph, Spacer, inch
from Analysis import Conclude
from Visualize import SpChart, DEChart, RiskChart
import os as OS
import tempfile as Temp

# ── 注册中文字体 ──────────────────────────────────────────────
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle

def _register_cn_font():
    """尝试注册系统中文字体，返回可用的字体名"""
    candidates = [
        ("SimHei",   "C:/Windows/Fonts/simhei.ttf"),
        ("SimSun",   "C:/Windows/Fonts/simsun.ttc"),
        ("MsYaHei",  "C:/Windows/Fonts/msyh.ttc"),
        ("STSONG",   "C:/Windows/Fonts/STSONG.TTF"),
    ]
    for name, path in candidates:
        try:
            if OS.path.exists(path):
                pdfmetrics.registerFont(TTFont(name, path))
                return name
        except Exception:
            continue
    return "Helvetica"   # 回退，仍会有方块但不崩溃

CN_FONT = _register_cn_font()

def _cn_style(base_style, size=10, bold=False):
    """基于已有样式创建支持中文的 ParagraphStyle"""
    return ParagraphStyle(
        name=f"CN_{base_style.name}_{size}",
        parent=base_style,
        fontName=CN_FONT,
        fontSize=size,
        leading=size * 1.4,
    )

def GetVideo(Data, PadBot=40, PadColor=(0, 0, 0), TxtColor=(255, 255, 255), Scale=1, Thick=2):
    def AddCaption2Frame(Img, Caption):
        Font = CV2.FONT_HERSHEY_SIMPLEX
        Size = CV2.getTextSize(Caption, Font, Scale, Thick)[0]
        GetPaddedImg = CV2.copyMakeBorder(Img, 0, PadBot, 0, 0, CV2.BORDER_CONSTANT, PadColor)
        Coord = (GetPaddedImg.shape[1] - Size[0]) // 2, Img.shape[0] + (PadBot + Size[1]) // 2
        (X, Y) = Coord
        CV2.putText(GetPaddedImg, Caption, (X, Y), Font, Scale, TxtColor, Thick, CV2.LINE_AA)
        return GetPaddedImg
    VideoBuffer = BIO()
    if not Data:
        return VideoBuffer

    StandardSize = None
    TempPath = None
    VideoWriter = None
    try:
        with Temp.NamedTemporaryFile(suffix=".mp4", delete=False) as TempFile:
            TempPath = TempFile.name

        for OrdNum, Dict in enumerate(Data, 1):
            PlotImg = Dict["Plot"]
            MaskImg = Dict["Draw"]
            if PlotImg.ndim == 2:
                PlotImg = CV2.cvtColor(PlotImg, CV2.COLOR_GRAY2RGB)
            if MaskImg.ndim == 2:
                MaskImg = CV2.cvtColor(MaskImg, CV2.COLOR_GRAY2RGB)
            if StandardSize is None:
                StandardSize = (MaskImg.shape[1], MaskImg.shape[0])
            PlotImg = CV2.resize(PlotImg, StandardSize)
            MaskImg = CV2.resize(MaskImg, StandardSize)
            Combination = NP.hstack((PlotImg, MaskImg))
            Cation = f"Frame:{OrdNum}  Conclude:{Conclude(Dict['Shape'])}  Risk:{Dict.get('RiskLevel')}"
            OutFrame = AddCaption2Frame(Combination, Cation)
            OutFrame = CV2.cvtColor(OutFrame, CV2.COLOR_RGB2BGR)

            if VideoWriter is None:
                Height, Width = OutFrame.shape[:2]
                FourCC = CV2.VideoWriter_fourcc(*"mp4v")
                VideoWriter = CV2.VideoWriter(TempPath, FourCC, 3, (Width, Height))

            VideoWriter.write(OutFrame)

        if VideoWriter is not None:
            VideoWriter.release()
            VideoWriter = None

        with open(TempPath, "rb") as RF:
            VideoBuffer.write(RF.read())
        VideoBuffer.seek(0)
        return VideoBuffer
    finally:
        if VideoWriter is not None:
            VideoWriter.release()
        if TempPath and OS.path.exists(TempPath):
            OS.remove(TempPath)

def _build_cover_page(elements, styles, meta):
    """生成 PDF 封面页，meta 为 dict，包含 project, operator, segment, date, conclusion, score"""
    from reportlab.lib import colors as _colors
    from reportlab.platypus import HRFlowable

    cn_title  = _cn_style(styles['Title'],   size=20)
    cn_normal = _cn_style(styles['Normal'],  size=10)
    cn_h2     = _cn_style(styles['Heading2'], size=13)

    elements.append(Spacer(1, 0.6 * inch))
    elements.append(Paragraph(meta.get("project", "下水道截面几何变形检测报告"), cn_title))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(HRFlowable(width="100%", thickness=2, color=_colors.HexColor("#2c3e50")))
    elements.append(Spacer(1, 0.3 * inch))

    info_data = [
        ["检测日期", meta.get("date", "—"),       "操作员",   meta.get("operator", "—")],
        ["管段编号", meta.get("segment", "—"),     "检测类型", meta.get("report_type", "—")],
        ["总帧/图数", str(meta.get("total", "—")), "高风险数", str(meta.get("high_risk", "—"))],
    ]
    # 把每个单元格包成 Paragraph 以支持中文
    info_data_p = []
    for row in info_data:
        info_data_p.append([Paragraph(str(c), cn_normal) for c in row])

    info_table = PDFTable(info_data_p, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), _colors.HexColor("#2c3e50")),
        ('BACKGROUND', (2, 0), (2, -1), _colors.HexColor("#2c3e50")),
        ('TEXTCOLOR',  (0, 0), (0, -1), _colors.white),
        ('TEXTCOLOR',  (2, 0), (2, -1), _colors.white),
        ('FONTNAME',   (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',       (0, 0), (-1, -1), 0.5, _colors.HexColor("#bdc3c7")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.3 * inch))

    conclusion = meta.get("conclusion", "—")
    score = meta.get("score")
    score_text = f"  健康评分：{score}" if score is not None else ""
    level_color = {
        "优良": _colors.HexColor("#27ae60"),
        "警告": _colors.HexColor("#e67e22"),
        "危险": _colors.HexColor("#e74c3c"),
    }.get(conclusion, _colors.HexColor("#7f8c8d"))

    cn_concl = _cn_style(styles['Normal'], size=14)
    concl_data = [[Paragraph(f"总体结论：{conclusion}{score_text}", cn_concl)]]
    concl_table = PDFTable(concl_data, colWidths=[7.0*inch])
    concl_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), level_color),
        ('TEXTCOLOR',     (0, 0), (-1, -1), _colors.white),
        ('FONTNAME',      (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE',      (0, 0), (-1, -1), 14),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
    ]))
    elements.append(concl_table)
    elements.append(Spacer(1, 0.2 * inch))

    notes = meta.get("notes", "")
    if notes:
        elements.append(Paragraph(f"备注：{notes}", cn_normal))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(HRFlowable(width="100%", thickness=1, color=_colors.HexColor("#bdc3c7")))
    elements.append(Spacer(1, 0.2 * inch))


def GetPDF(Data, meta=None):
    """视频检测 PDF 报告，引入图片缩放优化速度"""
    Buffer = BIO()
    Doc = SimpleDocTemplate(Buffer, pagesize=A4)
    Elements = []
    Styles = getSampleStyleSheet()

    if meta is None:
        meta = {}
    meta.setdefault("project", "下水道截面几何变形检测报告")
    meta.setdefault("report_type", "视频检测")
    meta.setdefault("total", len(Data))
    meta.setdefault("high_risk", sum(1 for d in Data if d.get("RiskLevel") == "高"))
    _build_cover_page(Elements, Styles, meta)

    cn_h2     = _cn_style(Styles['Heading2'], size=13)
    cn_normal = _cn_style(Styles['Normal'],   size=9)
    Elements.append(Paragraph("检测明细", cn_h2))
    Elements.append(Spacer(1, 8))

    # 预设图片高度优化
    TARGET_H = 150 
    
    TableData = [["帧号", "形状", "风险", "处置建议", "详细参数", "分割图", "掩码图"]]

    for OrdNum, Dict in enumerate(Data, 1):
        def _get_pdf_img(arr):
            if arr is None or (isinstance(arr, NP.ndarray) and arr.size == 0):
                return "无图片"
            try:
                img = Image.fromarray(arr)
                w, h = img.size
                if h > TARGET_H:
                    new_w = int(w * (TARGET_H / h))
                    img = img.resize((new_w, TARGET_H), Image.LANCZOS)
                
                buf = BIO()
                img.save(buf, format='JPEG', quality=75)
                buf.seek(0)
                
                render_h = 1.2 * inch
                aspect = img.width / img.height
                render_w = render_h * aspect
                return PDFImage(buf, width=render_w, height=render_h)
            except Exception:
                return "解析失败"

        PImg = _get_pdf_img(Dict.get("Plot"))
        DImg = _get_pdf_img(Dict.get("Draw"))

        Risk   = str(Dict.get("RiskLevel", ""))
        Advice = str(Dict.get("ActionSuggestion", ""))
        Details = f"AR: {Dict.get('AspectRatio')}\nOri: {Dict.get('Orientation')}\nDef: {Dict.get('Deformation')}"
        AdvicePara   = Paragraph(Advice.replace('\n', '<br/>'),   cn_normal)
        DetailsPara  = Paragraph(Details.replace('\n', '<br/>'),  cn_normal)

        TableData.append([str(OrdNum), str(Dict.get("Shape", "")), Risk, AdvicePara, DetailsPara, PImg, DImg])

    T = PDFTable(TableData, colWidths=[0.5*inch, 0.9*inch, 0.7*inch, 1.4*inch, 1.1*inch, 1.7*inch, 1.7*inch])
    T.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME',   (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID',       (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
    ]))

    Elements.append(T)
    Doc.build(Elements)
    Buffer.seek(0)
    return Buffer

def GetExcel(Data, ImageHeight=150):
    def GetResizeImage(Img, Height):
        W, H = Img.size
        Size = (int(W * (Height / H)), Height)
        return Img.resize(Size, Image.LANCZOS)
    def PXToColW(Pixel): return (Pixel - 5) / 7
    def PXToRowH(Pixel): return (Pixel * 0.750)
    def AddChart2Excel(Sheet, Img, Row, Col, Height=525):
        ChartRI = GetResizeImage(Image.open(Img), Height)
        XLByteIO = BIO()
        ChartRI.save(XLByteIO, "JPEG")
        XLByteIO.seek(0)
        Sheet.add_image(XLImage(XLByteIO), f"{Col}{Row}")
        (Height, Width) = (ChartRI.height, ChartRI.width)
        Sheet.row_dimensions[Row].height, Sheet.column_dimensions[Col].width = PXToRowH(Height), PXToColW(Width)
    XBuffer = BIO()
    WB = Workbook()
    WS = WB.active
    WS.append(["SegmentedImage", "SegmentedMask", "Frame", "Shape", "AspectRatio", "Orientation", "Deformation", "RiskLevel", "ActionSuggestion"])
    ImgCol = { "A": 0, "B": 0 }
    for OrdNum, Dict in enumerate(Data, 1):
        RowIdx = WS.max_row + 1
        WS.append(["", "", OrdNum, Dict["Shape"], Dict["AspectRatio"], Dict["Orientation"], Dict["Deformation"], Dict.get("RiskLevel"), Dict.get("ActionSuggestion")])
        for (ColLetter, Key) in zip(ImgCol.keys(), ["Plot", "Draw"]):
            if Dict.get(Key) is not None:
                try:
                    LoadImg = Image.fromarray(Dict[Key])
                    ResizedImage = GetResizeImage(LoadImg, ImageHeight)
                    ImageBytes = BIO()
                    ResizedImage.save(ImageBytes, "JPEG")
                    if ResizedImage.width > ImgCol[ColLetter]:
                        ImgCol[ColLetter] = ResizedImage.width
                    ImageBytes.seek(0)
                    WS.add_image(XLImage(ImageBytes), f"{ColLetter}{RowIdx}")
                except Exception:
                    WS.cell(RowIdx, list(ImgCol.keys()).index(ColLetter) + 1, "无图像")
            else:
                WS.cell(RowIdx, list(ImgCol.keys()).index(ColLetter) + 1, "无图像")
        WS.row_dimensions[RowIdx].height = PXToRowH(ImageHeight)
        for Cell in WS[RowIdx]:
            if Cell.column_letter not in ImgCol.keys(): Cell.alignment = Alignment(None, "CENTER".lower())
    for Each in WS.columns:
        Name = Each[0].column_letter
        if Name not in ImgCol.keys():
            MaxLg = 0
            for Cell in Each:
                if Cell.value:
                    MaxLg = max(MaxLg, len(str(Cell.value)))
            WS.column_dimensions[Name].width = (MaxLg + 3.3)
        else:
            Wid = ImgCol[Name]
            WS.column_dimensions[Name].width = PXToColW(Wid)
    NewSegmentResultTable = Table(1, "ResultTable", f"A1:{WS.cell(WS.max_row, WS.max_column).coordinate}")
    NewSegmentResultTable.tableStyleInfo = TableStyleInfo("TableStyleMedium23", False, False, True, False)
    WS.add_table(NewSegmentResultTable)
    WS.title = "视频检测结果"
    ChartSheet = WB.create_sheet("统计图表")
    AddChart2Excel(ChartSheet, SpChart(Data, False), 1, "A")
    AddChart2Excel(ChartSheet, DEChart(Data, False), 3, "A")
    AddChart2Excel(ChartSheet, RiskChart(Data, False), 5, "A")
    WB.save(XBuffer)
    XBuffer.seek(0)
    return XBuffer

def GetImagePDF(Data, meta=None):
    """图像批量检测 PDF 报告，引入图片缩放优化速度"""
    Buffer = BIO()
    Doc = SimpleDocTemplate(Buffer, pagesize=A4)
    Elements = []
    Styles = getSampleStyleSheet()

    if meta is None:
        meta = {}
    meta.setdefault("project", "下水道截面几何变形检测报告")
    meta.setdefault("report_type", "图像批量检测")
    meta.setdefault("total", len(Data))
    meta.setdefault("high_risk", sum(1 for d in Data if d.get("RiskLevel") == "高"))
    _build_cover_page(Elements, Styles, meta)

    cn_h2     = _cn_style(Styles['Heading2'], size=13)
    cn_normal = _cn_style(Styles['Normal'],   size=9)
    Elements.append(Paragraph("检测明细", cn_h2))
    Elements.append(Spacer(1, 8))

    # 预设图片高度（PDF 单位 1.2 inch 约为 86 pt，转换为像素建议 150-200px 兼顾清晰度与速度）
    TARGET_H = 150 
    
    TableData = [["序号", "文件名", "形状", "风险", "处置建议", "详细参数", "分割图", "掩码图"]]

    for OrdNum, Dict in enumerate(Data, 1):
        def _get_pdf_img(arr):
            if arr is None or (isinstance(arr, NP.ndarray) and arr.size == 0):
                return "无图片"
            try:
                img = Image.fromarray(arr)
                # 显著优化：等比例缩放图片，减小内存占用和处理时间
                w, h = img.size
                if h > TARGET_H:
                    new_w = int(w * (TARGET_H / h))
                    img = img.resize((new_w, TARGET_H), Image.LANCZOS)
                
                buf = BIO()
                img.save(buf, format='JPEG', quality=75) # 降低质量进一步提速
                buf.seek(0)
                
                # PDF 渲染尺寸保持 1.2 inch 高度
                render_h = 1.2 * inch
                aspect = img.width / img.height
                render_w = render_h * aspect
                return PDFImage(buf, width=render_w, height=render_h)
            except Exception:
                return "解析失败"

        PImg = _get_pdf_img(Dict.get("Plot"))
        DImg = _get_pdf_img(Dict.get("Draw"))

        Details = f"AR: {Dict.get('AspectRatio')}\nOri: {Dict.get('Orientation')}\nDef: {Dict.get('Deformation')}"
        AdvicePara  = Paragraph(str(Dict.get("ActionSuggestion", "")).replace('\n', '<br/>'), cn_normal)
        DetailsPara = Paragraph(Details.replace('\n', '<br/>'), cn_normal)
        NamePara    = Paragraph(str(Dict.get("Name", OrdNum)), cn_normal)

        TableData.append([str(OrdNum), NamePara, str(Dict.get("Shape", "")),
                          str(Dict.get("RiskLevel", "")), AdvicePara, DetailsPara, PImg, DImg])

    T = PDFTable(TableData, colWidths=[0.4*inch, 1.0*inch, 0.8*inch, 0.6*inch, 1.2*inch, 1.0*inch, 1.5*inch, 1.5*inch])
    T.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME',   (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID',       (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
    ]))

    Elements.append(T)
    Doc.build(Elements)
    Buffer.seek(0)
    return Buffer


def GetImageExcel(Data, ImageHeight=150):
    """图像批量检测 Excel 报告，用文件名替代帧号"""
    def GetResizeImage(Img, Height):
        W, H = Img.size
        Size = (int(W * (Height / H)), Height)
        return Img.resize(Size, Image.LANCZOS)
    def PXToColW(Pixel): return (Pixel - 5) / 7
    def PXToRowH(Pixel): return (Pixel * 0.750)

    XBuffer = BIO()
    WB = Workbook()
    WS = WB.active
    WS.title = "图像检测结果"
    WS.append(["SegmentedImage", "MaskImage", "No.", "Name", "Shape", "AspectRatio", "Orientation", "Deformation", "RiskLevel", "ActionSuggestion"])
    ImgCol = {"A": 0, "B": 0}

    for OrdNum, Dict in enumerate(Data, 1):
        RowIdx = WS.max_row + 1
        
        # 预处理数据，确保所有值都是标量，防止 openpyxl 无法处理列表
        def flatten(v):
            if isinstance(v, list):
                return v[0] if v else ""
            return v

        row_data = [
            "", "", OrdNum, 
            flatten(Dict.get("Name", "")), 
            flatten(Dict.get("Shape", "-")), 
            flatten(Dict.get("AspectRatio", 0)),
            flatten(Dict.get("Orientation", 0)), 
            flatten(Dict.get("Deformation", 0)), 
            flatten(Dict.get("RiskLevel", "未知")), 
            flatten(Dict.get("ActionSuggestion", "-"))
        ]
        WS.append(row_data)
        for (ColLetter, Key) in zip(ImgCol.keys(), ["Plot", "Draw"]):
            if Dict.get(Key) is not None:
                try:
                    LoadImg = Image.fromarray(Dict[Key])
                    ResizedImage = GetResizeImage(LoadImg, ImageHeight)
                    ImageBytes = BIO()
                    ResizedImage.save(ImageBytes, "JPEG")
                    if ResizedImage.width > ImgCol[ColLetter]:
                        ImgCol[ColLetter] = ResizedImage.width
                    ImageBytes.seek(0)
                    WS.add_image(XLImage(ImageBytes), f"{ColLetter}{RowIdx}")
                except Exception as e:
                    WS.cell(RowIdx, list(ImgCol.keys()).index(ColLetter) + 1, "无图像")
            else:
                WS.cell(RowIdx, list(ImgCol.keys()).index(ColLetter) + 1, "无图像")
        WS.row_dimensions[RowIdx].height = PXToRowH(ImageHeight)
        for Cell in WS[RowIdx]:
            if Cell.column_letter not in ImgCol.keys():
                Cell.alignment = Alignment(None, "center")

    for Each in WS.columns:
        Name = Each[0].column_letter
        if Name not in ImgCol.keys():
            MaxLg = max((len(str(Cell.value)) for Cell in Each if Cell.value), default=8)
            WS.column_dimensions[Name].width = MaxLg + 3.3
        else:
            WS.column_dimensions[Name].width = PXToColW(ImgCol[Name])

    NewTable = Table(1, "ImageResultTable", f"A1:{WS.cell(WS.max_row, WS.max_column).coordinate}")
    NewTable.tableStyleInfo = TableStyleInfo("TableStyleMedium23", False, False, True, False)
    WS.add_table(NewTable)

    # 统计图表 sheet
    ChartSheet = WB.create_sheet("统计图表")
    def AddChart2Excel(Sheet, Img, Row, Col, Height=525):
        ChartRI = GetResizeImage(Image.open(Img), Height)
        XLByteIO = BIO()
        ChartRI.save(XLByteIO, "JPEG")
        XLByteIO.seek(0)
        Sheet.add_image(XLImage(XLByteIO), f"{Col}{Row}")
        Sheet.row_dimensions[Row].height = PXToRowH(ChartRI.height)
        Sheet.column_dimensions[Col].width = PXToColW(ChartRI.width)

    AddChart2Excel(ChartSheet, SpChart(Data, False), 1, "A")
    AddChart2Excel(ChartSheet, DEChart(Data, False), 3, "A")
    AddChart2Excel(ChartSheet, RiskChart(Data, False), 5, "A")

    WB.save(XBuffer)
    XBuffer.seek(0)
    return XBuffer


def GetImageResults(Data, meta=None):
    """图像批量检测 ZIP 导出（Excel + PDF）"""
    OutputZip = BIO()
    ZipBuffer = Zip.ZipFile(OutputZip, "w", Zip.ZIP_DEFLATED)
    ZipBuffer.writestr("ImageReport.xlsx", GetImageExcel(Data).read())
    ZipBuffer.writestr("ImageReport.pdf", GetImagePDF(Data, meta=meta).read())
    ZipBuffer.close()
    OutputZip.seek(0)
    return OutputZip


def GetResults(Data, meta=None):
    OutputZip = BIO()
    ZipBuffer = Zip.ZipFile(OutputZip, "w", Zip.ZIP_DEFLATED)
    ZipBuffer.writestr("SegLogs.xlsx", GetExcel(Data).read())
    ZipBuffer.writestr("SegVideo.mp4", GetVideo(Data).read())
    ZipBuffer.writestr("SegReport.pdf", GetPDF(Data, meta=meta).read())
    ZipBuffer.close()
    OutputZip.seek(0)
    return OutputZip


def GetWorkOrderPDF(orders):
    """任务列表 PDF 报告"""
    Buffer = BIO()
    Doc = SimpleDocTemplate(Buffer, pagesize=A4)
    Elements = []
    Styles = getSampleStyleSheet()

    cn_title = _cn_style(Styles['Title'], size=18)
    cn_normal = _cn_style(Styles['Normal'], size=8)

    Elements.append(Paragraph("任务验收汇总报告", cn_title))
    Elements.append(Spacer(1, 12))

    headers = ["序号", "任务编号", "标题", "风险等级", "状态", "负责人", "创建时间"]
    TableData = [[Paragraph(h, cn_normal) for h in headers]]

    for idx, order in enumerate(orders, 1):
        TableData.append([
            str(idx),
            str(order.get("任务编号", order.get("工单编号", ""))),
            Paragraph(str(order.get("标题", ""))[:50], cn_normal),
            Paragraph(str(order.get("风险", "")), cn_normal),
            Paragraph(str(order.get("状态", "")), cn_normal),
            Paragraph(str(order.get("负责人", "")), cn_normal),
            str(order.get("创建时间", ""))[:10],
        ])

    T = PDFTable(TableData, colWidths=[0.4*inch, 1.2*inch, 2.2*inch, 0.6*inch, 1.0*inch, 0.8*inch, 1.0*inch])
    T.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
    ]))

    Elements.append(T)
    Doc.build(Elements)
    Buffer.seek(0)
    return Buffer


def GetTableExcel(Data, Title="数据报表"):
    """通用表格 Excel 导出，自动适配表头"""
    XBuffer = BIO()
    WB = Workbook()
    WS = WB.active
    WS.title = Title[:31] # Excel sheet name limit
    
    if not Data:
        WS.append(["暂无数据"])
        WB.save(XBuffer)
        XBuffer.seek(0)
        return XBuffer
        
    # 获取表头 (取字典的所有键)
    Headers = list(Data[0].keys())
    WS.append(Headers)
    
    for item in Data:
        row = []
        for h in Headers:
            val = item.get(h, "")
            # 扁平化处理列表
            if isinstance(val, list):
                val = val[0] if val else ""
            row.append(val)
        WS.append(row)
        
    # 样式处理: 自动列宽
    for Each in WS.columns:
        # 排除空列
        col_cells = [Cell for Cell in Each if Cell.value]
        if col_cells:
            MaxLg = max((len(str(Cell.value).encode('gbk')) for Cell in col_cells), default=10)
            WS.column_dimensions[Each[0].column_letter].width = min(MaxLg + 2, 60)
        
    # 居中对齐
    for row in WS.iter_rows(min_row=1, max_row=WS.max_row):
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
            
    # 添加表格样式
    if WS.max_row > 1:
        tab_range = f"A1:{WS.cell(WS.max_row, WS.max_column).coordinate}"
        NewTable = Table(1, "DataTable", tab_range)
        NewTable.tableStyleInfo = TableStyleInfo("TableStyleMedium23", False, False, True, False)
        WS.add_table(NewTable)
    
    WB.save(XBuffer)
    XBuffer.seek(0)
    return XBuffer


def GetWorkOrderExcel(orders):
    """任务列表 Excel 报告"""
    XBuffer = BIO()
    WB = Workbook()
    WS = WB.active
    WS.title = "任务验收汇总"
    WS.append(["序号", "任务编号", "标题", "风险等级", "优先级", "状态", "负责人", "处理备注", "创建时间"])

    for idx, order in enumerate(orders, 1):
        WS.append([
            idx,
            order.get("任务编号", order.get("工单编号")),
            order.get("标题"),
            order.get("风险"),
            order.get("优先级"),
            order.get("状态"),
            order.get("负责人"),
            order.get("处理记录", ""),
            order.get("创建时间"),
        ])

    for Each in WS.columns:
        MaxLg = max((len(str(Cell.value)) for Cell in Each if Cell.value), default=8)
        WS.column_dimensions[Each[0].column_letter].width = min(MaxLg + 3, 50)

    for row in WS.iter_rows(min_row=2, max_row=WS.max_row):
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")

    NewTable = Table(1, "WorkOrderTable", f"A1:{WS.cell(WS.max_row, WS.max_column).coordinate}")
    NewTable.tableStyleInfo = TableStyleInfo("TableStyleMedium23", False, False, True, False)
    WS.add_table(NewTable)

    WB.save(XBuffer)
    XBuffer.seek(0)
    return XBuffer
