"""Build the assignment report as a PDF using reportlab Platypus."""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont

pdfmetrics.registerFont(RLTTFont("Noto", "WQY.ttf"))
pdfmetrics.registerFont(RLTTFont("Noto-Bold", "WQY.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Noto", normal="Noto", bold="Noto-Bold", italic="Noto", boldItalic="Noto-Bold")
FONT = "Noto"; FONT_B = "Noto-Bold"

R = json.load(open("results.json"))
ds = R["dataset"]
bf = R["bloom"]; cms = R["cms"]

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT_B, fontSize=15, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1f3864"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT_B, fontSize=12, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2c6fbb"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName=FONT, fontSize=10, leading=15, spaceAfter=6, alignment=TA_LEFT)
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontName=FONT_B, fontSize=20, textColor=colors.HexColor("#1f3864"))
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontName=FONT, fontSize=11, alignment=TA_CENTER, textColor=colors.HexColor("#555"))
KEY = ParagraphStyle("KEY", parent=BODY, fontName=FONT, backColor=colors.HexColor("#fff4cc"), borderColor=colors.HexColor("#e0b000"), borderWidth=1, borderPadding=6, leftIndent=2, spaceAfter=8)
CAP = ParagraphStyle("CAP", parent=BODY, fontName=FONT, fontSize=8.5, textColor=colors.HexColor("#666"), alignment=TA_CENTER, spaceBefore=2)

story = []
def P(t, s=BODY): story.append(Paragraph(t, s))
def SP(h=6): story.append(Spacer(1, h))

def make_table(header, rows, colw=None):
    data = [header] + rows
    t = Table(data, colWidths=colw, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c6fbb")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), FONT_B),
        ("FONTNAME", (0,1), (-1,-1), FONT),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3fa")]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#b0b0b0")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t

# ---------------- Title ----------------
P("스트리밍 알고리즘 2종 구현 및<br/>정확도·메모리 트레이드오프 분석", TITLE)
SP(4)
P("Bloom Filter & Count-Min Sketch 비교 실험", SUB)
SP(2)
P("과제4 실험 결과 보고서", SUB)
SP(18)
P("소스코드: <link href='https://github.com/your-id/streaming-algo-experiment'>"
  "https://github.com/your-id/streaming-algo-experiment</link> "
  "<font color='#999'>(제출 전 본인 저장소 주소로 교체)</font>", CAP)
SP(14)

# ---------------- 1. Dataset ----------------
P("1. 데이터셋 설명", H1)
P(f"본 실험은 사용자 페이지 이동 로그(클릭스트림) 형태의 데이터를 사용한다. "
  f"각 레코드는 <b>(user_id, page_id, timestamp)</b> 3개 필드로 구성되며, 총 "
  f"<b>{ds['records']:,}건</b>의 이벤트를 포함한다(과제 최소 기준 100,000건 초과 충족). "
  f"페이지 인기도는 Zipf 분포를 따르도록 생성하여 소수의 페이지가 트래픽의 대부분을 차지하는 "
  f"실제 웹 로그의 특성을 재현하였다. 이는 빈도 추정 알고리즘(Count-Min Sketch)의 "
  f"heavy-hitter 추정 성능을 검증하기에 적합하다.")
SP(2)
story.append(make_table(
    ["항목", "값"],
    [["총 레코드 수", f"{ds['records']:,}"],
     ["고유 페이지 수 (page_id)", f"{ds['unique_pages']:,}"],
     ["고유 사용자 수 (user_id)", f"{ds['unique_users']:,}"],
     ["데이터 형식", "CSV (user_id, page_id, ts)"],
     ["처리 방식", "한 줄씩(streaming) 순차 처리 — 전체 적재 안 함"]],
    colw=[7*cm, 7*cm]))
SP(8)

# ---------------- 2. Algorithm overview ----------------
P("2. 선택한 알고리즘 개요", H1)
P("과제에서 추천한 2개 알고리즘을 선택하였다.", BODY)
P("2.1 Bloom Filter (원소 포함 여부 근사 판정)", H2)
P("비트 배열과 k개의 독립 해시 함수로 집합 소속 여부를 판정하는 확률적 자료구조다. "
  "원소 삽입 시 k개 해시 위치의 비트를 1로 설정하고, 조회 시 모든 위치가 1이면 '포함'으로 판정한다. "
  "<b>거짓 음성(False Negative)은 절대 발생하지 않으나</b>, 비트 충돌로 인해 거짓 양성(False Positive)이 "
  "발생할 수 있다. 이론적 FP율은 (1 - e^(-kn/m))^k 이며, 최적 해시 개수는 k = (m/n)·ln2 이다.")
P("2.2 Count-Min Sketch (항목별 빈도 근사 추정)", H2)
P("depth(d) × width(w) 크기의 카운터 행렬을 사용한다. 각 행은 독립 해시 함수를 가지며, "
  "원소가 들어오면 각 행에서 해당 해시 위치의 카운터를 증가시킨다. 빈도 조회 시 d개 카운터의 "
  "<b>최솟값</b>을 추정치로 사용한다. 항상 실제 값 이상으로 추정(과대추정만 발생)하며, "
  "오차는 확률 1-delta로 eps·N 이내에 들어온다. 여기서 w = ⌈e/eps⌉, d = ⌈ln(1/delta)⌉ 이다.")
SP(4)

# ---------------- 3. Implementation ----------------
P("3. 구현 방식", H1)
P("두 알고리즘 모두 외부 확률적 자료구조 라이브러리 없이 순수 Python으로 직접 구현하였다. "
  "해시 함수는 hashlib의 MD5/SHA-1을 이중 해싱(double hashing, h_i = h1 + i·h2)으로 결합하여 "
  "k개의 독립 해시를 생성한다. 핵심 로직 요약은 다음과 같다.", BODY)
P("• <b>Bloom Filter</b>: bytearray 비트 배열에 비트 단위 OR 연산으로 삽입, 조회는 모든 비트 AND 판정.", BODY)
P("• <b>Count-Min Sketch</b>: 행별로 salt를 달리한 해시로 위치 계산, add는 카운터 증가, "
  "estimate는 행별 카운터의 최솟값 반환.", BODY)
P("데이터는 csv.reader 제너레이터로 한 줄씩 스트리밍 처리하여 전체를 메모리에 적재하지 않는다.", BODY)
SP(4)

# ---------------- 4. Environment ----------------
P("4. 실험 환경", H1)
story.append(make_table(
    ["항목", "내용"],
    [["언어", "Python 3 (표준 라이브러리)"],
     ["주요 모듈", "hashlib, csv, time, tracemalloc, matplotlib"],
     ["메모리 측정", "tracemalloc (peak) + 자료구조 이론 크기"],
     ["시간 측정", "time.perf_counter (build time, throughput)"],
     ["반복성", "random.seed 고정으로 재현 가능"]],
    colw=[5*cm, 9*cm]))
SP(8)

# ---------------- 5. Parameters ----------------
P("5. 파라미터 설정", H1)
P("각 알고리즘에 대해 파라미터를 변화시키며 정확도·메모리·시간을 측정하였다(과제 7번 선택항목 포함).", BODY)
P("Bloom Filter: 목표 FP율 {0.10, 0.05, 0.01, 0.001}에 대해 최적 (m, k) 산출. "
  "삽입 집합은 고유 사용자의 80%, 나머지 20%는 미삽입 held-out 집합으로 두어 실제 FP율을 측정.", BODY)
P("Count-Min Sketch: eps {0.01, 0.001, 0.0005, 0.0001}, delta=0.01 고정. 전체 페이지 빈도를 추정.", BODY)
SP(6)

story.append(PageBreak())

# ---------------- 6. Accuracy ----------------
P("6. 정확도 비교 결과", H1)
P("6.1 Bloom Filter — 거짓 양성률(False Positive Rate)", H2)
rows = [[f"{r['fp_target']:.3f}", str(r['m_bits']), str(r['k']),
         f"{r['observed_fp_rate']:.4f}", str(r['false_negatives'])]
        for r in bf['sweep']]
story.append(make_table(
    ["목표 FP율", "m (비트)", "k (해시)", "실측 FP율", "거짓 음성"],
    rows, colw=[2.8*cm,2.8*cm,2*cm,2.8*cm,2.6*cm]))
SP(2)
P(f"실측 FP율이 목표값과 거의 일치하며(예: 목표 1% → 실측 "
  f"{[r for r in bf['sweep'] if r['fp_target']==0.01][0]['observed_fp_rate']*100:.2f}%), "
  f"<b>모든 설정에서 거짓 음성은 0건</b>으로 Bloom Filter의 이론적 보장(무손실 음성)이 확인되었다.", BODY)
SP(4)
story.append(Image("figs/bloom.png", width=15.5*cm, height=5.6*cm))
P("그림 1. Bloom Filter — 목표 대비 실측 FP율(좌), 목표 FP율에 따른 메모리 크기(우)", CAP)
SP(6)

P("6.2 Count-Min Sketch — 상대 오차(Relative Error)", H2)
rows = [[f"{r['eps']:.4f}", str(r['width']), str(r['depth']),
         f"{r['mean_rel_err']:.4f}", f"{r['top100_rel_err']:.5f}"]
        for r in cms['sweep']]
story.append(make_table(
    ["eps", "width", "depth", "전체 평균 오차", "Top-100 오차"],
    rows, colw=[2.4*cm,2.4*cm,2*cm,3.4*cm,3.4*cm]))
SP(2)
P("전체 페이지 평균 상대오차는 eps가 작아질수록(=width 증가) 급격히 감소한다. 특히 "
  "<b>트래픽 상위 100개 페이지(heavy hitter)의 오차는 전체 평균보다 훨씬 낮다.</b> "
  "이는 CMS가 과대추정만 하므로, 빈도가 높은 항목일수록 충돌로 인한 상대적 오차 비중이 작기 때문이다. "
  "실제 로그 분석에서 관심 대상이 대개 상위 항목임을 감안하면 매우 실용적인 특성이다.", BODY)
SP(4)
story.append(Image("figs/cms.png", width=15.5*cm, height=5.6*cm))
P("그림 2. Count-Min Sketch — 메모리 대비 정확도(좌), width에 따른 오차(우)", CAP)

story.append(PageBreak())

# ---------------- 7. Memory ----------------
P("7. 메모리 사용량 비교 결과", H1)
P(f"정확한 계산을 위한 자료구조(exact set / dict)와 스트리밍 자료구조의 메모리를 비교하였다. "
  f"멤버십 판정에서 정확한 set은 약 <b>{bf['exact_set_bytes']/1024:,.0f}KB</b>를 사용하지만, "
  f"1% FP Bloom Filter는 약 <b>{[r for r in bf['sweep'] if r['fp_target']==0.01][0]['struct_bytes']/1024:.0f}KB</b>로 "
  f"수십 배 작다. 빈도 추정에서도 정확한 dict({cms['exact_dict_bytes']/1024:,.0f}KB) 대비 "
  f"CMS(eps=1e-3)는 약 {[r for r in cms['sweep'] if r['eps']==0.001][0]['struct_bytes']/1024:.0f}KB 수준이다.", BODY)
SP(4)
story.append(Image("figs/memory.png", width=15.5*cm, height=5.6*cm))
P("그림 3. 정확한 자료구조 대비 스트리밍 자료구조 메모리", CAP)
SP(4)
P("주목할 점은 Bloom/CMS의 메모리는 <b>입력 원소 수와 무관하게 파라미터로 고정</b>된다는 것이다. "
  "데이터가 수억 건으로 늘어나도 구조 크기는 그대로이므로, 대용량 스트림에서 메모리 우위는 더욱 커진다.", BODY)
SP(6)

# ---------------- 8. Time ----------------
P("8. 처리 시간 비교 결과", H1)
rows = [["Bloom (FP=1%)",
         f"{[r for r in bf['sweep'] if r['fp_target']==0.01][0]['build_time_s']:.3f}",
         f"{[r for r in bf['sweep'] if r['fp_target']==0.01][0]['throughput']:,.0f}"],
        ["CMS (eps=1e-3)",
         f"{[r for r in cms['sweep'] if r['eps']==0.001][0]['build_time_s']:.3f}",
         f"{[r for r in cms['sweep'] if r['eps']==0.001][0]['throughput']:,.0f}"]]
story.append(make_table(["알고리즘", "구축 시간(초)", "초당 처리량(건/s)"],
                        rows, colw=[5*cm,4.5*cm,4.5*cm]))
SP(2)
P("두 알고리즘 모두 원소당 O(k) 또는 O(d)의 상수 시간 연산만 수행하므로 처리량이 안정적이다. "
  "Bloom Filter는 비트 연산만 하므로 CMS보다 빠르고, CMS는 행마다 해시·카운터 갱신이 필요해 "
  "다소 느리다. 파라미터를 키워도(정확도↑) 처리량 변화는 크지 않아, 시간보다는 "
  "메모리·정확도가 주된 트레이드오프 축임을 알 수 있다.", BODY)
SP(6)

# ---------------- 9. Pros/Cons ----------------
P("9. 알고리즘별 장단점 분석", H1)
story.append(make_table(
    ["알고리즘", "장점", "단점"],
    [["Bloom\nFilter",
      "거짓 음성 없음, 매우 작은 메모리,\n비트 연산으로 빠름, 구현 단순",
      "거짓 양성 존재, 원소 삭제·개수\n조회 불가, 빈도 정보 없음"],
     ["Count-Min\nSketch",
      "빈도 추정 가능, heavy-hitter에\n매우 정확, 메모리 고정",
      "항상 과대추정, 저빈도 항목 오차 큼,\nBloom보다 메모리·연산 많음"]],
    colw=[2.6*cm, 6*cm, 6*cm]))
SP(8)

# ---------------- 10. Conclusion ----------------
P("10. 결론", H1)
P("Bloom Filter와 Count-Min Sketch는 모두 '전체 데이터를 저장하지 않고' 고정된 메모리로 "
  "근사 질의에 답하는 데 효과적이었다. 정확도는 파라미터를 통해 메모리와 명확히 교환 가능하며, "
  "두 알고리즘 모두 정확한 자료구조 대비 수십 배의 메모리 절감을 달성하면서도 실용적 수준의 "
  "정확도를 유지하였다. 용도가 다르므로(멤버십 vs 빈도) 상황에 맞게 선택하거나 함께 사용하는 것이 바람직하다.", BODY)
SP(10)

# ---------------- 11. Final questions (highlighted) ----------------
P("11. 최종 분석 질문", H1)
P("<b>Q1. 정확도와 메모리 사이의 trade-off는?</b><br/>"
  "명확한 반비례 관계가 관찰되었다. Bloom Filter는 비트 배열 m을 키울수록 FP율이 지수적으로 "
  "감소했고(35KB에서 0.06% 수준), CMS는 width를 키울수록 상대오차가 급감했다(1MB에서 오차 거의 0). "
  "즉 메모리를 더 쓰면 정확도가 높아지나, 두 자원 모두 무한정 늘릴 수 없으므로 목표 정확도에 맞는 "
  "최소 파라미터를 선택하는 것이 핵심이다.", KEY)
P("<b>Q2. 파라미터 증가가 항상 성능 향상으로 이어졌는가?</b><br/>"
  "정확도 측면에서는 증가할수록 향상되었으나 <b>수확체감</b>이 뚜렷했다. CMS에서 eps를 1e-3→1e-4로 "
  "줄이면 메모리는 10배 늘지만 상위 항목 오차는 이미 충분히 낮아 개선 폭이 미미했다. 또한 "
  "메모리·(약간의) 처리시간 비용은 계속 증가하므로, 일정 지점 이후 파라미터 증가는 비효율적이다. "
  "즉 '항상' 성능 향상으로 이어지지는 않으며 비용 대비 효용이 줄어든다.", KEY)
P("<b>Q3. 어떤 알고리즘이 가장 실용적인가?</b><br/>"
  "목적이 다르므로 절대 우열은 없으나, <b>비용 대비 효용 면에서 Bloom Filter가 가장 실용적</b>이다. "
  "메모리·연산이 가장 적고 구현이 단순하며 거짓 음성이 없어 캐시·중복제거 등에 즉시 쓸 수 있다. "
  "다만 빈도 정보가 필요하면 CMS가 유일한 선택지이며, heavy-hitter 추정에서는 매우 정확했다.", KEY)
P("<b>Q4. 실제 서비스 로그 분석에 적용한다면?</b><br/>"
  "두 알고리즘의 <b>병행 사용</b>을 권한다. (1) '이 사용자/IP를 본 적 있는가'와 같은 중복·신규 판정에는 "
  "Bloom Filter를, (2) '어떤 페이지·검색어·에러 코드가 많이 발생하는가'와 같은 빈도·heavy-hitter "
  "분석에는 Count-Min Sketch를 사용한다. 고유 방문자 수 추정이 필요하면 HyperLogLog를 추가하는 구성이 "
  "실서비스 로그 파이프라인에 적합하다. 정확도 요구가 낮은 대시보드 지표라면 작은 파라미터로 메모리를 "
  "아끼고, 과금·보안처럼 정밀도가 중요한 곳은 파라미터를 키우거나 정확 계산을 병행한다.", KEY)

doc = SimpleDocTemplate("streaming_algo_report.pdf", pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=1.8*cm, bottomMargin=1.8*cm)
doc.build(story)
print("PDF built.")
