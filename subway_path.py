# -*- coding: utf-8 -*-
"""
소프트웨어실습 실습#8 - 지하철 최단경로 소프트웨어 (Streamlit 버전)
- subway.csv         : 역1, 역2, 소요시간(분)   예) 서울역(1),시청(1),2
- subwayLocation.csv : 역이름, 위도, 경도       예) 서울역,37.554648,126.972559
"""
import os
import csv

import pandas as pd
import folium
import streamlit as st
import streamlit.components.v1 as components


# ------------------------------------------------------------------
# 0. 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="지하철 최단경로", page_icon="🚇", layout="wide")

# app.py가 있는 폴더 기준으로 csv를 찾는다 (내 PC / Streamlit 서버 어디서든 동작)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBWAY_CSV = os.path.join(BASE_DIR, "subway.csv")
LOCATION_CSV = os.path.join(BASE_DIR, "subwayLocation.csv")

# 호선별 색상 (서울 지하철 공식 색상)
LINE_COLORS = {"1": "#0052A4", "2": "#00A84D", "3": "#EF7C1C", "4": "#00A5DE"}
TRANSFER_COLOR = "#888888"


# ------------------------------------------------------------------
# 1. 다익스트라 알고리즘 클래스
#    (교수님 코드 구조 유지 + 여러 번 호출해도 오류 나지 않도록 수정)
# ------------------------------------------------------------------
class Dijkstra:
    def __init__(self):
        self.g = {}  # {노드: {이웃노드: 비용}}

    def setEdge(self, a, b, w, bidirection=True):
        # 노드가 처음 나오면 빈 딕셔너리를 만들어 둔다
        self.g.setdefault(a, {})
        self.g.setdefault(b, {})
        self.g[a][b] = w
        if bidirection:  # 지하철은 양방향
            self.g[b][a] = w

    def _dicFilter(self, dist, unvisited):
        # 아직 방문하지 않은 노드 중 비용이 가장 작은 노드를 리턴한다
        mini = float("inf")
        curNode = None
        for key in sorted(unvisited):  # 정렬해서 매번 같은 결과가 나오게 함
            if dist[key][0] < mini:
                mini = dist[key][0]
                curNode = key
        return curNode  # 갈 수 있는 노드가 없으면 None

    def getPath(self, start, end):
        if start not in self.g or end not in self.g:
            raise ValueError("그래프에 없는 역입니다.")

        # 호출할 때마다 새로 초기화 → 여러 번 호출해도 안전
        dist = {node: [float("inf"), None] for node in self.g}  # [비용, 부모노드]
        unvisited = set(self.g)
        dist[start][0] = 0
        curNode = start

        while curNode is not None:
            unvisited.remove(curNode)          # 현재 노드 방문 처리
            if curNode == end:                 # 도착역 확정되면 종료
                break
            for node, w in self.g[curNode].items():
                # 현재 노드를 거쳐 가는 게 더 싸면 비용과 부모노드 갱신
                if node in unvisited and dist[curNode][0] + w < dist[node][0]:
                    dist[node][0] = dist[curNode][0] + w
                    dist[node][1] = curNode
            curNode = self._dicFilter(dist, unvisited)

        if dist[end][0] == float("inf"):       # 갈 수 없는 경우
            return [], []

        # 도착역에서 부모노드를 거꾸로 따라가며 경로 복원
        path = [end]
        costs = [dist[end][0]]
        node = end
        while node != start:
            node = dist[node][1]
            path.append(node)
            costs.append(dist[node][0])
        return path[::-1], costs[::-1]        # 경로, 각 역까지의 누적시간


# ------------------------------------------------------------------
# 2. 도우미 함수
# ------------------------------------------------------------------
def split_station(node):
    """'서울역(1)' → ('서울역', '1')"""
    idx = node.rfind("(")
    return node[:idx], node[idx + 1:-1]


@st.cache_resource  # csv는 처음 한 번만 읽음 (rerun 때마다 다시 읽지 않음)
def load_data():
    # 역 위치: 같은 역이 여러 줄 있으므로 처음 나온 값만 저장
    subway_loc = {}
    with open(LOCATION_CSV, "r", encoding="utf-8-sig") as f:
        for line in csv.reader(f):
            if len(line) < 3:
                continue
            if line[0] not in subway_loc:
                subway_loc[line[0]] = [float(line[1]), float(line[2])]

    # 역 사이 소요시간 → 그래프
    d = Dijkstra()
    edges = []
    with open(SUBWAY_CSV, "r", encoding="utf-8-sig") as f:
        for line in csv.reader(f):
            if len(line) < 3:
                continue
            a, b, w = line[0].strip(), line[1].strip(), int(line[2])
            d.setEdge(a, b, w)
            edges.append((a, b, w))

    stations = sorted(d.g.keys())
    return subway_loc, d, edges, stations


def station_label(node):
    name, line = split_station(node)
    return f"{name} ({line}호선)"


def make_path_map(path, costs, subway_loc):
    """최단경로 지도 (folium) 만들기"""
    coords = [subway_loc[split_station(n)[0]] for n in path]
    m = folium.Map(location=coords[0], zoom_start=12)

    # 역 사이를 빨간 선으로 연결
    folium.PolyLine(coords, color="red", weight=5, opacity=0.8).add_to(m)

    # 각 역을 호선 색 원으로 표시 (enumerate로 순서 보장 → 환승역도 정확한 색)
    for i, node in enumerate(path):
        name, line = split_station(node)
        color = LINE_COLORS.get(line, "black")
        folium.CircleMarker(
            location=coords[i], radius=7, color=color, fill=True,
            fill_color=color, fill_opacity=0.9,
            tooltip=f"{name} ({line}호선) · {costs[i]}분",
        ).add_to(m)

    # 출발 / 도착 핀
    folium.Marker(coords[0], tooltip=f"출발: {station_label(path[0])}",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], tooltip=f"도착: {station_label(path[-1])}",
                  icon=folium.Icon(color="red")).add_to(m)

    # 경로가 화면에 꽉 차게 맞춤
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m.get_root().render()


@st.cache_data  # 전체 노선도는 항상 같으므로 한 번만 만든다
def make_network_map_html():
    subway_loc, d, edges, stations = load_data()
    m = folium.Map(location=[37.55, 126.99], zoom_start=11)

    for a, b, w in edges:
        name_a, line_a = split_station(a)
        name_b, line_b = split_station(b)
        loc_a, loc_b = subway_loc[name_a], subway_loc[name_b]
        if line_a == line_b:   # 같은 호선 구간
            folium.PolyLine([loc_a, loc_b], color=LINE_COLORS.get(line_a, "black"),
                            weight=5, opacity=0.9, tooltip=f"{name_a} ↔ {name_b} : {w}분").add_to(m)

    drawn = set()
    for node in stations:
        name, line = split_station(node)
        if name in drawn:      # 환승역은 한 번만 표시
            continue
        drawn.add(name)
        lines = [split_station(n)[1] for n in stations if split_station(n)[0] == name]
        is_transfer = len(lines) > 1
        folium.CircleMarker(
            location=subway_loc[name], radius=6 if is_transfer else 4,
            color="black" if is_transfer else LINE_COLORS.get(line, "black"),
            fill=True, fill_color="white", fill_opacity=1, weight=2,
            tooltip=f"{name} ({', '.join(sorted(lines))}호선)",
        ).add_to(m)
    return m.get_root().render()


# ------------------------------------------------------------------
# 3. 화면 구성
# ------------------------------------------------------------------
subway_loc, dijkstra, edges, stations = load_data()

st.title("🚇 지하철 최단경로 찾기")

# ---- 사이드바: 출발역 / 도착역 선택 ----
with st.sidebar:
    st.header("경로 검색")
    default_start = stations.index("남태령(4)") if "남태령(4)" in stations else 0
    default_end = stations.index("종각(1)") if "종각(1)" in stations else 1
    start = st.selectbox("출발역", stations, index=default_start, format_func=station_label)
    end = st.selectbox("도착역", stations, index=default_end, format_func=station_label)
    search = st.button("최단경로 찾기", type="primary")

    st.markdown("---")
    st.markdown(" ".join(
        f"<span style='color:{c}; font-weight:bold'>● {k}호선</span>"
        for k, c in LINE_COLORS.items()), unsafe_allow_html=True)

# 버튼을 누르면 결과를 session_state에 저장 (rerun 돼도 결과 유지)
if search:
    if start == end:
        st.session_state.pop("result", None)
        st.sidebar.warning("출발역과 도착역이 같아요. 다른 역을 골라주세요!")
    else:
        path, costs = dijkstra.getPath(start, end)
        st.session_state["result"] = (path, costs)

tab1, tab2 = st.tabs(["🗺️ 최단경로", "🚇 전체 노선도"])

# ---- 탭1: 최단경로 결과 ----
with tab1:
    result = st.session_state.get("result")
    if not result:
        st.info("왼쪽에서 출발역과 도착역을 고르고 **최단경로 찾기** 버튼을 눌러주세요.")
    elif not result[0]:
        st.error("두 역을 잇는 경로가 없어요.")
    else:
        path, costs = result
        names = [split_station(n)[0] for n in path]
        lines = [split_station(n)[1] for n in path]
        transfers = sum(1 for i in range(1, len(path))
                        if names[i] == names[i - 1] and lines[i] != lines[i - 1])

        st.subheader(f"{station_label(path[0])} → {station_label(path[-1])}")
        c1, c2, c3 = st.columns(3)
        c1.metric("총 소요시간", f"{costs[-1]}분")
        c2.metric("거치는 역", f"{len(dict.fromkeys(names))}개")
        c3.metric("환승", f"{transfers}회")

        left, right = st.columns([3, 2])
        with left:
            components.html(make_path_map(path, costs, subway_loc), height=520)
        with right:
            rows = []
            for i in range(len(path)):
                note = ""
                if i == 0:
                    note = "출발"
                elif i == len(path) - 1:
                    note = "도착"
                elif names[i] == names[i - 1]:
                    note = f"🔁 {lines[i]}호선 환승"
                rows.append({"순서": i + 1, "역": names[i], "호선": f"{lines[i]}호선",
                             "누적시간(분)": costs[i], "비고": note})
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=520)

# ---- 탭2: 전체 노선도 ----
with tab2:
    st.caption("선 위에 마우스를 올리면 역 사이 소요시간이, 역 위에 올리면 호선이 보여요. (검은 테두리 = 환승역)")
    components.html(make_network_map_html(), height=650)
