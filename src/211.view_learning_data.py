"""
학습 데이터 뷰어
- data/learning 폴더의 labels.txt를 읽고 학습 이미지에 xy좌표를 표시
- 한 페이지에 20개 이미지 표시, 페이지네이션 지원
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import csv


class LabelingDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("학습 데이터 뷰어 (data/learning)")
        self.root.geometry("1400x900")

        # 데이터 변수
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "learning")
        self.current_folder = self.base_path
        self.labels_data = []
        self.current_page = 0
        self.items_per_page = 20
        self.image_cache = {}

        # 포인트 표시 토글 상태
        self.show_center = tk.BooleanVar(value=True)
        self.show_floor = tk.BooleanVar(value=True)
        self.show_front = tk.BooleanVar(value=True)
        self.show_side = tk.BooleanVar(value=True)

        # UI 구성
        self.create_widgets()

        # 데이터 로드 및 표시
        self.load_labels()
        self.display_page()

    def create_widgets(self):
        # 상단 컨트롤 영역
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 작업 폴더 표시
        tk.Label(control_frame, text="작업 폴더: data/learning", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        # 페이지 정보 및 네비게이션
        tk.Label(control_frame, text="  |  ").pack(side=tk.LEFT, padx=5)

        self.prev_button = tk.Button(control_frame, text="◀ 이전", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.page_label = tk.Label(control_frame, text="페이지: 0 / 0")
        self.page_label.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(control_frame, text="다음 ▶", command=self.next_page)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # 통계 정보
        self.stats_label = tk.Label(control_frame, text="총 이미지: 0")
        self.stats_label.pack(side=tk.LEFT, padx=20)

        # 포인트 토글 버튼들
        tk.Label(control_frame, text="  |  ").pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, text="표시:").pack(side=tk.LEFT, padx=5)

        center_cb = tk.Checkbutton(control_frame, text="Center", variable=self.show_center,
                                   command=self.refresh_display, fg="red")
        center_cb.pack(side=tk.LEFT, padx=2)

        floor_cb = tk.Checkbutton(control_frame, text="Floor", variable=self.show_floor,
                                  command=self.refresh_display, fg="blue")
        floor_cb.pack(side=tk.LEFT, padx=2)

        front_cb = tk.Checkbutton(control_frame, text="Front", variable=self.show_front,
                                  command=self.refresh_display, fg="green")
        front_cb.pack(side=tk.LEFT, padx=2)

        side_cb = tk.Checkbutton(control_frame, text="Side", variable=self.show_side,
                                 command=self.refresh_display, fg="orange")
        side_cb.pack(side=tk.LEFT, padx=2)

        # 메인 스크롤 영역
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 캔버스와 스크롤바
        self.canvas = tk.Canvas(main_frame, bg="white")
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 마우스 휠 스크롤 바인딩
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_labels(self):
        """labels.txt 파일을 읽어서 데이터 로드"""
        labels_file = os.path.join(self.current_folder, "labels.txt")
        self.labels_data = []

        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.labels_data.append(row)

            self.stats_label.config(text=f"총 이미지: {len(self.labels_data)}")

        except Exception as e:
            messagebox.showerror("오류", f"labels.txt 파일 읽기 오류:\n{str(e)}")

    def display_page(self):
        """현재 페이지의 이미지들을 표시"""
        # 기존 위젯 제거
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if not self.labels_data:
            return

        # 페이지 계산
        total_pages = (len(self.labels_data) - 1) // self.items_per_page + 1
        self.page_label.config(text=f"페이지: {self.current_page + 1} / {total_pages}")

        # 페이지 버튼 상태 업데이트
        self.prev_button.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_page < total_pages - 1 else tk.DISABLED)

        # 현재 페이지의 아이템들
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.labels_data))
        page_items = self.labels_data[start_idx:end_idx]

        # 그리드로 이미지 배치 (4x5)
        cols = 4
        for idx, item in enumerate(page_items):
            row = idx // cols
            col = idx % cols
            self.create_image_frame(item, row, col)

        # 캔버스 스크롤 위치 초기화
        self.canvas.yview_moveto(0)

    def create_image_frame(self, label_data, row, col):
        """개별 이미지 프레임 생성"""
        frame = tk.Frame(self.scrollable_frame, relief=tk.RAISED, borderwidth=2, bg="lightgray")
        frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # 파일명 라벨
        filename = label_data['filename']
        name_label = tk.Label(frame, text=filename, font=("Arial", 9, "bold"), bg="lightgray")
        name_label.pack()

        # 이미지 로드 및 포인트 그리기
        image_path = os.path.join(self.current_folder, filename)

        if os.path.exists(image_path):
            try:
                # 이미지 로드
                img = Image.open(image_path)

                # 좌표 추출 (5개 컬럼 또는 11개 컬럼 지원)
                has_all_coords = 'center_x' in label_data

                floor_x = int(float(label_data['floor_x']))
                floor_y = int(float(label_data['floor_y']))

                if has_all_coords:
                    # 11개 컬럼: 모든 좌표 사용
                    center_x = int(float(label_data['center_x']))
                    center_y = int(float(label_data['center_y']))
                    front_x = int(float(label_data['front_x']))
                    front_y = int(float(label_data['front_y']))
                    side_x = int(float(label_data['side_x']))
                    side_y = int(float(label_data['side_y']))

                # 이미지에 포인트 그리기
                draw = ImageDraw.Draw(img)
                point_size = 4

                if has_all_coords:
                    # Center (빨강)
                    if self.show_center.get():
                        draw.ellipse([center_x - point_size, center_y - point_size,
                                     center_x + point_size, center_y + point_size],
                                    fill='red', outline='white', width=1)

                    # Front (녹색)
                    if self.show_front.get():
                        draw.ellipse([front_x - point_size, front_y - point_size,
                                     front_x + point_size, front_y + point_size],
                                    fill='green', outline='white', width=1)

                    # Side (노랑)
                    if self.show_side.get():
                        draw.ellipse([side_x - point_size, side_y - point_size,
                                     side_x + point_size, side_y + point_size],
                                    fill='yellow', outline='black', width=1)

                # Floor (파랑) - 항상 사용 가능
                if self.show_floor.get():
                    draw.ellipse([floor_x - point_size, floor_y - point_size,
                                 floor_x + point_size, floor_y + point_size],
                                fill='blue', outline='white', width=1)

                # 이미지 리사이즈 (썸네일)
                img.thumbnail((300, 200), Image.Resampling.LANCZOS)

                # PhotoImage로 변환
                photo = ImageTk.PhotoImage(img)

                # 이미지 라벨
                img_label = tk.Label(frame, image=photo, bg="lightgray")
                img_label.image = photo  # 참조 유지
                img_label.pack()

                # 좌표 정보 표시
                if has_all_coords:
                    info_text = (
                        f"● Center: ({center_x}, {center_y})\n"
                        f"● Floor: ({floor_x}, {floor_y})\n"
                        f"● Front: ({front_x}, {front_y})\n"
                        f"● Side: ({side_x}, {side_y})"
                    )
                else:
                    # 5개 컬럼: floor만 표시
                    info_text = f"● Floor: ({floor_x}, {floor_y})"

                info_label = tk.Label(frame, text=info_text, font=("Arial", 7),
                                    bg="lightgray", justify=tk.LEFT)
                info_label.pack()

            except Exception as e:
                error_label = tk.Label(frame, text=f"이미지 로드 실패:\n{str(e)}",
                                      fg="red", bg="lightgray")
                error_label.pack()
        else:
            error_label = tk.Label(frame, text="이미지 파일 없음", fg="red", bg="lightgray")
            error_label.pack()

    def prev_page(self):
        """이전 페이지로 이동"""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_page()

    def next_page(self):
        """다음 페이지로 이동"""
        total_pages = (len(self.labels_data) - 1) // self.items_per_page + 1
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.display_page()

    def refresh_display(self):
        """포인트 표시 토글 시 현재 페이지 다시 그리기"""
        self.display_page()


def main():
    root = tk.Tk()
    app = LabelingDataViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
