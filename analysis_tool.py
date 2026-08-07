import os
import copy
import time
import threading
import numpy as np
from enum import IntEnum, auto
from typing import List,Dict,Tuple,Any

from pydub import playback

from PIL import Image, ImageTk
import cv2

import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from common import MovieLoader
from det_shogiwars_move import WarsAudioDetector
from det_shogiwars_move import cfg as audio_detect_cfg


class ATEvent(IntEnum):
    OPEN_MOVIE_      = auto()
    CLICK_BTN_PLAY_START_ = auto()
    CLICK_BTN_PLAY_STOP_  = auto()
    PLAY_TO_END_          = auto()

class ATStatus:
    class StDef(IntEnum):
        INIT_            = auto()
        READY_           = auto()
        PLAYING_         = auto()
        PLAY_STOPPING_   = auto()

    def __init__(self):
        self.status_ = ATStatus.StDef.INIT_
        return

    def __eq__(self, other_stval:StDef) -> bool:
        return self.status_ == other_stval

    def __str__(self) -> str:
        str_status = "UNKNOWN"
        if self.status_ == ATStatus.StDef.INIT_:
            str_status = "INIT"
        elif self.status_ == ATStatus.StDef.READY_:
            str_status = "READY"
        elif self.status_ == ATStatus.StDef.PLAYING_:
            str_status = "PLAYING"
        elif self.status_ == ATStatus.StDef.PLAY_STOPPING_:
            str_status = "PLAY_STOPPING"
        else:
            pass
        return str_status

    def trans(self, event:ATEvent) -> bool:
        is_change_state = False

        if self.status_ == ATStatus.StDef.INIT_:
            if event == ATEvent.OPEN_MOVIE_:
                self.status_    = ATStatus.StDef.READY_
                is_change_state = True
            
        elif self == ATStatus.StDef.READY_:
            if event == ATEvent.CLICK_BTN_PLAY_START_:
                self.status_    = ATStatus.StDef.PLAYING_
                is_change_state = True

        elif self.status_ == ATStatus.StDef.PLAYING_:
            if event == ATEvent.PLAY_TO_END_:
                self.status_    = ATStatus.StDef.READY_
                is_change_state = True

            elif event == ATEvent.CLICK_BTN_PLAY_STOP_:
                self.status_    = ATStatus.StDef.PLAY_STOPPING_
                is_change_state = True
            else:
                pass

        elif self.status_ == ATStatus.StDef.PLAY_STOPPING_:
            if event == ATEvent.PLAY_TO_END_:
                self.status_    = ATStatus.StDef.READY_
                is_change_state = True

        else:
            pass

        # print(f"trans: -> {self}")

        return is_change_state

class CanvasPaintEventArg:
    def __init__(self, frame_no:int, frame_img:np.ndarray = None): # type: ignore
        self.frame_no_  = frame_no
        self.frame_img_:np.ndarray = None # type: ignore
        if frame_img is not None:
            self.frame_img_ = copy.deepcopy(frame_img)
        return


class AnalysisTool:

    MOVIE_CANVAS_W = 780
    MOVIE_CANVAS_H = 480

    BTN_LABEL_LOAD        = "Load"
    BTN_LABEL_PLAY        = "▶ play"
    BTN_LABEL_STOP        = "■ stop"

    def __init__(self, cfg:Dict[str,Any]):

        self.cfg_ = cfg
        self.status_ = ATStatus()
        self.movie_ = MovieLoader()
        self.audio_detect_ = WarsAudioDetector(audio_detect_cfg)

        # --- UI構築 ---
        self.main_win_ = tk.Tk()
        self.main_win_.title("Movie")

        self.sub_win_ = tk.Toplevel(self.main_win_)
        self.sub_win_.title("Disp Features")
        
        self.createMainWinUi()
        self.createSubWinUi()
        return

    def createMainWinUi(self):
        # --- Main Window ---

        # 終了時のハンドラ
        self.main_win_.protocol('WM_DELETE_WINDOW', self.destroy)

        # フォーカス
        self.main_win_.bind("<FocusIn>", self.onFocusMainWin)

        # キー入力
        self.main_win_.bind("<KeyPress>", self.onPressKey)

        # 動画キャンバス
        canvas_frame = ttk.Frame(self.main_win_, width=AnalysisTool.MOVIE_CANVAS_W, height=AnalysisTool.MOVIE_CANVAS_H)
        canvas_frame.pack_propagate(False) 
        canvas_frame.pack()

        self.canvas_ = tk.Label(canvas_frame)
        self.canvas_.pack(fill=tk.BOTH, expand=True)

        self.canvas_event_lock_ = threading.Lock()
        self.canvas_event_arg_  = CanvasPaintEventArg(0)
        self.canvas_.bind("<<PaintFrame>>", self.onPaintFrame)

        # コントロールボタン
        control_frame = ttk.Frame(self.main_win_)
        control_frame.pack()

        self.open_btn_ = ttk.Button(control_frame, text=AnalysisTool.BTN_LABEL_LOAD, command=self.onClickBtnOpenFile)
        self.open_btn_.grid(row=0, column=0, padx=5)

        self.play_btn_ = ttk.Button(control_frame, text=AnalysisTool.BTN_LABEL_PLAY, command=self.onClickBtnPlayStop)
        self.play_btn_.grid(row=0, column=1, padx=5)

        # ステータス表示
        self.status_label_ = ttk.Label(control_frame, text="---")
        self.status_label_.grid(row=0, column=2, padx=5)

        # シークバー
        self.seek_var_ = tk.DoubleVar()
        self.seekbar_ = ttk.Scale(self.main_win_, from_=0, to=100, orient="horizontal",
                                 variable=self.seek_var_, command=self.onMoveSeekbar)
        self.seekbar_.pack(fill="x", padx=10, pady=5)

        self.main_win_.update_idletasks() # 設定値を更新(これがないとwindowサイズ等が取得できない)
        return

    def createSubWinUi(self):
        # --- Sub Window ---

        # Main Windowの横に配置
        main_win_x = self.main_win_.winfo_rootx() # 画面左上からの位置X
        main_win_y = self.main_win_.winfo_rooty() # 画面左上からの位置Y
        main_win_w = self.main_win_.winfo_width()
        # main_win_h = self.main_win_.winfo_height()
        self.sub_win_.geometry(f"+{main_win_x+main_win_w}+{main_win_y-30}")
        # print(f"+{main_win_x+main_win_w}+{main_win_y}")

        # 終了時のハンドラ
        self.sub_win_.protocol('WM_DELETE_WINDOW', self.destroy)

        # フォーカス
        self.sub_win_.bind("<FocusIn>", self.onFocusSubWin)

        # キー入力
        self.sub_win_.bind("<KeyPress>", self.onPressKey)

        # マウスホイール
        self.sub_win_.bind("<MouseWheel>", self.onMouseWheel)

        # Audio特徴グラフ
        self.audio_feat_graph_ = FigureCanvasTkAgg(self.audio_detect_.graph_data_.fig_, 
                                                   self.sub_win_)
        self.audio_feat_graph_.draw()
        self.audio_feat_graph_.get_tk_widget().pack()


        # 値表示
        val_disp_frame = ttk.Frame(self.sub_win_)
        val_disp_frame.pack()

        self.audio_data_label_ = ttk.Label(val_disp_frame, text="---")
        self.audio_data_label_.grid(row=0, column=0, padx=5)

        self.audio_token_label_ = ttk.Label(val_disp_frame, text="---")
        self.audio_token_label_.grid(row=0, column=1, padx=5)

        return

    def mainloop(self):
        self.main_win_.mainloop()
        return

    def destroy(self):
        self.sub_win_.destroy()
        self.main_win_.destroy()
        return

    def transState(self, event:ATEvent):
        self.status_.trans(event)
        return

    def onClickBtnOpenFile(self):
        path = filedialog.askopenfilename(filetypes=[("AVI files", "*.avi"), ("All files", "*.*")])
        if path:
            self.loadMovie(path)
        return

    def loadMovie(self, path:str):

        self.movie_.load(path, int(self.cfg_["num_batch_frame"]))

        if self.movie_.isOpened() == False:
            self.status_label_.config(text=f"Can't open movie file [{path}]")

        else:
            # 読み込み
            num_frame = self.movie_.getNumFrame()

            self.seekbar_.config(to=num_frame - 1)

            path_basename = os.path.basename(path)
            self.status_label_.config(text=f"Success to load [{path_basename}]")
            self.main_win_.title(path_basename)

            # Audio特徴抽出
            print(f"Now extract feature from audio..")
            self.audio_detect_.extractFeature(self.movie_.audio_)
            self.audio_detect_.makeGraph()

            if os.path.exists("output") == False:
                os.makedirs("output", exist_ok=True)
            fpath_csv_feat = f"output/{os.path.splitext(path_basename)[0]}_feat.csv"
            fpath_csv_tok  = f"output/{os.path.splitext(path_basename)[0]}_tok.csv"
            self.audio_detect_.dumpFeature(fpath_csv_feat)
            self.audio_detect_.dumpFeatureToken(fpath_csv_tok)

            # 状態遷移
            self.transState(ATEvent.OPEN_MOVIE_)

            # 表示
            self.showFrame(self.movie_.cur_frame_no_)
            self.audio_feat_graph_.draw()

        return


    def onPaintFrame(self, event:tk.Event):
        frame_no = 0
        frame_img:np.ndarray = None # type: ignore

        with self.canvas_event_lock_:
            frame_no = self.canvas_event_arg_.frame_no_
            if self.canvas_event_arg_.frame_img_ is not None:
                frame_img = copy.deepcopy(self.canvas_event_arg_.frame_img_)

        self.showFrame(frame_no, frame_img)
        return
    
    def sendPaintFrameEvent(self, frame_no:int, frame_img:np.ndarray = None): # type: ignore
        with self.canvas_event_lock_:
            self.canvas_event_arg_ = CanvasPaintEventArg(frame_no, frame_img)
        
        self.canvas_.event_generate("<<PaintFrame>>")
        return 

    def showFrame(self, frame_index:int, frame:np.ndarray=None): # type: ignore
        if self.movie_.isOpened() == True:
            
            ret = True
            if frame is None:
                self.movie_.setCurFrame(frame_index)
                ret, frame = self.movie_.getCurFrame()

            if ret == True:

                # frameを表示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((AnalysisTool.MOVIE_CANVAS_W, AnalysisTool.MOVIE_CANVAS_H))
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas_.imgtk = imgtk # type: ignore
                self.canvas_.config(image=imgtk)

                # txt表示
                mov_fps = self.movie_.getMovieFps()
                time_cur_sec   = self.movie_.getTimeSec(frame_index)
                time_total_sec = self.movie_.getTimeSec(self.movie_.num_cap_frame_)
                self.status_label_.config(text=f"{time_cur_sec:.2f}/{time_total_sec:.2f} [sec]")

                # シークバーを動かす
                self.seek_var_.set(frame_index)

                # Audio特徴グラフに現在時刻バーを表示
                self.audio_detect_.setGraphTimeBar(time_cur_sec)
                self.audio_feat_graph_.draw()

                # Audio特徴値（現在時刻）の表示
                (val_input1, val_input2) = self.audio_detect_.getFeatureData(time_cur_sec,"input")
                (val_lpf1, val_lpf2) = self.audio_detect_.getFeatureData(time_cur_sec,"LPF")
                (val_hpf1, val_hpf2) = self.audio_detect_.getFeatureData(time_cur_sec,"HPF")
                audio_token = self.audio_detect_.getFeatureToken(time_cur_sec)

                text_str  = f" time: {time_cur_sec:.2f} [sec]\n"
                text_str += f"input: {val_input1}, {val_input2}\n"
                text_str += f"  LPF: {val_lpf1:.0f}, {val_lpf2:.0f}\n"
                text_str += f"  HPF: {val_hpf1:.0f}, {val_hpf2:.0f}"
                self.audio_data_label_.config(text=text_str)

                self.audio_token_label_.config(text=f"{audio_token}")

        return

    def onFocusMainWin(self, e:tk.Event):
        # Sub Windowを前面に
        self.sub_win_.attributes("-topmost", True)
        self.sub_win_.attributes("-topmost", False) # これをしないと他Windowがずっと前面に出れなくなる
        return

    def onFocusSubWin(self, e:tk.Event):
        # Main Windowを前面に
        self.main_win_.attributes("-topmost", True)
        self.main_win_.attributes("-topmost", False) # これをしないと他Windowがずっと前面に出れなくなる
        return

    def onPressKey(self, e:tk.Event):
        # print(f"e.keysym = {e.keysym}")

        if e.keysym == "Left":
            self.onClickBtnPrevFrame()
        elif e.keysym == "Right":
            self.onClickBtnNextFrame()
        else:
            pass
        return

    def onMouseWheel(self, e:tk.Event):
        # print(f"e.delta={e.delta}")
        self.audio_detect_.changeGraphScaleX(self.movie_.getTimeSec(), e.delta)
        self.audio_feat_graph_.draw()
        return

    def onClickBtnNextFrame(self):
        if self.movie_.isOpened() == True:
            self.movie_.nextCurFrame()
            self.showFrame(self.movie_.cur_frame_no_)

        return

    def onClickBtnPrevFrame(self):
        if self.movie_.isOpened() == True:
            self.movie_.prevCurFrame()
            self.showFrame(self.movie_.cur_frame_no_)
        return

    def onMoveSeekbar(self, val):
        if self.movie_.isOpened() == True:
            # 動画表示位置を更新
            frame_idx = int(float(val))
            self.movie_.setCurFrame(frame_idx)
            self.showFrame(self.movie_.cur_frame_no_)

        return

    def onClickBtnPlayStop(self):

        if self.movie_.isOpened() == True:
            if self.status_ == ATStatus.StDef.READY_:
                self.transState(ATEvent.CLICK_BTN_PLAY_START_)
                t = threading.Thread(target=self.threadPlayMovie, daemon=True)
                t.start()
            else:
                self.transState(ATEvent.CLICK_BTN_PLAY_STOP_)

        return

    def threadPlayMovie(self):
        self.play_btn_.config(text=AnalysisTool.BTN_LABEL_STOP) # STOPボタンにする

        sleep_time_sec = 1.0 / self.movie_.play_fps_

        try:

            while (self.status_ == ATStatus.StDef.PLAYING_):
                # 画像、Audio取得
                (batch_fno, batch_imgs, audio) = self.movie_.__next__()

                # Audio再生（バックグラウンド再生）
                audio_thread = threading.Thread(target=playback.play, args=(audio,))
                audio_thread.start()

                # 画像表示
                for fno, img in zip(batch_fno, batch_imgs):
                    self.sendPaintFrameEvent(fno, img)
                    time.sleep(sleep_time_sec)

                audio_thread.join()

        except StopIteration as e:
            pass

        # 再生終了
        self.transState(ATEvent.PLAY_TO_END_)
        self.play_btn_.config(text=AnalysisTool.BTN_LABEL_PLAY) # ボタンを元に戻す

        return

if __name__ == "__main__":

    # Analysis Tool config
    cfg = {
        "num_batch_frame" : 32,
    }

    app = AnalysisTool(cfg)
    app.mainloop()
