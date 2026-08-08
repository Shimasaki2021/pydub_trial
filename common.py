import os
import copy

from enum import IntEnum, auto
from typing import List,Dict,Tuple,Any

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplotFigure

import cv2
from pydub import AudioSegment, playback

class IntEnum0(IntEnum):
    """0始まりのIntEnum
    """
    def _generate_next_value_(name, start, count, last_values):
        return count  # countは0から始まるため、0始まりになる


class MovieLoader:
    """ 動画読み込み
    """
    def __init__(self, movie_fpath:str = None, play_fps=-1.0, num_batch_frame=1): # type: ignore
        """ コンストラクタ

        Args:
            movie_fpath (str, optional)    : 動画ファイルパス Defaults to None.
            play_fps (float, optional)     : 再生速度(fps). Defaults to -1.0.
            num_batch_frame (int, optional): 一度に読み込む（バッチ処理する）フレーム数[frame/cycle]. Defaults to 1.
        """
        self.cap_:cv2.VideoCapture = None # type: ignore
        self.audio_:AudioSegment = None # type: ignore
        self.cur_frame_no_    = 0
        self.num_cap_frame_   = 0
        self.play_fps_        = play_fps
        self.num_batch_frame_ = num_batch_frame
        self.frame_play_step_ = 1

        if movie_fpath is not None:
            self.load(movie_fpath, num_batch_frame, play_fps)
        return

    def release(self):
        self.cap_.release()
        self.audio_ = None # type: ignore
        self.cur_frame_no_    = 0
        self.num_cap_frame_   = 0
        self.play_fps_        = 0.0
        self.num_batch_frame_ = 0
        self.frame_play_step_ = 1
        return
    
    def load(self, movie_fpath:str, num_batch_frame=1, play_fps=-1.0):
        """ ロード

        Args:
            movie_fpath (str)              : 動画ファイルパス
            num_batch_frame (int, optional): 一度に読み込む（バッチ処理する）フレーム数[frame/cycle]. Defaults to 1.
            play_fps (float, optional)     : 再生速度(fps). Defaults to -1.0.
        """
        # 入力動画読み込み
        if self.cap_ is not None:
            self.cap_.release()
        if self.audio_ is not None:
            self.audio_ = None # type: ignore

        self.cap_ = cv2.VideoCapture(movie_fpath)

        if self.cap_ is not None:
            # Audio読み込み
            file_ext = os.path.splitext(movie_fpath)[1]
            self.audio_ = AudioSegment.from_file(movie_fpath, file_ext[1:])

        self.num_cap_frame_ = int(self.cap_.get(cv2.CAP_PROP_FRAME_COUNT))

        # 再生速度の設定
        cap_fps = self.cap_.get(cv2.CAP_PROP_FPS)
        self.play_fps_ = cap_fps
        if play_fps > 0.0:
            self.play_fps_ = play_fps

        self.frame_play_step_ = int((cap_fps + 0.1) / self.play_fps_)
        if self.frame_play_step_ < 1:
            self.frame_play_step_ = 1

        # バッチ処理するフレーム数
        self.num_batch_frame_ = num_batch_frame

        # 現在のフレーム番号
        self.cur_frame_no_ = 0
        return
    
    def isOpened(self) -> bool:
        ret = False
        if self.cap_ is not None:
            ret = self.cap_.isOpened()
        return ret

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[List[int], List[np.ndarray], AudioSegment]:
        """ 画像読み込み

        Raises:
            StopIteration: iteration終了

        Returns:
            Tuple[List[int], List[np.ndarray]]: (frame番号, 読み込んだ画像) ※バッチ処理数分のリスト
        """
        ret_batch_frame_nos:List[int]   = []
        ret_batch_imgs:List[np.ndarray] = []
        ret_audio = None

        # 動画末尾フレームまで再生していたらiteration終了
        if self.cur_frame_no_ >= self.num_cap_frame_:
            raise StopIteration()

        # フレーム読み込み
        self.cur_frame_no_ = int(self.cap_.get(cv2.CAP_PROP_POS_FRAMES)) 
        while (len(ret_batch_frame_nos) < self.num_batch_frame_) and (self.cur_frame_no_ < self.num_cap_frame_):

            img_org:np.ndarray = None # type: ignore
            (_, img_org) = self.cap_.read()

            if (img_org is not None) and (self.cur_frame_no_ % self.frame_play_step_ == 0):
                ret_batch_frame_nos.append(self.cur_frame_no_)
                ret_batch_imgs.append(copy.deepcopy(img_org))

            self.cur_frame_no_ = int(self.cap_.get(cv2.CAP_PROP_POS_FRAMES))

        # Audio読み込み
        cap_fps = self.getMovieFps()
        time_st_sec = float(ret_batch_frame_nos[0]) / cap_fps
        time_ed_sec = float(ret_batch_frame_nos[-1]) / cap_fps
        ret_audio = self.audio_[int(time_st_sec*1000.0): int(time_ed_sec*1000.0)]


        # 動画フレーム画像、フレーム番号を返す（バッチ処理分のリスト）
        return (ret_batch_frame_nos, ret_batch_imgs, ret_audio) # type: ignore

    def __len__(self) -> int:
        num_iter = int(self.num_cap_frame_ // (self.num_batch_frame_ * self.frame_play_step_))
        return num_iter

    def getNumFrame(self) -> int:
        return self.num_cap_frame_

    def getFrameSize(self) -> Tuple[int,int]:
        frame_w = int(self.cap_.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.cap_.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (frame_w, frame_h)

    def getPlayFps(self) -> float:
        return self.play_fps_
    
    def getMovieFps(self) -> float:
        cap_fps = self.cap_.get(cv2.CAP_PROP_FPS)
        return cap_fps

    def resetIter(self):
        self.setCurFrame(0)
        return

    def setCurFrame(self, frame_no:int):
        self.cur_frame_no_ = frame_no
        self.cap_.set(cv2.CAP_PROP_POS_FRAMES, float(self.cur_frame_no_))
        return
    
    def getCurFrame(self) -> Tuple[bool, np.ndarray]:
        (ret, img) = self.cap_.read()
        return (ret, img)

    def nextCurFrame(self) -> bool:
        next_frame = self.cur_frame_no_ + 1
        is_end     = (next_frame >= self.num_cap_frame_)

        self.cur_frame_no_ = min(next_frame, self.num_cap_frame_ - 1)
        return is_end

    def prevCurFrame(self) -> bool:
        prev_frame = self.cur_frame_no_ - 1
        is_start   = (prev_frame < 0)

        self.cur_frame_no_ = max(prev_frame, 0)
        return is_start

    def getTimeSec(self, frame_no=-1) -> float:
        if frame_no < 0:
            frame_no = self.cur_frame_no_
    
        mov_fps = self.getMovieFps()
        time_cur_sec   = float(frame_no) / mov_fps

        return time_cur_sec

class GraphDataRange:
    NUM_CHANGE_STEP = 30
    W_HALF_MIN = 10.0

    def __init__(self):
        self.full_min_ = 0.0
        self.full_max_ = 0.0
        self.full_w_harf_ = 0.0
        self.cur_min_ = 0.0
        self.cur_max_ = 0.0
        self.cur_w_harf_ = 0.0
        self.delta_step_ = GraphDataRange.W_HALF_MIN
        return

    def setFullRange(self, min_val, max_val, is_reset_cur:bool):
        self.full_min_ = min_val
        self.full_max_ = max_val
        self.full_w_harf_ = (max_val - min_val) / 2.0
        self.delta_step_ = (self.full_w_harf_ * 2.0) / float(GraphDataRange.NUM_CHANGE_STEP)

        if is_reset_cur == True:
            self.cur_min_ = self.full_min_
            self.cur_max_ = self.full_max_
            self.cur_w_harf_ = self.full_w_harf_
        else:
            if self.cur_min_ < self.full_min_:
                self.cur_min_ = self.full_min_
            if self.cur_max_ > self.full_max_:
                self.cur_max_ = self.full_max_
            if self.cur_w_harf_ > self.full_w_harf_:
                self.cur_w_harf_ = self.full_w_harf_
        return

    def changeCurRange(self, center_val:float, delta_sign:int):
        if delta_sign > 0:
            # 範囲を狭める（ズームイン）
            self.cur_w_harf_ -= self.delta_step_
            if self.cur_w_harf_ < GraphDataRange.W_HALF_MIN:
                self.cur_w_harf_ = GraphDataRange.W_HALF_MIN
        elif delta_sign < 0:
            # 範囲を広げる（ズームアウト）
            self.cur_w_harf_ += self.delta_step_
            if self.cur_w_harf_ > self.full_w_harf_:
                self.cur_w_harf_ = self.full_w_harf_
        else:
            # [delta_sign==0] 範囲不変
            pass

        # センタリング
        self.cur_min_ = center_val - self.cur_w_harf_
        self.cur_max_ = center_val + self.cur_w_harf_

        if self.cur_min_ < self.full_min_:
            self.cur_min_ = self.full_min_
            self.cur_max_ = self.cur_min_ + self.cur_w_harf_ * 2.0

        if self.cur_max_ > self.full_max_:
            self.cur_max_ = self.full_max_
            self.cur_min_ = self.cur_max_ - self.cur_w_harf_ * 2.0

        return

class GraphData:
    GRAPH_INCH = 100.0

    class GTYPE(IntEnum):
        GTYPE_PLOT = auto()
        GTYPE_SCATTER = auto()


    def __init__(self, graph_w_px:int, graph_h_px_one:int, num_row:int, num_col:int):
        self.num_row_ = num_row
        self.num_col_ = num_col
        self.num_ax_ = self.num_row_ * self.num_col_
        self.range_x_ = GraphDataRange()
        self.range_y_min_ = 0.0
        self.range_y_max_ = 0.0

        graph_w_inch = float(graph_w_px) / GraphData.GRAPH_INCH
        graph_h_inch = (float(graph_h_px_one) * float(num_row)) / GraphData.GRAPH_INCH

        self.fig_ = plt.figure(figsize=(graph_w_inch, graph_h_inch))

        self.ax_ = []
        idx = 0
        for _ in range(self.num_row_):
            for _ in range(self.num_col_):
                self.ax_.append( self.fig_.add_subplot(self.num_row_, self.num_col_, idx + 1) )
                idx += 1

        # 空リスト(要素数＝self.num_ax_)を作成
        self.plot_xbar_ = [None for _ in range(self.num_ax_)]

        self.cur_x_val_ = 0.0
        return

    def getIndex(self, row:int, col:int) -> int:
        return (row * self.num_col_ + col)

    def isValidRowCol(self, row:int, col:int) -> bool:
        return (row < self.num_row_) and (col < self.num_col_)

    def changeRangeX(self, x_center:float, delta_sign:int):
        self.range_x_.changeCurRange(x_center, delta_sign)
        for ax in self.ax_:
            ax.set_xlim(self.range_x_.cur_min_, self.range_x_.cur_max_)
        return

    def centeringRangeX(self, x_center:float):
        self.changeRangeX(x_center, 0) # 範囲不変、センタリングのみ実施
        return

    def setRangeX(self, x_min:float, x_max:float):
        self.range_x_.setFullRange(x_min, x_max, True)
        return

    def setRangeY(self, y_min:float, y_max:float):
        self.range_y_min_ = y_min
        self.range_y_max_ = y_max
        return

    def setData(self, row:int, col:int, 
                x_data:ArrayLike, 
                y_data:ArrayLike, 
                color:str, name:str, 
                is_fix_rangeY:bool, 
                graph_type=GTYPE.GTYPE_PLOT):
        
        if self.isValidRowCol(row,col) == True:
            idx = self.getIndex(row, col)

            # xmin = np.min(x_data)
            # xmax = np.max(x_data)
            # self.range_x_.setFullRange(xmin, xmax, True)

            self.ax_[idx].set_xlim(self.range_x_.cur_min_, self.range_x_.cur_max_)

            if is_fix_rangeY == True:
                self.ax_[idx].set_ylim(self.range_y_min_, self.range_y_max_)
            else:
                ymin = np.min(y_data)
                ymax = np.max(y_data)
                if ymax - ymin > 0.1:
                    self.ax_[idx].set_ylim(ymin, ymax)
                else:
                    self.ax_[idx].set_ylim(ymin-0.1, ymax+0.1)

            if graph_type == GraphData.GTYPE.GTYPE_SCATTER:
                self.ax_[idx].scatter(x_data, y_data, color=color, label=name, s=10)
            else:
                self.ax_[idx].plot(x_data, y_data, color=color, label=name)


            self.plot_xbar_[idx] = None
            self.setXBar(row, col, self.cur_x_val_)
        return

    def setXBar(self, row:int, col:int, x_val:float):
        if self.isValidRowCol(row,col) == True:
            idx = self.getIndex(row, col)

            self.cur_x_val_ = x_val

            x_data = [x_val, x_val]
            y_data = [self.range_y_min_, self.range_y_max_]

            if self.plot_xbar_[idx] is not None:
                self.plot_xbar_[idx].remove() # type: ignore

            (self.plot_xbar_[idx], ) = \
                self.ax_[idx].plot(x_data, y_data, color="red", alpha=0.5, label="cur time")

        return

    def setLegend(self, row:int, col:int):
        if self.isValidRowCol(row,col) == True:
            idx = self.getIndex(row, col)

            # self.ax_[idx].legend("upper left")
            # self.ax_[idx].legend()

            # legend("upper left")がなぜか上手くいかないので、手動で凡例を描画
            _, labels = self.ax_[idx].get_legend_handles_labels()
            self.ax_[idx].text(0.01, 0.93, 
                               labels[0], 
                               va="top", 
                               transform=self.ax_[idx].transAxes, 
                               color="black", 
                               fontweight="bold",
                               bbox=dict(facecolor=(1, 1, 1, 0.8), edgecolor='none'))
        return

