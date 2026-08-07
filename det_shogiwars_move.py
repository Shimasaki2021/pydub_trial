import os
import copy
from enum import IntEnum, auto
from typing import List,Dict,Tuple,Any

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplotFigure
# from matplotlib.axes._axes import Axes as MplotAxes

from pydub import AudioSegment
# from pydub import playback

import tkinter as tk
from tkinter import filedialog

from common import GraphData,IntEnum0

class AudioToken:
    TOKEN_ID_NONE = 0
    TOKEN_ID_START = 1
    VOL_EPS = 1.0

    param_duration_th0_ = 0.0
    param_duration_th1_ = 0.0
    param_duration_th2_ = 0.0
    param_rate_hpf_org_th_ = 0.0
    param_rate_lpf_org_th_ = 0.0
    param_rate_lpf_hpf_th_ = 0.0

    class TKIND(IntEnum0):
        TOK_UNKNOWN = auto()
        TOK_OPEN1 = auto()
        TOK_OPEN2 = auto()
        TOK_OPEN3 = auto()
        TOK_MOVE1 = auto()
        TOK_MOVE2 = auto()
        TOK_EFECT = auto()
        TOK_TIME  = auto()
        TOK_END1  = auto()
        TOK_END2  = auto()

    def __init__(self, id=TOKEN_ID_NONE, time_s=-1.0, time_e=-1.0, tok_kind=TKIND.TOK_UNKNOWN):
        self.id_ = id
        self.time_s_ = time_s
        self.time_e_ = time_e
        self.duration_ = time_e - time_s

        self.volsize_org_ = 0.0
        self.volsize_LPF_ = 0.0
        self.volsize_HPF_ = 0.0

        self.audio_rate_LPF_org_ = 0.0
        self.audio_rate_HPF_org_ = 0.0
        self.audio_rate_HPF_LPF_ = 0.0

        self.tok_kind_ = tok_kind
        return

    @classmethod
    def setConfigParam(cls, cfg:Dict[str,Any]):
        cls.param_duration_th0_ = float(cfg["token_kind_duration_th0"])
        cls.param_duration_th1_ = float(cfg["token_kind_duration_th1"])
        cls.param_duration_th2_ = float(cfg["token_kind_duration_th2"])
        cls.param_rate_hpf_org_th_ = float(cfg["token_kind_rate_hpf_th"])
        cls.param_rate_lpf_org_th_ = float(cfg["token_kind_rate_lpf_th"])
        cls.param_rate_lpf_hpf_th_ = float(cfg["token_kind_rate_lpf_hpf_th"])
        return

    @staticmethod
    def strTokKind(kind:TKIND) -> str:
        ret_str = "UNKNOWN"
        if kind == AudioToken.TKIND.TOK_OPEN1:
            ret_str = "OPEN1"
        elif kind == AudioToken.TKIND.TOK_OPEN2:
            ret_str = "OPEN2"
        elif kind == AudioToken.TKIND.TOK_OPEN3:
            ret_str = "OPEN3"
        elif kind == AudioToken.TKIND.TOK_MOVE1:
            ret_str = "MOVE1"
        elif kind == AudioToken.TKIND.TOK_MOVE2:
            ret_str = "MOVE2"
        elif kind == AudioToken.TKIND.TOK_EFECT:
            ret_str = "EFECT"
        elif kind == AudioToken.TKIND.TOK_TIME:
            ret_str = "TIME"
        elif kind == AudioToken.TKIND.TOK_END1:
            ret_str = "END1"
        elif kind == AudioToken.TKIND.TOK_END2:
            ret_str = "END2"
        else:
            pass

        return ret_str
    
    def __str__(self) -> str:
        if self.id_ != AudioToken.TOKEN_ID_NONE:
            ret_str  = f"token({self.id_}):\n"
            ret_str += f"  time[sec]: {self.time_s_:.2f} - "
            ret_str +=              f"{self.time_e_:.2f}, "
            ret_str +=              f"duration={self.duration_:.2f}\n"
            ret_str += f"  kind: {AudioToken.strTokKind(self.tok_kind_)}\n"
            ret_str += f"  volume: org={self.volsize_org_:.2f}, "
            ret_str +=           f"LPF={self.volsize_LPF_:.2f}, "
            ret_str +=           f"HPF={self.volsize_HPF_:.2f}\n"
            ret_str += f"  volume_rate: LPF/org={self.audio_rate_LPF_org_:.2f}, "
            ret_str +=           f"HPF/org={self.audio_rate_HPF_org_:.2f}, "
            ret_str +=           f"HPF/LPF={self.audio_rate_HPF_LPF_:.2f}\n"
        else:
            ret_str  = f"no token"

        return ret_str

    def calcTokFeatures(self,
                        audio_seg_org:np.ndarray,
                        audio_seg_LPF:np.ndarray,
                        audio_seg_HPF:np.ndarray):

        self.volsize_org_ = np.max(audio_seg_org) - np.min(audio_seg_org)
        self.volsize_LPF_ = np.max(audio_seg_LPF) - np.min(audio_seg_LPF)
        self.volsize_HPF_ = np.max(audio_seg_HPF) - np.min(audio_seg_HPF)

        self.audio_rate_LPF_org_ = 0.0
        self.audio_rate_HPF_org_ = 0.0
        self.audio_rate_HPF_LPF_ = 0.0
        if self.volsize_org_ > AudioToken.VOL_EPS:
            self.audio_rate_LPF_org_ = self.volsize_LPF_ / self.volsize_org_
            self.audio_rate_HPF_org_ = self.volsize_HPF_ / self.volsize_org_
        if self.volsize_LPF_ > AudioToken.VOL_EPS:
            self.audio_rate_HPF_LPF_ = self.volsize_HPF_ / self.volsize_LPF_

        return
    
    def analyze(self, 
                audio_seg_org:np.ndarray,
                audio_seg_LPF:np.ndarray,
                audio_seg_HPF:np.ndarray) -> TKIND:

        # 特徴の統計量算出
        self.calcTokFeatures(audio_seg_org, audio_seg_LPF, audio_seg_HPF)

        # カテゴリ分類（未分類のものをここで分類）
        if self.tok_kind_ == AudioToken.TKIND.TOK_UNKNOWN:

            if self.duration_ > AudioToken.param_duration_th0_:

                if self.duration_ < AudioToken.param_duration_th1_:
                    # [durationが短い] 通常手
                    self.tok_kind_ = AudioToken.TKIND.TOK_MOVE1
                elif self.duration_ > AudioToken.param_duration_th2_:
                    # [durationが長い] 対局終了
                    self.tok_kind_ = AudioToken.TKIND.TOK_END1
                else:

                    if self.audio_rate_HPF_org_ < AudioToken.param_rate_hpf_org_th_:
                        # [高周波成分が少ない] 時間読み上げ
                        self.tok_kind_ = AudioToken.TKIND.TOK_TIME
                    else:
                        if self.audio_rate_LPF_org_ < AudioToken.param_rate_lpf_org_th_:
                            # [高周波成分多め ＆ 低周波成分が少なめ] エフェクト
                            self.tok_kind_ = AudioToken.TKIND.TOK_EFECT
                        else:
                            # [高周波成分多め ＆ 低周波成分が多め] 駒を取ったときの効果音
                            self.tok_kind_ = AudioToken.TKIND.TOK_MOVE2

        return self.tok_kind_
  

class WarsAudioDetector:

    class GID(IntEnum0):
        GRAPH_ORG  = auto()
        GRAPH_LPF  = auto()
        GRAPH_HPF  = auto()
        GRAPH_TOK  = auto()
        GRAPH_MOVTM = auto()
        NUM_GRAPH  = auto()

    def __init__(self, cfg:Dict[str,Any]):
        self.audio_fps_ = float(cfg["audio_fps"]) 
        self.audio_sample_fps_ = self.audio_fps_

        self.audio_data_org_ :np.ndarray = None # type: ignore
        self.audio_data_LPF_ :np.ndarray = None # type: ignore
        self.audio_data_HPF_ :np.ndarray = None # type: ignore

        self.audio_token_ids_:np.ndarray = None # type: ignore
        self.audio_tokens_:List[AudioToken] = None # type: ignore
        self.move_times_:np.ndarray = None # type: ignore

        self.audio_duration_sec_ = 0.0

        self.graph_data_ = GraphData(int(cfg["graph_w_px"]), 
                                     int(cfg["graph_h_px_one"]),
                                     WarsAudioDetector.GID.NUM_GRAPH, 
                                     1)

        self.param_cutoff_LPF_th_ = float(cfg["cutoff_LPF_th"])
        self.param_cutoff_HPF_th_ = float(cfg["cutoff_HPF_th"])

        self.param_token_s_th_ = float(cfg["token_s_th"])
        self.param_token_e_th_ = float(cfg["token_e_th"])

        self.param_move_time_s_offset_ = float(cfg["move_time_start_offset"])

        # token認識用閾値の設定
        AudioToken.setConfigParam(cfg)

        return

    def release(self):
        self.audio_data_org_ = None # type: ignore
        self.audio_data_LPF_ = None # type: ignore
        self.audio_data_HPF_ = None # type: ignore
        self.audio_token_ids_ = None # type: ignore
        self.audio_tokens_ = None # type: ignore
        self.move_times_ = None # type: ignore
        return

    @staticmethod
    def minmaxPooling1D(in_data:np.ndarray, step:int) -> np.ndarray:
        # 各区間の最小値、最大値を出力

        # データ長が step の倍数でない場合は切り捨てる
        n = len(in_data) - (len(in_data) % step)

        # 区間ごとに reshape
        chunks = in_data[:n].reshape(-1, step)

        # 各区間の最小値と最大値を取得
        mins = chunks.min(axis=1)
        maxs = chunks.max(axis=1)

        # 最小値→最大値の順で並べた配列を作る
        out_data = np.column_stack([mins, maxs]).ravel()

        if n != len(in_data):
            # [データ長が step の倍数でない場合] 切り捨てた部分の最大値と最小値を取得し、末尾にマージ
            chunks = in_data[n:len(in_data)]
            min = chunks.min()
            max = chunks.max()
            out_data = np.append(out_data, [min, max])

        return out_data

    def extractTokens(self,
                      audio_data:np.ndarray) -> Tuple[np.ndarray, List[AudioToken]]:

        # 所属tokenの認識（グルーピング） 
        audio_data_abs = np.abs(audio_data)

        token_ids = np.zeros(audio_data_abs.shape[0], dtype=np.uint32)

        cur_token_id = AudioToken.TOKEN_ID_START
        is_token = False

        for idx, audio_val in enumerate(audio_data_abs):

            if is_token == False:
                # [token外]
                if audio_val > self.param_token_s_th_:
                    # [val > token開始閾値] ここからtoken開始
                    is_token = True
                    token_ids[idx] = cur_token_id # token記録(開始位置)
            else:
                # [token内]
                if audio_val < self.param_token_e_th_:
                    # [val < token終了閾値] ここでtoken終了
                    is_token = False
                    cur_token_id += 1 # 次のtoken
                else:
                    token_ids[idx] = cur_token_id # token記録

        # token毎の集合を抽出
        token_list:List[AudioToken] = []

        start_idx = -1
        end_idx = -1
        cur_token_id = AudioToken.TOKEN_ID_NONE
        cur_tokkind = AudioToken.TKIND.TOK_OPEN1 # 先頭tokenはOPEN1

        for idx, token_id in enumerate(token_ids):

            if token_id != AudioToken.TOKEN_ID_NONE:
                # [token_id != NONE]
                if (start_idx == -1) and (end_idx == -1):
                    # [start未認識] token開始
                    start_idx = idx
                    cur_token_id = token_id

            else:
                # [token_id == NONE]

                if (start_idx != -1) and (end_idx == -1):
                    # [end未認識] token終了
                    end_idx = idx

                    # token登録
                    token_list.append(AudioToken(cur_token_id,
                                                 self.convIdx2Time(start_idx),
                                                 self.convIdx2Time(end_idx),
                                                 cur_tokkind)) 

                    # 未認識状態に戻す
                    start_idx = -1
                    end_idx = -1

                    if cur_tokkind == AudioToken.TKIND.TOK_OPEN1:
                        cur_tokkind = AudioToken.TKIND.TOK_OPEN2 # OPEN1の後はOPEN2
                    elif cur_tokkind == AudioToken.TKIND.TOK_OPEN2:
                        cur_tokkind = AudioToken.TKIND.TOK_OPEN3 # OPEN2の後はOPEN3
                    elif cur_tokkind == AudioToken.TKIND.TOK_OPEN3:
                        cur_tokkind = AudioToken.TKIND.TOK_UNKNOWN # OPEN2の後は未定(後で分類)

        return (token_ids, token_list)

    def analyzeTokens(self, token_list:List[AudioToken]) -> Tuple[np.ndarray,List[AudioToken]]:

        cnt_move1 = 0
        ave_duration_move1 = 0.0

        # ノイズtoken(＝durationが短いtoken)消去
        token_list_org = copy.deepcopy(token_list)
        token_list = [token for token in token_list_org 
                            if token.duration_ > AudioToken.param_duration_th0_]

        # token分類
        for idx, token in enumerate(token_list):
            
            token_idx_s = self.convTime2Idx(token.time_s_)
            token_idx_e = self.convTime2Idx(token.time_e_)

            audio_seg_org = self.audio_data_org_[token_idx_s:token_idx_e]
            audio_seg_LPF = self.audio_data_LPF_[token_idx_s:token_idx_e]
            audio_seg_HPF = self.audio_data_HPF_[token_idx_s:token_idx_e]

            tok_kind= token.analyze(audio_seg_org, audio_seg_LPF, audio_seg_HPF)

            if tok_kind == AudioToken.TKIND.TOK_MOVE1:
                ave_duration_move1 += token.time_e_ - token.time_s_
                cnt_move1 += 1

        # token分類2
        #   (EFECT or MOVE2)の末尾に、MOVE1が含まれていれば分離
        if cnt_move1 > 0:
            ave_duration_move1 /= float(cnt_move1)

        insert_toks:List[Tuple[int,AudioToken]] = []

        for idx, token in enumerate(token_list):
            if      (   (token.tok_kind_ == AudioToken.TKIND.TOK_EFECT) \
                     or (token.tok_kind_ == AudioToken.TKIND.TOK_MOVE2)) \
                and (token.duration_ > ave_duration_move1):

                ptok_time_s = token.time_e_ - ave_duration_move1
                ptok_time_e = token.time_e_
                ptok_idx_s = self.convTime2Idx(ptok_time_s)
                ptok_idx_e = self.convTime2Idx(ptok_time_e)

                audio_seg_org = self.audio_data_org_[ptok_idx_s:ptok_idx_e]
                audio_seg_LPF = self.audio_data_LPF_[ptok_idx_s:ptok_idx_e]
                audio_seg_HPF = self.audio_data_HPF_[ptok_idx_s:ptok_idx_e]

                ptok_move1 = AudioToken(token.id_, 
                                        ptok_time_s,
                                        ptok_time_e,
                                        AudioToken.TKIND.TOK_MOVE1)

                ptok_move1.calcTokFeatures(audio_seg_org, audio_seg_LPF, audio_seg_HPF)

                if ptok_move1.audio_rate_HPF_LPF_ > AudioToken.param_rate_lpf_hpf_th_:
                    # [EFECT末尾の高周波/低周波割合が高い] EFECT or MOVE2末尾＝MOVE1
                    token.time_e_ = ptok_time_s
                    token.duration_ = token.time_e_ - token.time_s_

                    insert_toks.append((idx+1, ptok_move1))

        # 分離したtokenを挿入
        #   挿入後は後ろのindexがずれるので、indexが大きい方から順に挿入
        for insert_tok in reversed(insert_toks):
            loc = insert_tok[0]
            tok = insert_tok[1]
            token_list.insert(loc, tok)

        # ID振りなおし
        self.audio_token_ids_[:] = AudioToken.TOKEN_ID_NONE

        for idx, token in enumerate(token_list):
            token.id_ = idx + AudioToken.TOKEN_ID_START

            idx_time_s = self.convTime2Idx(token.time_s_)
            idx_time_e = self.convTime2Idx(token.time_e_)
            self.audio_token_ids_[idx_time_s:idx_time_e] = token.id_

        # 手を指した時刻を算出
        move_times = []
        for token in token_list:
            if (token.tok_kind_ == AudioToken.TKIND.TOK_OPEN3) \
                or (token.tok_kind_ == AudioToken.TKIND.TOK_MOVE1) \
                or (token.tok_kind_ == AudioToken.TKIND.TOK_MOVE2) \
                or (token.tok_kind_ == AudioToken.TKIND.TOK_EFECT) \
                or (token.tok_kind_ == AudioToken.TKIND.TOK_END1):

                move_time = token.time_s_

                if token.tok_kind_ == AudioToken.TKIND.TOK_OPEN3:
                    move_time += self.param_move_time_s_offset_

                move_times.append(move_time)

        return (np.array(move_times), token_list)


    def extractFeature(self, audio:AudioSegment) -> np.ndarray:

        self.audio_data_org_ = np.array(audio.get_array_of_samples())[::audio.channels]

        if self.audio_data_org_ is not None:
            self.audio_data_org_ = self.audio_data_org_.astype(np.float64)

            self.audio_duration_sec_ = audio.duration_seconds

            dt = 1.0/float(audio.frame_rate)  # サンプリング時間

            # DFT
            N = len(self.audio_data_org_)
            X = np.fft.fft(self.audio_data_org_)
            f = np.fft.fftfreq(N, dt) # Xのindexに対応する周波数のnumpy配列を取得

            # ローパスフィルタ
            X_LPFed = X.copy()
            X_LPFed[(f > self.param_cutoff_LPF_th_) | (f < -self.param_cutoff_LPF_th_)] = 0.0 # カットオフ周波数より大きい周波数成分を0に
            self.audio_data_LPF_ = np.real(np.fft.ifft(X_LPFed))

            # ハイパスフィルタ
            X_HPFed = X.copy()
            X_HPFed[((f > 0) & (f < self.param_cutoff_HPF_th_)) | ((f < 0) & (f > -self.param_cutoff_HPF_th_))] = 0.0 #カットオフ周波数より小さい周波数成分を0に
            self.audio_data_HPF_ = np.real(np.fft.ifft(X_HPFed))

            # データ間引き (audio.frame_rate(44100)[fps] → self.audio_fps_ [fps])
            data_step = int(float(audio.frame_rate) / self.audio_fps_)
            self.audio_data_org_ = WarsAudioDetector.minmaxPooling1D(self.audio_data_org_, data_step)
            self.audio_data_LPF_ = WarsAudioDetector.minmaxPooling1D(self.audio_data_LPF_, data_step)
            self.audio_data_HPF_ = WarsAudioDetector.minmaxPooling1D(self.audio_data_HPF_, data_step)
            # self.audio_data_org_ = self.audio_data_org_[::data_step]
            # self.audio_data_LPF_ = self.audio_data_LPF_[::data_step]
            # self.audio_data_HPF_ = self.audio_data_HPF_[::data_step]

            # データ間引き後のfps: min-max pooling実施 → 2倍のfpsでサンプリング
            self.audio_sample_fps_ = self.audio_fps_ * 2.0  

            # token認識＆手を指した時刻を算出
            (self.audio_token_ids_, self.audio_tokens_) = self.extractTokens(self.audio_data_org_)
            (self.move_times_, self.audio_tokens_) = self.analyzeTokens(self.audio_tokens_)

        return self.move_times_

    def convTime2Idx(self, time_cur_sec:float) -> int:
        idx = int(time_cur_sec * self.audio_sample_fps_)
        return idx

    def convIdx2Time(self, idx:int) -> float:
        time_cur_sec = float(idx) / self.audio_sample_fps_
        return time_cur_sec

    def getFeatureData(self, time_cur_sec:float, feature_name:str) -> Tuple[float,float]:

        val0 = 0.0
        val1 = 0.0

        if (self.audio_data_org_ is not None) \
            and (self.audio_data_LPF_ is not None) \
            and (self.audio_data_HPF_ is not None):

            idx = self.convTime2Idx(time_cur_sec)

            if feature_name == "input":
                val0 = self.audio_data_org_[idx + 0]
                val1 = self.audio_data_org_[idx + 1]
            elif feature_name == "LPF":
                val0 = self.audio_data_LPF_[idx + 0]
                val1 = self.audio_data_LPF_[idx + 1]
            elif feature_name == "HPF":
                val0 = self.audio_data_HPF_[idx + 0]
                val1 = self.audio_data_HPF_[idx + 1]
            else:
                pass

        return (val0, val1)

    def getFeatureToken(self, time_cur_sec:float) -> AudioToken:

        ret_token = AudioToken()

        if (self.audio_token_ids_ is not None) \
            and (self.audio_tokens_ is not None):

            idx = self.convTime2Idx(time_cur_sec)
            token_id = self.audio_token_ids_[idx]

            if token_id != AudioToken.TOKEN_ID_NONE:
                ret_token = self.audio_tokens_[token_id - AudioToken.TOKEN_ID_START]

        return ret_token

    def makeGraph(self) -> MplotFigure:
        if (self.audio_data_org_ is not None) \
            and (self.audio_data_LPF_ is not None) \
            and (self.audio_data_HPF_ is not None):

            tms = 0.0
            tme = self.audio_duration_sec_
            tm = np.linspace(tms, tme, len(self.audio_data_org_), endpoint=True) # 時間numpy配列を作成

            self.graph_data_.setRangeX(tms, tme)
            self.graph_data_.setRangeY(np.min(self.audio_data_org_),
                                       np.max(self.audio_data_org_))

            self.graph_data_.setData(WarsAudioDetector.GID.GRAPH_ORG,0,tm,self.audio_data_org_,"black","input",True) 
            self.graph_data_.setLegend(WarsAudioDetector.GID.GRAPH_ORG,0)

            self.graph_data_.setData(WarsAudioDetector.GID.GRAPH_LPF,0,tm,self.audio_data_LPF_,"blue","LPF",True) 
            self.graph_data_.setLegend(WarsAudioDetector.GID.GRAPH_LPF,0)

            self.graph_data_.setData(WarsAudioDetector.GID.GRAPH_HPF,0,tm,self.audio_data_HPF_,"orange","HPF",True)
            self.graph_data_.setLegend(WarsAudioDetector.GID.GRAPH_HPF,0)

            if self.audio_token_ids_ is not None:
                self.graph_data_.setData(WarsAudioDetector.GID.GRAPH_TOK,0,tm,self.audio_token_ids_,"purple","token id",False)
                self.graph_data_.setLegend(WarsAudioDetector.GID.GRAPH_TOK,0)

            if self.move_times_ is not None:
                y_data = np.ones(self.move_times_.shape[0])
                self.graph_data_.setData(WarsAudioDetector.GID.GRAPH_MOVTM,0,
                                        self.move_times_, y_data,
                                        "green","move time", False,
                                        GraphData.GTYPE.GTYPE_SCATTER)
                self.graph_data_.setLegend(WarsAudioDetector.GID.GRAPH_MOVTM,0)

        return self.graph_data_.fig_

    def setGraphTimeBar(self, time_cur_sec:float):
        for graph_id in range(WarsAudioDetector.GID.NUM_GRAPH):
            self.graph_data_.setXBar(graph_id,0,time_cur_sec)

        self.graph_data_.centeringRangeX(time_cur_sec)
        return

    def changeGraphScaleX(self, x_center:float, delta_sign:int):
        self.graph_data_.changeRangeX(x_center, delta_sign)
        return

    def dumpFeature(self, fpath:str):
        if (self.audio_data_org_ is not None) \
            and (self.audio_data_LPF_ is not None) \
            and (self.audio_data_HPF_ is not None):

            with open(fpath,"w") as fp:
                # ヘッダ
                line_str  = f"time[sec],data_org,data_LPF,data_HPF\n"
                fp.write(line_str)

                for idx,(org,lpf,hpf) in enumerate(zip(self.audio_data_org_, 
                                                       self.audio_data_LPF_, 
                                                       self.audio_data_HPF_)):

                    time_cur = self.convIdx2Time(idx)
                    line_str  = f"{time_cur},"
                    line_str += f"{org},"
                    line_str += f"{lpf},"
                    line_str += f"{hpf}\n"
                    fp.write(line_str)

        return

    def dumpFeatureToken(self, fpath:str):

        if self.audio_tokens_ is not None:

            with open(fpath,"w") as fp:
                # ヘッダ
                line_str  = f"id,time_s[sec],time_e[sec],duration[sec]"
                line_str += f",vol.org,vol.LPF,vol.HPF"
                line_str += f",rate:LPF/org,rate:HPF/org,rate:HPF/LPF"
                line_str += f",kind\n"
                fp.write(line_str)

                # データ
                for token in self.audio_tokens_:
                    line_str  = f"{token.id_},{token.time_s_:.2f},{token.time_e_:.2f},{token.duration_:.2f}"
                    line_str += f",{token.volsize_org_:.2f},{token.volsize_LPF_:.2f},{token.volsize_HPF_:.2f}"
                    line_str += f",{token.audio_rate_LPF_org_},{token.audio_rate_HPF_org_},{token.audio_rate_HPF_LPF_}"
                    line_str += f",{AudioToken.strTokKind(token.tok_kind_)}\n"
                    fp.write(line_str)

        return



# WarsAudioDetectorのconfig
#   将棋ウォーズ対局動画専用（BGM OFF, 効果音のみON ）parameter
cfg = {
    "audio_fps"  :  120,
    "cutoff_LPF_th" :  5.0e2,    # LPF閾値
    "cutoff_HPF_th" : 70.0e2,    # HPF閾値
    "token_s_th": 20.0,  # token分割閾値(token開始)
    "token_e_th":  5.0,  # token分割閾値(token終了)

    "token_kind_duration_th0": 0.3, # token分類閾値: duration0(ノイズ分類)
    "token_kind_duration_th1": 0.5, # token分類閾値: duration1
    "token_kind_duration_th2": 4.0, # token分類閾値: duration2
    "token_kind_rate_hpf_th" : 0.1, # token分類閾値: 高周波割合(hpf/org)
    "token_kind_rate_lpf_th" : 0.7, # token分類閾値: 低周波割合(lpf/org)
    "token_kind_rate_lpf_hpf_th" : 0.6, # token分類閾値: 高周波,低周波割合(hpf/lpf)

    "move_time_start_offset" : 0.51, # 手を指した時刻算出: 開始token(OPEN3)時刻offset

    "graph_w_px" : 800,
    "graph_h_px_one" : 150,
}

if __name__ == "__main__":

    # メインウィンドウは非表示
    root = tk.Tk()
    root.withdraw()

    # ファイルダイアログでaviファイル選択
    file_type = [("aviファイル","*.avi")] 
    movie_fpath = filedialog.askopenfilename(filetypes = file_type) 

    if movie_fpath != "":
        print(f"loading {movie_fpath}..")

        audio_detect = WarsAudioDetector(cfg)

        # Audio読み込み
        file_ext = os.path.splitext(movie_fpath)[1]
        audio = AudioSegment.from_file(movie_fpath, file_ext[1:])

        if audio is not None:
            print(f"extract feature..")
            audio_detect.extractFeature(audio)

            audio_detect.makeGraph()
            plt.show(block=True)








