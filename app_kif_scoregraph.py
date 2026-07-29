import sys
import io
import tkinter as tk
from tkinter import filedialog, simpledialog
from datetime import datetime
from datetime import timedelta
import re
import copy
import math
import threading
import matplotlib.pyplot as plt
from matplotlib import animation
from time import sleep
import os
from typing import List,Tuple,Dict,Any
import cv2
import numpy as np

from common import MovieLoader
from det_shogiwars_move import WarsAudioDetector
from det_shogiwars_move import cfg as audio_detect_cfg

GRAPH_IMG_PREFIX = "F"

STAT_PARSE_HEADER:int=0
STAT_PARSE_CONTENTS_SENTE:int=1
STAT_PARSE_CONTENTS_GOTE:int=2

def transStat(stat_parse:int) -> int:
    if stat_parse == STAT_PARSE_HEADER:
        stat_parse = STAT_PARSE_CONTENTS_SENTE
    elif stat_parse == STAT_PARSE_CONTENTS_SENTE:
        stat_parse = STAT_PARSE_CONTENTS_GOTE
    else:
        stat_parse = STAT_PARSE_CONTENTS_SENTE
    
    return stat_parse

class MoveRecommend:
    NUM_MAX:int = 3

    def __init__(self):
        self.is_exist:bool = False
        self.score:int = 0
        self.moves:str = ""
        return
    
    def clear(self):
        self.is_exist = False
        self.score = 0
        self.moves = ""
        return
    
    def parse(self, data_line:str, is_sente:bool):
        # 以下文字列（例）から、スコア、推奨手（最大NUM_MAX）を抽出
        #   *#推奨手[244] ▲５六歩  △同　歩  ▲同　銀  △５四銀直  ▲５五歩打  △６三銀 ...
        str_score:List[str] = re.findall(r"\[([^*]*)\]", data_line)
        self.score = int(str_score[0])
        if is_sente == False:
            self.score = -self.score
        
        moves_tmp:List[str] = data_line.split("]")[1].split(" ")

        num_moves:int = 0
        self.moves = ""
        for move in moves_tmp:
            if num_moves < MoveRecommend.NUM_MAX:
                if move != "":
                    self.moves += move
                    num_moves += 1
        
        self.is_exist = True

        return

class Record:
    def __init__(self, no_val=-1):
        self.no              = no_val
        self.move            = ""
        self.move_kind       = ""
        self.move_extinfo    = ""
        self.move_recommend  = MoveRecommend()
        self.is_sente        = False
        self.is_players_move = False
        self.is_bad_move     = False

        self.score            = 0
        self.elapsed_time_str = ""
        self.elapsed_time_sec = 0.0
        self.thinking_time    = 0.0
        self.disp_time_s      = 0.0
        self.disp_time_s_org  = 0.0
        self.disp_time_e      = 0.0
        self.cnt_seq_0sec_    = 0

        self.time_s:datetime = datetime.strptime("00:00:00","%H:%M:%S")
        self.time_max:datetime = datetime.strptime("00:10:00","%H:%M:%S")
        self.time_rest:timedelta = self.time_max - self.time_s
        return

    def clear(self, no_val=-1):
        self.no              = no_val
        self.move            = ""
        self.move_kind       = ""
        self.move_extinfo    = ""
        self.move_recommend.clear()
        self.is_sente        = False
        self.is_players_move = False
        self.is_bad_move     = False

        self.score            = 0
        self.elapsed_time_str = ""
        self.elapsed_time_sec = 0.0
        self.thinking_time    = 0.0
        self.disp_time_s      = 0.0
        self.disp_time_s_org  = 0.0
        self.disp_time_e      = 0.0
        self.cnt_seq_0sec_    = 0

        self.time_s    = datetime.strptime("00:00:00","%H:%M:%S")
        self.time_max  = datetime.strptime("00:10:00","%H:%M:%S")
        self.time_rest = self.time_max - self.time_s
        return
        
    def isValid(self) -> bool:
        is_valid:bool = False
        if self.no != -1:
            is_valid = True
        
        return is_valid

    def setBaseInfo(self, data_line:str, is_sente:bool, is_player_sente:bool):
        self.no = int(data_line[0:4].replace(" ",""))
        self.is_sente = is_sente

        self.is_players_move = False
        if self.is_sente == is_player_sente:
            self.is_players_move = True

        cur_move = data_line[5:8]
        if data_line[8] == "打" or data_line[8] == "成" or data_line[7] == "成":
            cur_move = data_line[5:9]
        
        if self.is_sente == True:
            self.move = "▲" + cur_move
        else:
            self.move = "△" + cur_move

        # 「1 ７六歩(77)   ( 0:01/00:00:01)」の「/00:00:01)」部分を抽出
        str_time = re.findall(r"\/([^*]*)\)", data_line)
        time_cur = datetime.strptime(str_time[0],"%H:%M:%S")
        self.elapsed_time_str = time_cur.strftime("%M:%S")
        self.elapsed_time_sec = (time_cur - self.time_s).total_seconds()
        self.time_rest = self.time_max - time_cur

        # 「1 ７六歩(77)   ( 0:01/00:00:01)」の「( 0:01/」部分を抽出
        str_time = re.findall(r"\( ([^*]*)\/", data_line)
        time_cur = datetime.strptime(str_time[0],"%M:%S")
        self.thinking_time = (time_cur - self.time_s).total_seconds()

        return


    def setScore(self, data_line:str, is_player_sente:bool):
        str_score = re.findall(r"\[([^*]*)\]", data_line)
        self.score = int(str_score[0]) 
        if is_player_sente == False:
            self.score = -self.score # 対局playerが後手の場合は符号反転
        
        self.move_kind = ""
        if "好手" in data_line:
            self.move_kind = "好手"
            self.is_bad_move = False
            
        elif "悪手" in data_line:
            self.move_kind = "悪手"
            self.is_bad_move = True

        elif "疑問" in data_line:
            self.move_kind = "疑問"
            self.is_bad_move = True
        else:
            self.move_kind = ""
            self.is_bad_move = False
        
        return


    def setExtInfo(self, data_line:str):
        #print("setExtInfo: ",data_line)
        # if data_line.find("#") == -1:
        idx = data_line.find("戦型")
        if idx != -1:
            #print("find!!  ",str(idx+len("戦型：")), ", ", data_line[idx + len("戦型："):])
            self.move_extinfo = copy.deepcopy(data_line[idx + len("戦型："):len(data_line)-1])
        else:
            idx = data_line.find("戦法")
            if idx != -1:
                self.move_extinfo = copy.deepcopy(data_line[idx + len("戦法"):len(data_line)-1])
            else:
                idx = data_line.find("囲い")
                if idx != -1:
                    #print("find!!  ",str(idx+len("囲い：")), ", ", data_line[idx + len("囲い："):])
                    self.move_extinfo = copy.deepcopy(data_line[idx + len("囲い："):len(data_line)-1])
                else:
                    idx = data_line.find("手筋")
                    if idx != -1:
                        self.move_extinfo = copy.deepcopy(data_line[idx + len("手筋："):len(data_line)-1])
        return

    def __str__(self) -> str:
        ret_str = "[" + str(self.no) + "] " + self.move + ": score=" + str(self.score)
        if self.move_extinfo != "":
            ret_str += " " + self.move_extinfo
        if self.move_kind != "":
            ret_str += " (" + self.move_kind + ")"
        return ret_str

class RecordSet:

    RESULT_UNKNOWN       = 0
    RESULT_WIN_CHECKMATE = 1 # 勝ち（詰み）
    RESULT_WIN_GIVEUP    = 2 # 勝ち（投了）
    RESULT_WIN_TIMEOUT   = 3 # 勝ち（時間切れ）

    RESULT_LOSE_CHECKMATE = -1 # 負け（詰み）
    RESULT_LOSE_GIVEUP    = -2 # 負け（投了）
    RESULT_LOSE_TIMEOUT   = -3 # 負け（時間切れ）

    def __init__(self):
        self.records:List[Record] = []
        self.cur_idx         = 0
        self.is_player_sente = True
        self.is_reverse_     = False
        self.result_         = RecordSet.RESULT_UNKNOWN
        return

    def initIter(self, is_reverse=False):
        self.is_reverse_ = is_reverse
        if is_reverse == False:
            self.cur_idx = 0
        else:
            self.cur_idx = len(self.records) - 1
        return
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[Record,Record]:
        cur_val = Record()
        next_val = Record()

        if self.is_reverse_ == False:
            # 順scan
            if self.cur_idx >= len(self.records):
                raise StopIteration()
            elif self.cur_idx >= len(self.records) - 1:
                cur_val = self.records[self.cur_idx]
                self.cur_idx += 1
            else:
                cur_val = self.records[self.cur_idx]
                self.cur_idx += 1
                next_val = self.records[self.cur_idx]

        else:
            # 逆scan
            if self.cur_idx < 0:
                raise StopIteration()
            elif self.cur_idx == 0:
                cur_val = self.records[self.cur_idx]
                self.cur_idx -= 1
            else:
                cur_val = self.records[self.cur_idx]
                self.cur_idx -= 1
                next_val = self.records[self.cur_idx]

        return (cur_val,next_val)

    def __len__(self) -> int:
        return len(self.records)
    
    def getCurRecord(self) -> Record:
        return self.records[self.cur_idx]
    
    def getNextRecord(self) -> Record:
        next_val = Record()
        if self.cur_idx < len(self.records) - 1:
            next_val = self.records[self.cur_idx + 1]
        return next_val

    def getTailRecord(self) -> Record:
        return self.records[-1]

    def addRecord(self, record:Record):
        self.records.append(copy.deepcopy(record))
        return

    def getTotalElapsedTime(self) -> float:
        total_time = 0.0
        if len(self.records) >= 2:
            total_time = self.records[-1].elapsed_time_sec + self.records[-2].elapsed_time_sec
        elif len(self.records) == 1:
            total_time = self.records[-1].elapsed_time_sec
        return total_time
    
    def analysisGameResult(self) -> int:
        result = RecordSet.RESULT_UNKNOWN

        if len(self) > 1:
            tail_record  = self.getTailRecord()

            if "詰み" in tail_record.move:
                if tail_record.score > 0:
                    result = RecordSet.RESULT_WIN_CHECKMATE
                else:
                    result = RecordSet.RESULT_LOSE_CHECKMATE

            elif "投了" in tail_record.move:
                if tail_record.score > 0:
                    result = RecordSet.RESULT_WIN_GIVEUP
                else:
                    result = RecordSet.RESULT_LOSE_GIVEUP

            elif "切れ" in tail_record.move:
                if tail_record.score > 0:
                    result = RecordSet.RESULT_WIN_TIMEOUT
                else:
                    result = RecordSet.RESULT_LOSE_TIMEOUT

            else:
                pass

        self.result_ = result

        return result

    def isGameWin(self) -> bool:
        is_game_win = False
        if self.result_ > 0:
            is_game_win = True
        return is_game_win

    def print(self):
        for cur_record in self.records:
            # cur_record.print()
            print(cur_record)
        return

    def adjustTime(self, last_mov_time_s_str:str):
        # elapsed_time_sec（Total消費時間）を使って開始／終了時刻を算出2

        # 最終手の開始時刻（ユーザー入力）
        last_mov_time_s     = datetime.strptime(last_mov_time_s_str,"%M:%S")
        time_0              = datetime.strptime("00:00:00","%H:%M:%S")
        last_mov_time_s_sec = (last_mov_time_s - time_0).total_seconds()

        #  1. elapsed_time_sec（Total消費時間）を使って開始時刻を算出
        #      thinking_time=0の場合、開始時刻を+αずらす
        for idx, cur_record in enumerate(self.records):
            if idx == 0:
                cur_record.disp_time_s_org = 0.0
            else:
                cur_record.disp_time_s_org = self.records[idx].elapsed_time_sec + self.records[idx-1].elapsed_time_sec

            if math.isclose(cur_record.thinking_time, 0.0):
                cur_record.disp_time_s_org += 0.33

        #  2. disp_time_sに一定倍率（＝最終手の開始時刻が合う倍率）をかける（なぜか動画と一定倍率のずれがあるようなのでかける）
        # DISP_TIME_S_DIFF_RATE = 1.086
        # DISP_TIME_S_DIFF_RATE = 1.0
        # DISP_TIME_S_DIFF_RATE = 1.05
        # DISP_TIME_S_DIFF_RATE = 1.0256
        DISP_TIME_S_DIFF_RATE = last_mov_time_s_sec / self.records[-2].disp_time_s_org
        self.initIter(is_reverse=False)
        for cur_record, _ in self:
            cur_record.disp_time_s = cur_record.disp_time_s_org * DISP_TIME_S_DIFF_RATE

        #  3. 終了時刻＝次の手の開始時刻
        self.initIter(is_reverse=False)
        for cur_record, nxt_record in self:
            cur_record.disp_time_e = nxt_record.disp_time_s

        tail_record = self.getTailRecord()
        tail_record.disp_time_e = tail_record.disp_time_s + 10.0 

        return
    
    def createTime(self, ave_elapsed_time_sec:float):
        # 手毎の開始／終了時刻を作成（均等に時間を配分(ave_elapsed_time_sec)）
        cur_disp_time_s = 0.0
        for cur_record in self.records:
            cur_record.disp_time_s = cur_disp_time_s

            cur_disp_time_s += ave_elapsed_time_sec
            cur_record.disp_time_e = cur_disp_time_s

        return

    def assignTime(self, move_times:np.ndarray, elapsed_time_for_rest:float, mov_total_sec:float):
        # move_times[]の時間を割り当て
        num_records = len(self.records)
        num_move_times = len(move_times)
        cur_disp_time_s = float(move_times[0])

        for idx, cur_record in enumerate(self.records):
            if idx < (num_move_times-1):
                cur_record.disp_time_s = float(move_times[idx + 0])
                cur_record.disp_time_e = float(move_times[idx + 1])
            elif idx == (num_move_times-1):
                cur_record.disp_time_s = float(move_times[idx + 0])
                if (num_records - idx) > 0:
                    # 残り手の平均時間　※最終手が動画時間以内に収まるよう、少し余裕を持たせる(2.0)
                    elapsed_time_for_rest = (mov_total_sec - cur_record.disp_time_s - 2.0) / (num_records - idx)
                    # elapsed_time_for_rest /= 2.0
                cur_record.disp_time_e = cur_record.disp_time_s + elapsed_time_for_rest
            else:
                cur_record.disp_time_s = cur_disp_time_s
                cur_record.disp_time_e = cur_record.disp_time_s + elapsed_time_for_rest

            cur_disp_time_s = cur_record.disp_time_e

        return

    def loadKif(self, kif_fpath:str, player_name:str):

        with open(kif_fpath,"r", encoding="shift_jis") as kif_file:
            data_lines = kif_file.readlines()
            stat_parse = STAT_PARSE_HEADER
            cur_data = Record(no_val=0)

            for data_line in data_lines:

                if stat_parse == STAT_PARSE_HEADER:
                    if data_line.find(player_name) != -1 or data_line.find("プレイヤー") != -1:
                        if data_line.find("先手") != -1:
                            self.is_player_sente = True
                        else:
                            if data_line.find("後手") != -1:
                                self.is_player_sente = False

                if data_line[0:4].replace(" ","").isdecimal() == True:
                    # [データ(1手分)の先頭行]
                    stat_parse = transStat(stat_parse) 

                    # 一手前の手を登録
                    self.addRecord(cur_data)

                    prev_score = cur_data.score # 最終手のスコアがクリアされるのを防ぐ
                    cur_data.clear(no_val=0)
                    cur_data.score = prev_score

                    # 今回の手を解析
                    if stat_parse == STAT_PARSE_CONTENTS_SENTE:
                        cur_data.setBaseInfo(data_line, True, self.is_player_sente)
                    else:
                        cur_data.setBaseInfo(data_line, False, self.is_player_sente)
                
                else:
                    if data_line.find("*#指し手") != -1:
                        # [データ(1手分)のスコア記載行]
                        cur_data.setScore(data_line, self.is_player_sente)

                    elif data_line.find("*#推奨手") != -1:
                        cur_data.move_recommend.parse(data_line, self.is_player_sente)

                    elif data_line.find("戦法") != -1 or data_line.find("戦型") != -1 or data_line.find("囲い") != -1 or data_line.find("手筋") != -1:
                        # [データ(1手分)の戦型/囲い/手筋の記載行]
                        cur_data.setExtInfo(data_line)


            # 最後のデータを登録
            self.addRecord(cur_data)
        
        # 勝ち負け判定
        game_res = self.analysisGameResult()

        # 最終手の「切れ負」の処理
        if game_res == RecordSet.RESULT_WIN_TIMEOUT:
            self.getTailRecord().move = "切れ勝"

        return

    def debugOut(self, outdir:str):
        with open(f"{os.path.dirname(outdir)}/debug.csv","w") as fp_dbg:
            fp_dbg.write(f"frame,sente/gote,move,")
            fp_dbg.write(f"disp_s_org,disp_s,disp_e,")
            fp_dbg.write(f"elapse_time_sec,elapse_time_str,")
            fp_dbg.write(f"think_time,cnt_0_sec\n")

            self.initIter(is_reverse=False)
            for idx, (cur_record, _) in enumerate(self):
                fp_dbg.write(f"frame_{idx:03}.png,{"sente" if cur_record.is_sente==True else "gote"},{cur_record.move},")
                fp_dbg.write(f"{cur_record.disp_time_s_org:.1f},{cur_record.disp_time_s:.1f},{cur_record.disp_time_e:.1f},")
                fp_dbg.write(f"{cur_record.elapsed_time_sec:.1f},{cur_record.elapsed_time_str},")
                fp_dbg.write(f"{cur_record.thinking_time},{cur_record.cnt_seq_0sec_}\n")
        return

def evalScore(cur_score:int) -> Tuple[str,str]:
    str_color:str = "black"
    str_game_status:str = "互角"
    if abs(cur_score) > 500:
        if cur_score > 20000:
            str_color="lime"
            str_game_status = "勝勢"
        elif cur_score > 0:
            str_color="green"
            str_game_status = "優勢"
        elif cur_score < -20000:
            str_color="navy"
            str_game_status = "敗勢"
        else:    
            str_color="blue"
            str_game_status = "劣勢"
    
    return (str_color, str_game_status)

# 日本語パス対応版 (https://qiita.com/SKYS/items/cbde3775e2143cad7455)
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8) -> np.ndarray:
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img # type: ignore
    except Exception as e:
        print(e)
        return None # type: ignore

def makeScoreGraph(record_set:RecordSet, kif_dir:str, kif_filename:str) -> str:
    graph_outdir = kif_dir + "/" + kif_filename + "_graph"

    if os.path.exists(graph_outdir) == False:
        os.mkdir(graph_outdir)
    
    num_records = len(record_set)

    graph_x_frame = []
    graph_y_score = []
    max_score_abs = 0.0

    graph_fig = plt.figure(figsize=(6.0, 2.0))
    graph_ax = graph_fig.add_subplot(1, 1, 1)
    #plt.rcParams['font.family'] = "MS Gothic"
    plt.rcParams['font.family'] = "Meiryo"
    #plt.rcParams['font.family'] = "Yu Gothic"
    plt.rcParams["font.size"] = 16

    record_set.initIter()

    for idx, (cur_record , _) in enumerate(record_set):
        cur_score    = 0
        str_cur_move = ""
        str_rec_move = ""

        str_color       = "black"
        str_rec_color   = "black"
        str_game_status = "互角"

        if cur_record.isValid() == True:
            cur_score = cur_record.score
            graph_x_frame.append(idx)
            graph_y_score.append(cur_score)

            if max_score_abs < abs(cur_score):
                max_score_abs = abs(cur_score)
            
            # str_cur_move = cur_record.toString()
            str_cur_move = str(cur_record)
            (str_color, str_game_status) = evalScore(cur_score)

            if cur_record.move_recommend.is_exist == True:
                # if cur_record.is_bad_move == True:
                str_rec_move = "推奨手: " + cur_record.move_recommend.moves + " (score=" + str(cur_record.move_recommend.score) + ")"
                (str_rec_color, _) = evalScore(cur_record.move_recommend.score)
        
        # グラフ描画＆保存
        plt.cla()

        graph_ax.text(0.01, 0.99, str_game_status, va="top", transform=graph_ax.transAxes, color=str_color, fontweight="bold")
        graph_ax.text(0.01, 0.02, str_cur_move, transform=graph_ax.transAxes, color=str_color, fontweight="bold")
        graph_ax.text(0.01, -0.11, str_rec_move, transform=graph_ax.transAxes, color=str_rec_color, fontweight="bold", fontsize=13)

        plt.xlim(0, num_records)
        plt.ylim(-max_score_abs - 10.0, max_score_abs + 10.0)
        plt.tick_params(labelbottom=False, labelsize=12)
        plt.plot(graph_x_frame, graph_y_score, color=str_color)

        # graph_imfname:str = "frame_{x:03}".format(x=idx) + ".png"
        graph_imfname = f"{GRAPH_IMG_PREFIX}{idx:03}.png"
        graph_outfname = graph_outdir + "/" + graph_imfname
        print(graph_outfname)

        # グラフ画像保存
        plt.savefig(graph_outfname) 
        # plt.savefig(graph_outfname,transparent=True) 

        # グラフ画像表示
        im:np.ndarray = imread(graph_outfname)
        if im is not None:
            cv2.imshow("graph",im)
            cv2.waitKey(1)

    return graph_outdir



def writeExo(fp:io.TextIOWrapper, idx_obj:int, out_info:Dict[str,Any]):
    MOV_LAYER = 1
    IMG_LAYER = 2

    if idx_obj <= 0:
        mov_fpath = out_info["mov_fpath"].replace("/","\\")

        # ヘッダ
        fp.write(f"[exedit]\n")
        fp.write(f"width={out_info["out_img_w"]}\n")
        fp.write(f"height={out_info["out_img_h"]}\n")
        fp.write(f"rate={out_info["mov_fps"]}\n")
        fp.write(f"scale=1\n")
        fp.write(f"length={out_info["mov_num_frame"]}\n")

        # 動画レイヤ
        fp.write(f"[0]\n")
        fp.write(f"start=1\n")
        fp.write(f"end={out_info["mov_num_frame"]}\n")
        fp.write(f"layer={MOV_LAYER}\n")
        fp.write(f"overlay=1\n")
        fp.write(f"camera=0\n")

        fp.write(f"[0.0]\n")
        fp.write(f"_name=動画ファイル\n")
        fp.write(f"再生位置=1\n")
        fp.write(f"再生速度=100.0\n")
        fp.write(f"ループ再生=0\n")
        fp.write(f"アルファチャンネルを読み込む=0\n")
        fp.write(f"file={mov_fpath}\n")

        fp.write(f"[0.1]\n")
        fp.write(f"_name=標準描画\n")
        fp.write(f"X={out_info["mov_pos_x"]}\n")
        fp.write(f"Y={out_info["mov_pos_y"]}\n")
        fp.write(f"Z=0.0\n")
        fp.write(f"拡大率={out_info["mov_scale"]}\n")
        fp.write(f"透明度=0.0\n")
        fp.write(f"回転=0.00\n")
        fp.write(f"blend=0\n")

    else:
        # スコアグラフ画像レイヤ
        graph_img_fpath = out_info["graph_img_fpath"].replace("/","\\")

        fp.write(f"[{idx_obj}]\n")
        fp.write(f"start={out_info["frame_start_idx"]}\n")
        fp.write(f"end={out_info["frame_end_idx"]}\n")
        fp.write(f"layer={IMG_LAYER}\n")
        fp.write(f"overlay=1\n")
        fp.write(f"camera=0\n")
        fp.write(f"[{idx_obj}.0]\n")
        fp.write(f"_name=画像ファイル\n")
        fp.write(f"file={graph_img_fpath}\n")
        fp.write(f"[{idx_obj}.1]\n")
        fp.write(f"_name=標準描画\n")
        fp.write(f"X={out_info["mov_pos_x"]}\n")
        fp.write(f"Y={out_info["mov_pos_y"]}\n")
        fp.write(f"Z=0.0\n")
        fp.write(f"拡大率=100.00\n")
        fp.write(f"透明度=0.0\n")
        fp.write(f"回転=0.00\n")
        fp.write(f"blend=0\n")

    return

# def makeAviutilExoFile(record_set:RecordSet, mov_fpath:str, graph_dirpath:str, kif_fname:str, last_time_s_str=""):
def makeAviutilExoFile(record_set:RecordSet, mov_fpath:str, graph_dirpath:str, kif_fname:str):

    # 対局動画をOpen
    mov = MovieLoader()
    mov.load(mov_fpath)
    # movie = cv2.VideoCapture(mov_fpath)

    # if movie is not None:
    if mov.isOpened() == True:

        mov_fdir  = os.path.dirname(mov_fpath)
        mov_fname = os.path.splitext(os.path.basename(mov_fpath))[0]

        # フレーム数やフレームサイズ等を取得
        num_mov_frame = mov.getNumFrame()
        (mov_img_w, mov_img_h) = mov.getFrameSize()
        mov_fps = mov.getMovieFps()
        # num_mov_frame = int(mov.cap_.get(cv2.CAP_PROP_FRAME_COUNT))
        # mov_img_w = int(mov.cap_.get(cv2.CAP_PROP_FRAME_WIDTH))
        # mov_img_h = int(mov.cap_.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # mov_fps = mov.cap_.get(cv2.CAP_PROP_FPS)
        mov_ms_per_frame = 1000.0 / mov_fps
        mov_total_sec = float(num_mov_frame) / mov_fps

        # 効果音から、各手の時刻を認識
        wars_audio_detect = WarsAudioDetector(audio_detect_cfg)
        move_times = wars_audio_detect.extractFeature(mov.audio_)

        # 認識した時刻をrecord_setに反映
        REST_ELAPSED_TIME_SEC = 6
        record_set.assignTime(move_times, REST_ELAPSED_TIME_SEC, mov_total_sec)

        # 時刻調整
        # if len(last_time_s_str) > 0:
        #     # [時刻ありkifの場合] 最終手の時刻と合うよう調整
        #     record_set.adjustTime(last_time_s_str) 
        # else:
        #     # [時刻なしkifの場合] 一手毎の時間を均等配分
        #     AVE_ELAPSED_TIME_SEC = 6
        #     ave_elapsed_time_sec_mov = mov_total_sec / float(len(record_set))
        #     # print(f"ave_elapsed_time_sec_mov={ave_elapsed_time_sec_mov}")

        #     if AVE_ELAPSED_TIME_SEC < ave_elapsed_time_sec_mov:
        #         record_set.createTime(AVE_ELAPSED_TIME_SEC) 
        #     else:
        #         record_set.createTime(int(ave_elapsed_time_sec_mov))

        # record_set.debugOut(mov_fdir)

        # exoファイルを作成
        # https://qiita.com/tset-tset-tset/items/e245c9c1a4fdbb18cf6d
        exo_fpath = f"{mov_fdir}/{kif_fname}.exo"
        with open(exo_fpath, "w", encoding="shift_jis", newline="\r\n") as exo_file:


            # スコアデータ
            record_set.initIter()
            num_records = len(record_set)
            idx_record = 0
            cur_record = record_set.getCurRecord()
            tail_record = record_set.getTailRecord()

            # スコアグラフ画像
            graph_img_fname = f"{graph_dirpath}/{GRAPH_IMG_PREFIX}{idx_record:03}.png"
            graph_img = imread(graph_img_fname)
            (graph_img_h, graph_img_w, _) = graph_img.shape

            disp_img_w = max(mov_img_w, graph_img_w)
            disp_img_h = mov_img_h + graph_img_h

            # exo出力（ヘッダ＆動画レイヤ）
            out_info = {"out_img_w":disp_img_w, 
                        "out_img_h":disp_img_h,
                        "mov_fps":int(mov_fps),
                        "mov_num_frame":num_mov_frame,
                        "mov_fpath":mov_fpath,
                        "mov_pos_x":0,
                        "mov_pos_y":graph_img_h/2,
                        "mov_scale":100}
            writeExo(exo_file, 0, out_info)

            # exo出力（スコアグラフ（先頭））
            prev_record_time_e_idx = 1
            cur_record_time_e_ms  = cur_record.disp_time_e * 1000.0
            cur_record_time_e_idx = int(cur_record_time_e_ms/mov_ms_per_frame) + 1

            out_info = {"frame_start_idx":1, 
                        "frame_end_idx": cur_record_time_e_idx,
                        "mov_pos_x":-(disp_img_w - graph_img_w)/2,
                        "mov_pos_y":-(disp_img_h - graph_img_h)/2,
                        "graph_img_fpath":graph_img_fname}
            writeExo(exo_file, 1, out_info)

            ms_now = 0.0
            is_out_tailrecord = False

            for idx_mov_frame in range(num_mov_frame):
                # 対局画像取得
                ret, mov_img = mov.cap_.read()

                if ret == True:

                    # スコアグラフ画像を重畳
                    disp_img = np.zeros((disp_img_h, disp_img_w, 3), np.uint8)
                    disp_img[0:graph_img_h, 0:graph_img_w, :] = graph_img[:,:,:]
                    disp_img[graph_img_h:disp_img_h, :, :]    = mov_img[:,:,:]

                    cv2.imshow("movie", disp_img)
                    cv2.waitKey(1)

                    if ms_now > cur_record_time_e_ms:

                        try:
                            prev_record_time_e_idx = cur_record_time_e_idx

                            # 次のデータに進む
                            (_,_) = record_set.__next__()

                            if record_set.cur_idx < num_records - 1:
                                # print(f"ms_now = {ms_now}[ms], cur_record_time_e_ms = {cur_record_time_e_ms}")

                                idx_record = record_set.cur_idx
                                cur_record = record_set.getCurRecord()
                                graph_img_fname = f"{graph_dirpath}/{GRAPH_IMG_PREFIX}{idx_record:03}.png"
                                graph_img = imread(graph_img_fname)

                                # exo出力（スコアグラフ）
                                cur_record_time_e_ms  = cur_record.disp_time_e * 1000.0
                                cur_record_time_e_idx = int(cur_record_time_e_ms/mov_ms_per_frame) + 1

                                out_info["frame_start_idx"] = prev_record_time_e_idx + 1
                                out_info["frame_end_idx"]   = cur_record_time_e_idx
                                out_info["graph_img_fpath"] = graph_img_fname
                                writeExo(exo_file, idx_record + 1, out_info)

                        except StopIteration as e:
                            pass

                        if (ms_now > tail_record.disp_time_e * 1000.0) and (is_out_tailrecord == False):
                            # exo出力（スコアグラフ末端）
                            idx_record = num_records - 1
                            graph_img_fname = f"{graph_dirpath}/{GRAPH_IMG_PREFIX}{idx_record:03}.png"
                            out_info["frame_start_idx"] = prev_record_time_e_idx + 1
                            out_info["frame_end_idx"]   = prev_record_time_e_idx + 30
                            out_info["graph_img_fpath"] = graph_img_fname
                            writeExo(exo_file, idx_record + 1, out_info)
                            is_out_tailrecord = True

                    ms_now += mov_ms_per_frame


        mov.release()

    return

def is_mm_ss_format(s: str) -> bool:
    return bool(re.match(r'^\d+:[0-5]\d$', s))

def main(player_name:str):
 
    # メインウィンドウは非表示
    root = tk.Tk()
    root.withdraw()

    # ファイルダイアログでkifファイル選択
    file_type = [('kifファイル','*.kif')] 
    kif_fpath = filedialog.askopenfilename(filetypes = file_type) 

    if kif_fpath != "":
        
        # kifファイルLOAD
        record_set = RecordSet()
        record_set.loadKif(kif_fpath, player_name)
        record_set.print()

        # グラフ作成＆ファイル出力
        kif_dir = os.path.dirname(kif_fpath)
        kif_fname = os.path.splitext(os.path.basename(kif_fpath))[0]
        outdir_graph = makeScoreGraph(record_set, kif_dir, kif_fname)


        # ファイルダイアログで対局動画ファイル選択
        file_type = [("aviファイル","*.avi"),("mp4ファイル","*.mp4")] 
        mov_fpath = filedialog.askopenfilename(filetypes = file_type) 

        if mov_fpath != "":
            # print(mov_fpath)

            makeAviutilExoFile(record_set, mov_fpath, outdir_graph, kif_fname)

            # if math.isclose(record_set.getTailRecord().disp_time_e, 0.0) == False:
            #     # [時刻ありkifの場合]
            #     last_time_s_str = simpledialog.askstring("最終手の時刻入力", "最終手の時刻を入力してください(mm:ss)")
            #     while is_mm_ss_format(last_time_s_str) == False: # type: ignore
            #         last_time_s_str = simpledialog.askstring("最終手の時刻入力", "最終手の時刻を入力してください(mm:ss)")
            #         print(last_time_s_str)

            #     makeAviutilExoFile(record_set, mov_fpath, outdir_graph, kif_fname, last_time_s_str) # type: ignore
            
            # else:
            #     # [時刻なしkifの場合]
            #     makeAviutilExoFile(record_set, mov_fpath, outdir_graph, kif_fname)

    return

if __name__ == "__main__":
    args = sys.argv

    if len(args) < 2:
        print("Usage: ", args[0], " [player name]")
    else:
        print("exec: ", args[0], " ", args[1])
        main(args[1])

