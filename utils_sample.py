import re
import datetime
import pandas as pd
import numpy as np
import torch
from torch import nn
import random


# state
def gen_state(state_s):
    s = []
    p = state_s['pres']
    if p == '0' or p == 0:
        s += [np.nan, np.nan]
    else:
        try:
            p_l = re.findall("\d+", p)
        except:
            p_l = re.findall("\d+", p.decode('utf-8'))
        p1, p2 = pressure(float(p_l[0]), float(p_l[1]))
        s.append(p1)
        s.append(p2)

    crea_v = 0
    for k in ['crea', 'urine', 'Na', 'Ka', 'tem', 'brea', 'pulse']:
        v = state_s[k]
        try:
            if np.isnan(v) == False:
                v = float(v)
                if v == 0:
                    s.append(np.nan)
                else:
                    if v == float(p_l[0]):
                        s.append(np.nan)
                    else:
                        if k == 'crea':
                            crea_v = v
                            v_ = crea(v)
                        elif k == 'tem':
                            v_ = excess(k, v)
                        else:
                            v_ = norm(k, v)

                        s.append(v_)
            else:
                s.append(np.nan)
        except:
            s.append(np.nan)

    for k in ['呼吸困难', '啰音', '腹胀', '水肿', '浮肿', 'sym', '心悸', '黑矇', '晕厥', '头晕',
              'rhy', '胸闷', '胸痛', 'mus', '咳嗽', '咳痰', '发热', 'infect']:
        s.append(state_s[k])

    return s, crea_v


# <133	133-177	178-442	443-707	>707
# 0	1	2	3	4
def crea(v):
    if v > 707:
        v_ = 4
    elif v >= 443:
        v_ = 3
    elif v >= 178:
        v_ = 2
    elif v >= 133:
        v_ = 1
    else:
        v_ = 0
    return v_


def pressure(p1, p2):
    if p1 >= 180:
        p1_ = 3
    elif p1 >= 160:
        p1_ = 2
    elif p1 >= 140:
        p1_ = 1
    elif p1 >= 101:
        p1_ = 0
    elif p1 >= 90:
        p1_ = -1
    else:
        p1_ = -2

    if p2 >= 110:
        p2_ = 3
    elif p2 >= 100:
        p2_ = 2
    elif p2 >= 90:
        p2_ = 1
    elif p2 >= 60:
        p2_ = 0
    elif p2 >= 50:
        p2_ = -1
    else:
        p2_ = -2

    return p1_, p2_


def norm(item, v):
    test_dic = {'urine': (3.1, 8.8), 'Na': (136, 147), 'Ka': (3.5, 5.0), '镁(Mg)': (0.75, 1.02),
                'brea': (12, 20), 'pulse': (60, 100), '总胆固醇': (3, 6), '甘油三脂': (0.4, 1.8), '低密度脂蛋白胆固醇': (2, 3.1),
                }
    a, b = test_dic[item]
    if v < a:
        v_ = -1
    elif v < b:
        v_ = 0
    else:
        v_ = 1
    return v_


def excess(item, v):
    test_dic = {'tem': 37.4, 'B型钠尿肽前体测定': 300, '肌钙蛋白I': 0.12, '肌钙蛋白T': 0.13, '肌酸激酶同工酶': 20,
                '白细胞(WBC)': 10, '中性粒细胞比率': 0.75, 'C-反应蛋白': 6, '降钙素原': 0.05,

                }
    a = test_dic[item]
    if v > a:
        v_ = 1
    else:
        v_ = 0

    return v_


def lower(item, v):
    test_dic = {'白蛋白（免疫比浊)': 40, '肾小球': 60}
    a = test_dic[item]
    if v < a:
        v_ = -1
    else:
        v_ = 0

    return v_


def add_s(test_s, s):
    overlap_test = ['肌酐', '钾(K)', '钠(Na)', '尿素']
    map_test = ['crea', 'Ka', 'Na', 'urine']
    ind = [2, 5, 4, 3]
    for i in range(4):
        k = overlap_test[i]
        v = test_s[k]
        try:
            if v == 0 or np.isnan(v) == True:
                pass
            else:
                v = float(v)
                map_k = map_test[i]
                if map_k == 'crea':
                    v_ = crea(v)
                else:
                    v_ = norm(map_k, v)
                s[ind[i]] = v_
        except:
            pass

    k = '镁(Mg)'
    v = test_s[k]
    if v == 0 or np.isnan(v) == True:
        s.append(np.nan)
    else:
        v_ = norm(k, v)
        s.append(v_)

    for k in ['白蛋白（免疫比浊)', 'B型钠尿肽前体测定', '总胆固醇', '甘油三脂', '低密度脂蛋白胆固醇',
              '肌钙蛋白I', '肌钙蛋白T', '肌酸激酶同工酶', '白细胞(WBC)', '中性粒细胞比率', 'C-反应蛋白', '降钙素原']:
        try:
            try:
                v = float(test_s[k])
            except:
                v = float(re.findall("\d+", test_s[k])[0])
        except:
            v = 0
        # print(v)
        if np.isnan(v) == True or v == 0:
            s.append(np.nan)
        else:
            if k in ['总胆固醇', '甘油三脂', '低密度脂蛋白胆固醇']:
                v_ = norm(k, v)

            elif k in ['白蛋白（免疫比浊)']:
                v_ = lower(k, v)
            else:
                v_ = excess(k, v)
            s.append(v_)

    return s
def ckd_cal(scr,age,gen):
    if scr == 0:
        ckd = np.nan
    else:

        if age == 0:
            age = 66

        if gen == 1:
            gen = 0
        ckd = 186 * (scr / 88.41) ** (-1.154) * age ** (-0.203) * (0.258 * gen + 1)
        ckd = lower('肾小球',ckd )
    return ckd

def info_rep(model,h_state,model_name,action,obs):
    if model_name=='rnn':
        obs = torch.where(
            torch.isnan(obs),
            torch.full_like(obs, 0), obs)
        x = torch.concat([obs, action], dim=-1).unsqueeze(0)
        prediction, h_state, zt = model(x, h_state)
        zt = zt.squeeze(0).detach()

    else:
        zt=obs
    return zt