#pragma once

//设置全局参数常量

#define HOG_BIN 9
#define HOG_WIN {32, 32}
#define HOG_STRIDE {8, 8}
#define HOG_BLOCK {16, 16}
#define HOG_CELL {4, 4}

#define UPDATE_JUDGER 0.85
#define DEFALUT 1

#define TMPSIZE_H 32
#define TMPSIZE_W 32

#define NORMAL_H 480
#define NORMAL_W 640

#define NEG_SLIDING_H 150
#define NEG_SLIDING_W 150
#define NEG_SRIDE 10
#define NEG_EXCLUDE_RATIO 0.3

#define POS_SLIDING_H 5
#define POS_SLIDING_W 6

#define PARTICLE_N 100
#define PARTICLE_CONDENSE 0.05
#define PARTICLE_AFFSIG {6, 6, 0.03f, 0.03f}


#define COLDBOOT_ITER 1000
#define ITER 7