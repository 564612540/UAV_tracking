#pragma once

void grad1(float *I, float *Gx, float *Gy, int h, int w, int x);

void grad2(float *I, float *Gx, float *Gy, int h, int w, int d);

float* acosTable();

void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full);

void gradMagNorm(float *M, float *S, int h, int w, float norm);

void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1,
	int nb, int n, float norm, int nOrients, bool full, bool interpolate);

void gradHist(float *M, float *O, float *H, int h, int w,
	int bin, int nOrients, int softBin, bool full);

float* hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin);

void hogChannels(float *H, const float *R, const float *N,
	int hb, int wb, int nOrients, float clip, int type);

void hog(float *M, float *O, float *H, int h, int w, int binSize,
	int nOrients, int softBin, bool full, float clip);

void fhog(float *M, float *O, float *H, int h, int w, int binSize,
	int nOrients, int softBin, float clip);