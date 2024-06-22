/*
    Created by Rainyl.
    Licensed: Apache 2.0 license. Copyright (c) 2024 Rainyl.
*/

#include "stitching.h"

CvStatus *Stitcher_Create(int mode, PtrStitcher *rval)
{
  BEGIN_WRAP
  const auto ptr = cv::Stitcher::create(static_cast<cv::Stitcher::Mode>(mode));
  *rval = {new cv::Ptr<cv::Stitcher>(ptr)};
  END_WRAP
}

void Stitcher_Close(PtrStitcherPtr stitcher)
{
  stitcher->ptr->reset();
  CVD_FREE(stitcher);
}

CvStatus *Stitcher_Get(PtrStitcher self, Stitcher *rval)
{
  BEGIN_WRAP
  *rval = {self.ptr->get()};
  END_WRAP
}

CvStatus *Stitcher_GetRegistrationResol(Stitcher self, double *rval)
{
  BEGIN_WRAP
  *rval = self.ptr->registrationResol();
  END_WRAP
}

CvStatus *Stitcher_SetRegistrationResol(Stitcher self, double inval)
{
  BEGIN_WRAP
  self.ptr->setRegistrationResol(inval);
  END_WRAP
}

CvStatus *Stitcher_GetSeamEstimationResol(Stitcher self, double *rval)
{
  BEGIN_WRAP
  *rval = self.ptr->seamEstimationResol();
  END_WRAP
}
CvStatus *Stitcher_SetSeamEstimationResol(Stitcher self, double inval)
{
  BEGIN_WRAP
  self.ptr->setSeamEstimationResol(inval);
  END_WRAP
}

CvStatus *Stitcher_GetCompositingResol(Stitcher self, double *rval)
{
  BEGIN_WRAP
  *rval = self.ptr->compositingResol();
  END_WRAP
}
CvStatus *Stitcher_SetCompositingResol(Stitcher self, double inval)
{
  BEGIN_WRAP
  self.ptr->setCompositingResol(inval);
  END_WRAP
}

CvStatus *Stitcher_GetPanoConfidenceThresh(Stitcher self, double *rval)
{
  BEGIN_WRAP
  *rval = self.ptr->panoConfidenceThresh();
  END_WRAP
}
CvStatus *Stitcher_SetPanoConfidenceThresh(Stitcher self, double inval)
{
  BEGIN_WRAP
  self.ptr->setPanoConfidenceThresh(inval);
  END_WRAP
}

CvStatus *Stitcher_GetWaveCorrection(Stitcher self, bool *rval)
{
  BEGIN_WRAP
  *rval = self.ptr->waveCorrection();
  END_WRAP
}
CvStatus *Stitcher_SetWaveCorrection(Stitcher self, bool inval)
{
  BEGIN_WRAP
  self.ptr->setWaveCorrection(inval);
  END_WRAP
}

CvStatus *Stitcher_GetInterpolationFlags(Stitcher self, int *rval)
{
  BEGIN_WRAP
  *rval = static_cast<int>(self.ptr->interpolationFlags());
  END_WRAP
}
CvStatus *Stitcher_SetInterpolationFlags(Stitcher self, int inval)
{
  BEGIN_WRAP
  self.ptr->setInterpolationFlags(static_cast<cv::InterpolationFlags>(inval));
  END_WRAP
}

CvStatus *Stitcher_GetWaveCorrectKind(Stitcher self, int *rval)
{
  BEGIN_WRAP
  *rval = static_cast<int>(self.ptr->waveCorrectKind());
  END_WRAP
}
CvStatus *Stitcher_SetWaveCorrectKind(Stitcher self, int inval)
{
  BEGIN_WRAP
  self.ptr->setWaveCorrectKind(static_cast<cv::detail::WaveCorrectKind>(inval));
  END_WRAP
}

CvStatus *Stitcher_EstimateTransform(Stitcher self, VecMat mats, VecMat masks, int *rval)
{
  BEGIN_WRAP
  if (masks.ptr->size() > 0) {
    *rval = static_cast<int>(self.ptr->estimateTransform(*mats.ptr, *masks.ptr));
  } else
    *rval = static_cast<int>(self.ptr->estimateTransform(*mats.ptr));
  END_WRAP
}

CvStatus *Stitcher_ComposePanorama(Stitcher self, Mat rpano, int *rval)
{
  BEGIN_WRAP
  *rval = static_cast<int>(self.ptr->composePanorama(*rpano.ptr));
  END_WRAP
}
CvStatus *Stitcher_ComposePanorama_1(Stitcher self, VecMat mats, Mat rpano, int *rval)
{
  BEGIN_WRAP
  *rval = static_cast<int>(self.ptr->composePanorama(*mats.ptr, *rpano.ptr));
  END_WRAP
}

CvStatus *Stitcher_Stitch(Stitcher self, VecMat mats, Mat rpano, int *rval)
{
  BEGIN_WRAP
  *rval = static_cast<int>(self.ptr->stitch(*mats.ptr, *rpano.ptr));
  END_WRAP
}
CvStatus *Stitcher_Stitch_1(Stitcher self, VecMat mats, VecMat masks, Mat rpano, int *rval)
{
  BEGIN_WRAP
  *rval = static_cast<int>(self.ptr->stitch(*mats.ptr, *masks.ptr, *rpano.ptr));
  END_WRAP
}

CvStatus *Stitcher_Component(Stitcher self, VecInt *rval)
{
  BEGIN_WRAP
  std::vector<int> _rval = self.ptr->component();
  *rval = {new std::vector<int>(_rval)};
  END_WRAP
}
