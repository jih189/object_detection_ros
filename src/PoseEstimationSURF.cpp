
#include "object_tracking_2D/PoseEstimationSURF.h"
#include "object_tracking_2D/ObjectModel.h"
#include "object_tracking_2D/epnp.h"

#include "object_tracking_2D/Timer.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

CPoseEstimationSURF::CPoseEstimationSURF(int width, int height, std::string &template_path, CObjectModel *obj_model, CvMat *intrinsic_params, CvMat *distortion_params, std::string &objName, bool dispaly /*=true*/)
    : verbose_(false), display_(dispaly), draw_type_(CV_AA) // anti-aliasing drawing (CV_AA is 16), about 2ms more required
{
  ms_ = cvCreateMemStorage(0);

  // allocate images
  img_input_ = cvCreateImage(cvSize(width, height), 8, 1);
  //img_object_ = cvCreateImage(cvSize(width, height), 8, 1);
  obj_model_ = obj_model;

  // Init CvSeq data
  seq_keypoints_ = NULL;
  seq_descriptors_ = NULL;

  // creat and init
  pose_ = cvCreateMat(4, 4, CV_32F);
  cvSetIdentity(pose_);

  intrinsic_ = intrinsic_params;
  distortion_ = distortion_params;

  // check that there is the file
  string templateFileName(template_path + '/' + objName + ".txt");

  std::cout << "template file name: = " << templateFileName.c_str() << std::endl;
  fstream file;
  file.open(templateFileName.c_str());
  if (file.is_open())
  {
    file.close();
    lf_.Configure((template_path + "/para_line_fitter.txt").c_str());
    lm_.Configure((template_path + "/para_line_matcher.txt").c_str());
    lf_.Init();
    lm_.Init(template_path.c_str(), (objName + ".txt").c_str());
  }
}

CPoseEstimationSURF::~CPoseEstimationSURF(void)
{
  cvReleaseMat(&pose_);
  cvReleaseImage(&img_input_);
}

void CPoseEstimationSURF::buildKdTree(vector<IplImage *> &keyframe, vector<CvMat *> &pose, vector<CvMat *> &keypoint2D, vector<CvMat *> &keypoint3D, vector<CvMat *> &descriptor)
{
  if (keyframe.size() == 0 || pose.size() == 0 || keypoint2D.size() == 0 || keypoint3D.size() == 0 || descriptor.size() == 0)
    return;

  assert(keyframe.size() == keypoint2D.size());
  assert(keypoint2D.size() == keypoint3D.size());
  assert(keypoint3D.size() == descriptor.size());

  keyframe_images_ = keyframe;

  // Save keyframe descriptor into CvMat
  int dims = descriptor[0]->cols;
  int row = 0;
  for (int i = 0; i < keyframe.size(); i++)
  {
    row += keypoint2D[i]->rows;
  }

  cv::Mat kfd(row, dims, CV_32F);

  keyframe_lut_.resize(row);
  keyframe_keypoints_2d_.resize(row);
  keyframe_keypoints_3d_.resize(row);

  // Save keyframe descriptor into CvMat
  int k = 0;
  for (int h = 0; h < keyframe.size(); h++)
  {
    for (int i = 0; i < keypoint2D[h]->rows; i++)
    {
      keyframe_keypoints_2d_[i + k] = CV_MAT_ELEM(*keypoint2D[h], CvPoint2D32f, i, 0);
      keyframe_keypoints_3d_[i + k] = CV_MAT_ELEM(*keypoint3D[h], CvPoint3D32f, i, 0);
      keyframe_lut_[i + k] = h;
      for (int j = 0; j < dims; j++)
        kfd.at<float>(i + k, j) = CV_MAT_ELEM(*descriptor[h], float, i, j);
    }
    k += keypoint2D[h]->rows;
  }

  kfd_ = kfd;
}

bool sortBySize(const LMDetWind &left, const LMDetWind &right)
{
  return (left.width_ * left.height_ > right.width_ * right.height_);
}

int CPoseEstimationSURF::PF_estimatePosesFDCM(float maxThreshold, int numOfDetections, vector<CvMat *> &states, vector<LMDetWind> &detWind, int smoothSize /*=1*/, int cannyLow /*=20*/, int cannyHigh /*=40*/, IplImage *displayImage /*=NULL*/) // the display image is used to filter edge
{
  Timer timer;
  timer.start();
  // Using edge templates + random draw
  IplImage *inputImage = cvCloneImage(img_input_);
  IplImage *edgeImage = cvCloneImage(inputImage);
  IplImage *objecttemplate = cvCreateImage(cvGetSize(img_input_), IPL_DEPTH_8U, 1);
  IplImage *colorimage = cvCreateImage(cvGetSize(img_input_), IPL_DEPTH_8U, 3);

  // jiaming
  //displayImage = cvCreateImage(cvGetSize(inputImage), IPL_DEPTH_8U, 3);

  //cvCopy(inputImage, displayImage);

  if (smoothSize > 0)
    cvSmooth(inputImage, inputImage, CV_GAUSSIAN, smoothSize, smoothSize);

  cvCanny(inputImage, edgeImage, cannyLow, cannyHigh);
  IplImage *edgeImagetemp = cvCloneImage(edgeImage);
  // cv::imshow("edge", cv::Mat(edgeImage));
  // cv::waitKey(0);

  /*
  std::cout << "frank: filter the image\n";
  // filter out the edge image

  for(int i = 0; i < edgeImage->width; i++){
    for(int j = 0; j < edgeImage->height; j++){
      if(CV_IMAGE_ELEM(displayImage,uchar, j, 3 * i + 0) == 0 &&
         CV_IMAGE_ELEM(displayImage,uchar, j, 3 * i + 1) == 0 &&
         CV_IMAGE_ELEM(displayImage,uchar, j, 3 * i + 2) == 0){
      
        CV_IMAGE_ELEM(edgeImage,uchar, j, i) = 0;
      }
    }
  }
*/

  //if(displayImage) cvCvtColor(inputImage, displayImage, CV_GRAY2RGB);
  cvReleaseImage(&inputImage);
  // Line Fitting
  lf_.FitLine(edgeImage);
  //lf_.FitLine_omp(edgeImage);

  if (edgeImage)
    cvReleaseImage(&edgeImage);

  // FDCM Matching
  lm_.MultiShapeDetectionWithVaryingTemplateSize(lf_, (double)maxThreshold, detWind);

  std::cout << detWind.size() << " detections..." << std::endl;

  std::cout << "x	y	width	height	cost		count	scale	aspect	template_id" << std::endl;
  for (size_t i = 0; i < detWind.size(); i++)
  {
    std::cout << detWind[i].x_ << "	" << detWind[i].y_ << "	" << detWind[i].width_ << "	" << detWind[i].height_ << "	" << detWind[i].cost_ << "	" << detWind[i].count_ << "	" << detWind[i].scale_ << "	" << detWind[i].aspect_ << "	" << detWind[i].tidx_ << std::endl;
    //DrawDetWind(displayImage, detWind[i].x_, detWind[i].y_, detWind[i].width_, detWind[i].height_, cvScalar(255), 3);
  }

  if (detWind.size() > 0)
  {
    // Calculate coarse pose
    float lamda = 1.0f;

    int numOfDet = detWind.size();
    vector<CvMat *> poses;
    //vector<float> vweight;
    poses.resize(numOfDet);
    //vweight.resize(numOfDet);
    float u0 = CV_MAT_ELEM(*intrinsic_, float, 0, 2);
    float v0 = CV_MAT_ELEM(*intrinsic_, float, 1, 2);
    float fx = CV_MAT_ELEM(*intrinsic_, float, 0, 0);
    float fy = CV_MAT_ELEM(*intrinsic_, float, 1, 1);

    float tx = 0.0;
    float ty = 0.0;
    // get the pose of template which matches the object
    for (int d = 0; d < numOfDet; d++)
    {
      poses[d] = cvCreateMat(4, 4, CV_32F);
      cvCopy(obj_model_->getEdgeTemplatePose(static_cast<int>(detWind[d].tidx_)), poses[d]);

      // get the center position of the object template
      int x = detWind[d].x_ + (int)((obj_model_->getPosePosition(static_cast<int>(detWind[d].tidx_))).x * detWind[d].scale_);
      int y = detWind[d].y_ + (int)((obj_model_->getPosePosition(static_cast<int>(detWind[d].tidx_))).y * detWind[d].scale_);

      float Z = (CV_MAT_ELEM(*poses[d], float, 2, 3) / detWind[d].scale_);
      float X = ((float(x) - u0) * Z - tx) / fx;
      float Y = ((float(y) - v0) * Z - ty) / fy;

      CV_MAT_ELEM(*poses[d], float, 0, 3) = X;
      CV_MAT_ELEM(*poses[d], float, 1, 3) = Y;
      CV_MAT_ELEM(*poses[d], float, 2, 3) = Z;

      // refine the pose
      /*
      obj_model_->setModelviewMatrix(poses[d]);
      obj_model_->findVisibleSamplePoints();
      obj_model_->getVisibleArea(objecttemplate->height, objecttemplate->width, true, objecttemplate);
      //cv::imshow("test" + std::to_string(d), cv::cvarrToMat(objecttemplate));

      //obj_model_->displayPoseLine(displayImage, poses[i], CV_RGB(0, 255, 0), 1, false);
      lf_.FitLine(objecttemplate);

      EIEdgeImage templateImage;
      templateImage.SetNumDirections(60);
      templateImage.Read(lf_);
      templateImage.SetDirectionIndices();

      double minx, miny, maxx, maxy;
      templateImage.Boundary(minx, miny, maxx, maxy);
      double centerx = minx + (maxx - minx) / 2;
      double centery = miny + (maxy - miny) / 2;

      double totalcost = 0.0;
      double mincost = 1000000;
      int seachSize = 10;

      int fitx = 0;
      int fity = 0;
      int fitth = 0;
      double fitsc = 1.0;
      for (int i = -seachSize; i < seachSize; i+=2)
      {
        for (int j = -seachSize; j < seachSize; j+=2)
        {
          for (double th = -0.3; th < 0.3; th += 0.05)
          {
            for (double sc = 0.8; sc < 1.2; sc += 0.1)
            {
              double totalcost = 0.0;
              templateImage.Read(lf_, th, i, j, sc, centerx, centery);
              templateImage.SetDirectionIndices();

              //std::cout << "edgeImage with " << edgeImage->height << " " << edgeImage->width << " " << edgeImage->nChannels << std::endl;
              //std::cout << "objecttemplate with " << objecttemplate->height << " " << objecttemplate->width << " " << objecttemplate->nChannels << std::endl;
              cvCvtColor(edgeImagetemp, colorimage, CV_GRAY2BGR);
              for (int l = 0; l < templateImage.nLines_; l++)
              {
                cvLine(colorimage,
                       cvPoint((int)((templateImage.lines_)[l].sx_),
                               (int)((templateImage.lines_)[l].sy_)),
                       cvPoint((int)((templateImage.lines_)[l].ex_),
                               (int)((templateImage.lines_)[l].ey_)),
                       cvScalar(255,255,0));
                totalcost += (lm_.GetCost(templateImage.directionIndices_[l], (int)((templateImage.lines_)[l].sx_), (int)((templateImage.lines_)[l].sy_), (int)((templateImage.lines_)[l].ex_), (int)((templateImage.lines_)[l].ey_)) / templateImage.Length());
                //if (totalcost >= mincost)
                //break;
              }

              //cv::imshow("check" + std::to_string(d), cv::cvarrToMat(colorimage));
              //std::cout << "cost: " << totalcost << std::endl;
              //cv::waitKey(0);
              cvSet(objecttemplate, cvScalar(0));

              if (totalcost < mincost)
              {
                mincost = totalcost;
                fitx = i;
                fity = j;
                fitth = th;
                fitsc = sc;
              }
            }
          }
        }
      }
      // std::cout << "best matching with fit x = " << fitx
      //           << " fit y = " << fity
      //           << " fit th = " << fitth
      //           << " fit sc = " << fitsc
      //           << " with cost = " << mincost << std::endl;
      // demonstrate the result
      templateImage.Read(lf_, fitth, fitx, fity, fitsc, centerx, centery);
      templateImage.SetDirectionIndices();
      cvCvtColor(edgeImagetemp, colorimage, CV_GRAY2BGR);
      for (int l = 0; l < templateImage.nLines_; l++)
      {
        cvLine(colorimage,
               cvPoint((int)((templateImage.lines_)[l].sx_),
                       (int)((templateImage.lines_)[l].sy_)),
               cvPoint((int)((templateImage.lines_)[l].ex_),
                       (int)((templateImage.lines_)[l].ey_)),
               cvScalar(255,255,0));
      }
      //cv::imshow("check" + std::to_string(d), cv::cvarrToMat(colorimage));
      //cv::waitKey(0);
      */
    }

    for (int d = 0; d < numOfDet; d++)
    {
      states.push_back(cvCreateMat(4, 4, CV_32F));
      cvCopy(poses[d], states[d]);
    }

    /*approach the object*/
    //cvReleaseImage(&displayImage);

    for (int d = 0; d < numOfDet; d++)
    {
      cvReleaseMat(&poses[d]);
    }
  }

  if (edgeImagetemp)
    cvReleaseImage(&edgeImagetemp);
  if (colorimage)
    cvReleaseImage(&colorimage);
  cvReleaseImage(&objecttemplate);

  timer.printTimeMilliSec("PF_estimatePosesFDCM()");

  return detWind.size();
}

// EPnP version
int CPoseEstimationSURF::refineCorrespondenceEpnpRANSAC(const vector<int> &ptpairs, vector<CvPoint2D32f> &objOutliers, vector<CvPoint3D32f> &objOutliers3D, vector<CvPoint2D32f> &imgOutliers, vector<CvPoint2D32f> &objInliers, vector<CvPoint3D32f> &objInliers3D, vector<CvPoint2D32f> &imgInliers, CvMat *pmPose)
{
  const int NOM = 7; // number of model parameters

  int n;
  int iter = 0;
  int k = 100000;
  const int max_k = 1000;
  int best_noi = 0;
  const float th = 20;
  const double p = 0.99;

  n = int(ptpairs.size() / 2);
  if (n < 8) // at least 8 points are needed to estimate fundamental matrix
    return -1;

  objOutliers.resize(n);
  imgOutliers.resize(n);
  objOutliers3D.resize(n);
  for (int i = 0; i < n; i++)
  {
    objOutliers[i] = keyframe_keypoints_2d_[ptpairs[i * 2]];
    imgOutliers[i] = input_keypoints_2d_[ptpairs[i * 2 + 1]];
    objOutliers3D[i] = keyframe_keypoints_3d_[ptpairs[i * 2]];
  }

  epnp ePnP;

  float fu = CV_MAT_ELEM(*intrinsic_, float, 0, 0);
  float fv = CV_MAT_ELEM(*intrinsic_, float, 1, 1);
  float uc = CV_MAT_ELEM(*intrinsic_, float, 0, 2);
  float vc = CV_MAT_ELEM(*intrinsic_, float, 1, 2);

  vector<int> inlier_idx;
  inlier_idx.resize(n);
  vector<int> best_inlier_idx;
  best_inlier_idx.resize(n);

  ePnP.set_internal_parameters(uc, vc, fu, fv);
  ePnP.set_maximum_number_of_correspondences(NOM);

  CvRNG rng = cvRNG(cvGetTickCount());
  int rand_idx[NOM];
  CvMat *P = cvCreateMat(3, 4, CV_32F);
  CvMat *P2 = cvCreateMat(3, 4, CV_32F);
  CvMat *x3d_h = cvCreateMat(4, n, CV_32F);
  CvMat *x2d_proj = cvCreateMat(3, n, CV_32F);

  for (int i = 0; i < n; i++)
  {
    CV_MAT_ELEM(*x3d_h, float, 0, i) = objOutliers3D[i].x;
    CV_MAT_ELEM(*x3d_h, float, 1, i) = objOutliers3D[i].y;
    CV_MAT_ELEM(*x3d_h, float, 2, i) = objOutliers3D[i].z;
    CV_MAT_ELEM(*x3d_h, float, 3, i) = 1.0;
  }

  double R_est[3][3], T_est[3];

  while (iter < k && iter < max_k)
  {
    // sampling
    for (int i = 0; i < NOM; i++)
    {
      int temp_idx = 0;
      bool found = true;
      while (found)
      {
        temp_idx = cvRandInt(&rng) % n;
        found = false;
        for (int j = 0; j < i; j++)
        {
          if (rand_idx[j] == temp_idx)
            found = true;
        }
      }
      rand_idx[i] = temp_idx;
    }
    // model parameters fitted to rand_idx
    ePnP.reset_correspondences();
    for (int i = 0; i < NOM; i++)
    {
      ePnP.add_correspondence(objOutliers3D[rand_idx[i]].x, objOutliers3D[rand_idx[i]].y, objOutliers3D[rand_idx[i]].z, imgOutliers[rand_idx[i]].x, imgOutliers[rand_idx[i]].y);
    }
    double err = ePnP.compute_pose(R_est, T_est);

    // project rest points into the image plane
    CV_MAT_ELEM(*P, float, 0, 0) = R_est[0][0];
    CV_MAT_ELEM(*P, float, 0, 1) = R_est[0][1];
    CV_MAT_ELEM(*P, float, 0, 2) = R_est[0][2];

    CV_MAT_ELEM(*P, float, 1, 0) = R_est[1][0];
    CV_MAT_ELEM(*P, float, 1, 1) = R_est[1][1];
    CV_MAT_ELEM(*P, float, 1, 2) = R_est[1][2];

    CV_MAT_ELEM(*P, float, 2, 0) = R_est[2][0];
    CV_MAT_ELEM(*P, float, 2, 1) = R_est[2][1];
    CV_MAT_ELEM(*P, float, 2, 2) = R_est[2][2];

    CV_MAT_ELEM(*P, float, 0, 3) = T_est[0];
    CV_MAT_ELEM(*P, float, 1, 3) = T_est[1];
    CV_MAT_ELEM(*P, float, 2, 3) = T_est[2];

    cvGEMM(intrinsic_, P, 1, NULL, 0, P2, 0);

    // x2d_proj = P * x3d_h
    cvGEMM(P2, x3d_h, 1, NULL, 0, x2d_proj, 0);

    for (int i = 0; i < n; i++)
    {
      float u = CV_MAT_ELEM(*x2d_proj, float, 0, i);
      float v = CV_MAT_ELEM(*x2d_proj, float, 1, i);
      float w = CV_MAT_ELEM(*x2d_proj, float, 2, i);

      CV_MAT_ELEM(*x2d_proj, float, 0, i) = u / w;
      CV_MAT_ELEM(*x2d_proj, float, 1, i) = v / w;
      // save reprojection error to third rows
      CV_MAT_ELEM(*x2d_proj, float, 2, i) = sqrt((u / w - imgOutliers[i].x) * (u / w - imgOutliers[i].x) + (v / w - imgOutliers[i].y) * (v / w - imgOutliers[i].y));
    }

    // Count number of inliers
    int noi = 0;
    for (int i = 0; i < n; i++)
    {
      if (rand_idx[i] != i && CV_MAT_ELEM(*x2d_proj, float, 2, i) < th)
      {
        inlier_idx[i] = 1;
        noi++;
      }
      else
        inlier_idx[i] = 0;
    }

    if (noi > best_noi)
    {
      for (int i = 0; i < NOM; i++)
        inlier_idx[rand_idx[i]] = 1;
      best_noi = noi;
      best_inlier_idx = inlier_idx;
      // Determine adaptive number of iteration
      double e = 1. - (double)best_noi / (double)n;
      k = (int)(log(1. - p) / log(1. - pow(1. - e, NOM)));
    }

    iter++;
    if (verbose_)
      printf("(%d/%d) iter: %d/%d\n", iter, k, best_noi, n);
  }

  if (best_noi > 0)
  {
    ePnP.set_maximum_number_of_correspondences(best_noi + NOM);
    ePnP.reset_correspondences();
    for (int i = 0; i < n; i++)
    {
      if (best_inlier_idx[i])
        ePnP.add_correspondence(objOutliers3D[i].x, objOutliers3D[i].y, objOutliers3D[i].z, imgOutliers[i].x, imgOutliers[i].y);
    }

    double err = ePnP.compute_pose(R_est, T_est);

    CV_MAT_ELEM(*pmPose, float, 0, 0) = R_est[0][0];
    CV_MAT_ELEM(*pmPose, float, 1, 0) = R_est[1][0];
    CV_MAT_ELEM(*pmPose, float, 2, 0) = R_est[2][0];
    CV_MAT_ELEM(*pmPose, float, 3, 0) = 0.0;
    CV_MAT_ELEM(*pmPose, float, 0, 1) = R_est[0][1];
    CV_MAT_ELEM(*pmPose, float, 1, 1) = R_est[1][1];
    CV_MAT_ELEM(*pmPose, float, 2, 1) = R_est[2][1];
    CV_MAT_ELEM(*pmPose, float, 3, 1) = 0.0;
    CV_MAT_ELEM(*pmPose, float, 0, 2) = R_est[0][2];
    CV_MAT_ELEM(*pmPose, float, 1, 2) = R_est[1][2];
    CV_MAT_ELEM(*pmPose, float, 2, 2) = R_est[2][2];
    CV_MAT_ELEM(*pmPose, float, 3, 2) = 0.0;
    CV_MAT_ELEM(*pmPose, float, 0, 3) = T_est[0];
    CV_MAT_ELEM(*pmPose, float, 1, 3) = T_est[1];
    CV_MAT_ELEM(*pmPose, float, 2, 3) = T_est[2];
    CV_MAT_ELEM(*pmPose, float, 3, 3) = 1.0;
  }

// Display estimated pose
#if 0
  cout << "Found pose:" << endl;
  ePnP.print_pose(R_est, T_est);
#endif

  // Refined points
  objInliers.clear();
  imgInliers.clear();
  objInliers3D.clear();
  vector<CvPoint2D32f> pt1_out, pt2_out;
  vector<CvPoint3D32f> pt3_out;
  for (int i = 0; i < n; i++)
  {
    if (best_inlier_idx[i] == 1) // inliers only
    {
      objInliers.push_back(objOutliers[i]);
      imgInliers.push_back(imgOutliers[i]);
      objInliers3D.push_back(objOutliers3D[i]);
    }
    else // outliers
    {
      pt1_out.push_back(objOutliers[i]);
      pt2_out.push_back(imgOutliers[i]);
      pt3_out.push_back(objOutliers3D[i]);
    }
  }

  objOutliers = pt1_out;
  imgOutliers = pt2_out;
  objOutliers3D = pt3_out;

  cvReleaseMat(&P);
  cvReleaseMat(&P2);
  cvReleaseMat(&x3d_h);
  cvReleaseMat(&x2d_proj);

  return int(objInliers.size());
}

// EPnP for PF version
int CPoseEstimationSURF::PF_estimateMultiplePosesEpnp(const vector<int> &ptpairs, vector<CvPoint2D32f> &objOutliers, vector<CvPoint3D32f> &objOutliers3D, vector<CvPoint2D32f> &imgOutliers, vector<CvPoint2D32f> &objInliers, vector<CvPoint3D32f> &objInliers3D, vector<CvPoint2D32f> &imgInliers, vector<CvMat *> &states, int numOfParticle)
{
  // For accurate estimation 'NOM' should be larger than '6'
  const int NOM = 7; // number of model parameters

  int noc;
  int best_noi = 0;
  const float th = 20;
  const double p = 0.99;

  noc = int(ptpairs.size() / 2);
  if (noc < NOM)
    return -1;

  objOutliers.resize(noc);
  imgOutliers.resize(noc);
  objOutliers3D.resize(noc);
  for (int i = 0; i < noc; i++)
  {
    objOutliers[i] = keyframe_keypoints_2d_[ptpairs[i * 2]];
    imgOutliers[i] = input_keypoints_2d_[ptpairs[i * 2 + 1]];
    objOutliers3D[i] = keyframe_keypoints_3d_[ptpairs[i * 2]];
    //cout << objOutliers3D[i].x << " " << objOutliers3D[i].y << " " << objOutliers3D[i].z << endl;
  }

  epnp ePnP;

  float fu = CV_MAT_ELEM(*intrinsic_, float, 0, 0);
  float fv = CV_MAT_ELEM(*intrinsic_, float, 1, 1);
  float uc = CV_MAT_ELEM(*intrinsic_, float, 0, 2);
  float vc = CV_MAT_ELEM(*intrinsic_, float, 1, 2);

  vector<int> inlier_idx;
  inlier_idx.resize(noc);
  vector<int> best_inlier_idx;
  best_inlier_idx.resize(noc);

  ePnP.set_internal_parameters(uc, vc, fu, fv);
  ePnP.set_maximum_number_of_correspondences(NOM);

  CvRNG rng = cvRNG(cvGetTickCount());
  int rand_idx[NOM];
  CvMat *P = cvCreateMat(3, 4, CV_32F);
  CvMat *P2 = cvCreateMat(3, 4, CV_32F);
  CvMat *x3d_h = cvCreateMat(4, noc, CV_32F);
  CvMat *x2d_proj = cvCreateMat(3, noc, CV_32F);

  for (int i = 0; i < noc; i++)
  {
    CV_MAT_ELEM(*x3d_h, float, 0, i) = objOutliers3D[i].x;
    CV_MAT_ELEM(*x3d_h, float, 1, i) = objOutliers3D[i].y;
    CV_MAT_ELEM(*x3d_h, float, 2, i) = objOutliers3D[i].z;
    CV_MAT_ELEM(*x3d_h, float, 3, i) = 1.0;
  }

  double R_est[3][3], T_est[3];

  float *weight = new float[numOfParticle];

  for (int par = 0; par < numOfParticle; par++)
  {
    // sampling
    for (int i = 0; i < NOM; i++)
    {
      int temp_idx = 0;
      bool found = true;
      while (found)
      {
        temp_idx = cvRandInt(&rng) % noc;
        found = false;
        for (int j = 0; j < i; j++)
        {
          if (rand_idx[j] == temp_idx)
            found = true;
        }
      }
      rand_idx[i] = temp_idx;
    }
    // model parameters fitted to rand_idx
    ePnP.reset_correspondences();
    for (int i = 0; i < NOM; i++)
    {
      ePnP.add_correspondence(objOutliers3D[rand_idx[i]].x, objOutliers3D[rand_idx[i]].y, objOutliers3D[rand_idx[i]].z, imgOutliers[rand_idx[i]].x, imgOutliers[rand_idx[i]].y);
    }
    double err = ePnP.compute_pose_gn(R_est, T_est);

    // project rest points into the image plane
    CV_MAT_ELEM(*P, float, 0, 0) = R_est[0][0];
    CV_MAT_ELEM(*P, float, 0, 1) = R_est[0][1];
    CV_MAT_ELEM(*P, float, 0, 2) = R_est[0][2];

    CV_MAT_ELEM(*P, float, 1, 0) = R_est[1][0];
    CV_MAT_ELEM(*P, float, 1, 1) = R_est[1][1];
    CV_MAT_ELEM(*P, float, 1, 2) = R_est[1][2];

    CV_MAT_ELEM(*P, float, 2, 0) = R_est[2][0];
    CV_MAT_ELEM(*P, float, 2, 1) = R_est[2][1];
    CV_MAT_ELEM(*P, float, 2, 2) = R_est[2][2];

    CV_MAT_ELEM(*P, float, 0, 3) = T_est[0];
    CV_MAT_ELEM(*P, float, 1, 3) = T_est[1];
    CV_MAT_ELEM(*P, float, 2, 3) = T_est[2];

    cvGEMM(intrinsic_, P, 1, NULL, 0, P2, 0);

    // >> x2d_proj = P * x3d_h
    cvGEMM(P2, x3d_h, 1, NULL, 0, x2d_proj, 0);

    for (int i = 0; i < noc; i++)
    {
      float u = CV_MAT_ELEM(*x2d_proj, float, 0, i);
      float v = CV_MAT_ELEM(*x2d_proj, float, 1, i);
      float w = CV_MAT_ELEM(*x2d_proj, float, 2, i);

      CV_MAT_ELEM(*x2d_proj, float, 0, i) = u / w;
      CV_MAT_ELEM(*x2d_proj, float, 1, i) = v / w;
      // save reprojection error to third rows
      CV_MAT_ELEM(*x2d_proj, float, 2, i) = (u / w - imgOutliers[i].x) * (u / w - imgOutliers[i].x) + (v / w - imgOutliers[i].y) * (v / w - imgOutliers[i].y);
    }

    // count number of inliers
    int noi = 0;
    for (int i = 0; i < noc; i++)
    {
      if (rand_idx[i] != i && CV_MAT_ELEM(*x2d_proj, float, 2, i) < th * th)
      {
        inlier_idx[i] = 1;
        noi++;
      }
      else
        inlier_idx[i] = 0;
    }

    CV_MAT_ELEM(*states[par], float, 0, 0) = R_est[0][0];
    CV_MAT_ELEM(*states[par], float, 1, 0) = R_est[1][0];
    CV_MAT_ELEM(*states[par], float, 2, 0) = R_est[2][0];
    CV_MAT_ELEM(*states[par], float, 3, 0) = 0.0;
    CV_MAT_ELEM(*states[par], float, 0, 1) = R_est[0][1];
    CV_MAT_ELEM(*states[par], float, 1, 1) = R_est[1][1];
    CV_MAT_ELEM(*states[par], float, 2, 1) = R_est[2][1];
    CV_MAT_ELEM(*states[par], float, 3, 1) = 0.0;
    CV_MAT_ELEM(*states[par], float, 0, 2) = R_est[0][2];
    CV_MAT_ELEM(*states[par], float, 1, 2) = R_est[1][2];
    CV_MAT_ELEM(*states[par], float, 2, 2) = R_est[2][2];
    CV_MAT_ELEM(*states[par], float, 3, 2) = 0.0;
    CV_MAT_ELEM(*states[par], float, 0, 3) = T_est[0];
    CV_MAT_ELEM(*states[par], float, 1, 3) = T_est[1];
    CV_MAT_ELEM(*states[par], float, 2, 3) = T_est[2];
    CV_MAT_ELEM(*states[par], float, 3, 3) = 1.0;

    float lamda = 3.0f;
    weight[par] = exp(-lamda * (float)(noc - NOM - noi) / float(noc - NOM));
  }

  // randomly draw from weight
  float sum = 0.0f;
  for (int i = 0; i < numOfParticle; i++)
    sum += weight[i];
  for (int i = 0; i < numOfParticle; i++)
    weight[i] = weight[i] / sum; // normalize

  float *idx = new float[numOfParticle];
  float *cumsum = new float[numOfParticle + 1];
  assert(numOfParticle > 0);
  idx[0] = (float)rand() / (float)RAND_MAX / (float)numOfParticle;
  for (int i = 1; i < numOfParticle; i++)
    idx[i] = idx[i - 1] + 1.0f / (float)numOfParticle;
  cumsum[0] = 0.0f;
  for (int i = 1; i < numOfParticle + 1; i++)
    cumsum[i] = cumsum[i - 1] + weight[i - 1];

  int *outindex = new int[numOfParticle];
  for (int i = 0; i < numOfParticle; i++)
  {
    outindex[i] = 0;
  }
  for (int i = 0; i < numOfParticle; i++)
  {
    for (int j = 1; j < numOfParticle + 1; j++)
    {
      if (idx[i] > cumsum[j - 1] && idx[i] <= cumsum[j])
      {
        outindex[i] = j - 1;
        break;
      }
    }
  }

  // update resampled results to states
  for (int i = 0; i < numOfParticle; i++)
  {
    cvCopy(states[outindex[i]], states[i]);
  }

  delete[] idx;
  delete[] cumsum;
  delete[] outindex;
  delete[] weight;

  cvReleaseMat(&P);
  cvReleaseMat(&P2);
  cvReleaseMat(&x3d_h);
  cvReleaseMat(&x2d_proj);

  return noc;
}

void CPoseEstimationSURF::setImage(IplImage *img)
{
  // Update test image
  cvCopy(img, img_input_);
}
