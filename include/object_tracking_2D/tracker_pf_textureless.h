#pragma once

#include "TooN/TooN.h" //
#include "TooN/so3.h"  // for special orthogonal group
#include "TooN/se3.h"  // for special Euclidean group
#include "TooN/wls.h"  // for weighted least square
#include "TooN/Cholesky.h"

#include "tracker_pf.h"
#include <omp.h>
#include <vector>

class TexturelessParticleFilterTracker : public ParticleFilterTracker
{
public:
  TexturelessParticleFilterTracker()
      : pe_surf_(NULL), th_cm_(0.2f)
  {
  }

  virtual ~TexturelessParticleFilterTracker()
  {
  }

  virtual void clean()
  {
    TrackerBase::clean();
    if (pe_surf_)
      delete pe_surf_;
  }

  CvPoint2D32f project3Dto2D(CvPoint3D32f pt3, CvMat *pose, CvMat *param_intrinsic)
  {
    CvPoint2D32f pt2;
    CvPoint3D32f pt3_cam;
    pt3_cam.x = CV_MAT_ELEM(*pose, float, 0, 0) * pt3.x + CV_MAT_ELEM(*pose, float, 0, 1) * pt3.y + CV_MAT_ELEM(*pose, float, 0, 2) * pt3.z + CV_MAT_ELEM(*pose, float, 0, 3);
    pt3_cam.y = CV_MAT_ELEM(*pose, float, 1, 0) * pt3.x + CV_MAT_ELEM(*pose, float, 1, 1) * pt3.y + CV_MAT_ELEM(*pose, float, 1, 2) * pt3.z + CV_MAT_ELEM(*pose, float, 1, 3);
    pt3_cam.z = CV_MAT_ELEM(*pose, float, 2, 0) * pt3.x + CV_MAT_ELEM(*pose, float, 2, 1) * pt3.y + CV_MAT_ELEM(*pose, float, 2, 2) * pt3.z + CV_MAT_ELEM(*pose, float, 2, 3);

    float fx = CV_MAT_ELEM(*param_intrinsic, float, 0, 0);
    float fy = CV_MAT_ELEM(*param_intrinsic, float, 1, 1);
    float ux = CV_MAT_ELEM(*param_intrinsic, float, 0, 2);
    float uy = CV_MAT_ELEM(*param_intrinsic, float, 1, 2);

    // jiaming hu: adding extrinsic matrix
    float tx = 0.0; //-148.43597135826298;
    float ty = 0.0;

    pt2.x = fx * pt3_cam.x / pt3_cam.z + ux + tx / pt3_cam.z;
    pt2.y = fy * pt3_cam.y / pt3_cam.z + uy + ty / pt3_cam.z;

    float th_d = 10000.0;
    if (pt2.x < -th_d || pt2.x > th_d || pt2.y < -th_d || pt2.y > th_d)
    {
      pt2.x = -1;
      pt2.y = -1;
    }

    return pt2;
  }

  virtual bool initTracker(std::string &obj_name, std::string &cam_name, std::string &intrinsic, std::string &distortion, int width, int height, CvMat *pose_init, std::string ach_channel)
  {
    TrackerBase::initTracker(obj_name, cam_name, intrinsic, distortion, width, height, pose_init, ach_channel);

    initPoseEstimationSURF(width, height, obj_name, obj_name);

    pf_->Init(pose_init);

    return (true);
  }

  void generate_tracker(TrackerBase *cloned_tracker, CvMat *pose_init)
  {
    TrackerBase::generate_tracker(cloned_tracker, pose_init);
    pe_surf_ = ((TexturelessParticleFilterTracker *)cloned_tracker)->pe_surf_;
    pf_->Init(pose_init);
  }

  inline void setThresholdCM(float th) { th_cm_ = th; };
  inline float getThresholdCM() { return th_cm_; };

  void rotate(CvMat *curr_pose, float x, float y, float z)
  {
    Mat R_x = (Mat_<float>(4, 4) << 1, 0, 0, 0,
               0, cos(x), -sin(x), 0,
               0, sin(x), cos(x), 0,
               0, 0, 0, 1);

    // Calculate rotation about y axis
    Mat R_y = (Mat_<float>(4, 4) << cos(y), 0, sin(y), 0,
               0, 1, 0, 0,
               -sin(y), 0, cos(y), 0,
               0, 0, 0, 1);

    // Calculate rotation about z axis
    Mat R_z = (Mat_<float>(4, 4) << cos(z), -sin(z), 0, 0,
               sin(z), cos(z), 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1);
    cv::Mat temp = R_z * R_y * R_x;
    for (int r = 0; r < 4; r++)
    {
      for (int c = 0; c < 4; c++)
      {
        CV_MAT_ELEM(*curr_pose, float, r, c) = temp.at<float>(r, c);
      }
    }
  }

  int pose_detection(int numOfDetections, std::vector<LMDetWind> detWind, std::vector<CvMat *> &detectedStates)
  {
    if (obj_model_->getNumOfEdgeTemplates() > 0)
    {
      pe_surf_->setImage(img_gray_);

      // calculate distance transform
      /*
      cv::Mat dt_input = cv::cvarrToMat(img_gray_);

      cv::Mat binary(dt_input.size(), dt_input.type());
      cv::Mat dt_image;

      cv::threshold(dt_input, binary, 100, 255, cv::THRESH_BINARY_INV);
      cv::distanceTransform(binary, dt_image, CV_DIST_L2, 5);
*/

      std::vector<CvMat *> resultPose;

      // estimate a set of poses based on random corresondences
      int numof_det = pe_surf_->PF_estimatePosesFDCM(th_cm_, numOfDetections, resultPose, detWind, smooth_size_, th_canny_l_, th_canny_h_, img_mask_);
      if (numof_det <= 0)
      {
        return numof_det;
      }

      obj_model_->extractEdge(img_gray_, smooth_size_, th_canny_l_, th_canny_h_, cam_->getEdge(), img_mask_);
      obj_model_->extractEdgeOri(img_gray_, smooth_size_);

      CvMat *intri_mat = cam_->getIntrinsicParams();
      const float match_rate_threshold = 3.6;
      const float c = 64.f;
      //IplImage *checkimg = cvCreateImage(cvGetSize(img_gray_), 8, 3);
      IplImage *checkimg = cvCreateImage(cvGetSize(img_gray_), 8, 1);
      CvMat *rotatestate = cvCreateMat(4, 4, CV_32F);
      CvMat *resultstate = cvCreateMat(4, 4, CV_32F);
      CvMat *maxmatchstate = cvCreateMat(4, 4, CV_32F);
      for (int d = 0; d < numof_det; d++)
      {

        //cvCvtColor(img_gray_, checkimg, CV_GRAY2BGR);
        cvZero(checkimg);
        float maxmatch = 0.0;

        float match_threshold = match_rate_threshold / CV_MAT_ELEM(*(resultPose[d]), float, 2, 3);
        if (match_threshold > maxd_)
          match_threshold = maxd_;

        float tempmatchrate = 0.0;
        for (int approachstep = 0; approachstep < 1; approachstep++)
        {
          obj_model_->setModelviewMatrix(resultPose[d]);
          obj_model_->findVisibleSamplePoints();
          obj_model_->getVisibleArea(height_, width_);
          obj_model_->keepOnlyContourPoints();

          /* draw the contour */
          IplImage tempimg = obj_model_->contourimg;
          cvCanny(&tempimg, checkimg, th_canny_l_, th_canny_h_);

          /* draw the contour */

          int numOfVisiblePoints = obj_model_->getVisibleSamplePoints().size();
          std::cout << "number of visible sample points on the contour = " << numOfVisiblePoints << std::endl;

          if (numOfVisiblePoints > 6)
          {

            obj_model_->findNormalUsingEdgeCoordCoarseOri();
            obj_model_->findEdgeCorrespondencesCoarseOri();
            if (th_ransac_ > 0.0f)
            {
              obj_model_->refineEdgeCorrespondences_RANSAC(resultPose[d], th_ransac_iter_, th_ransac_);
            }

            CvMat *J = NULL, *e = NULL;

            edge_tracker_->PF_getJacobianAndError(resultPose[d], obj_model_->getVisibleSamplePoints(), &J, &e);
            if (J == NULL)
            {
              continue;
            }
            int N = J->rows;

            CvMat *W = cvCreateMat(N, N, CV_32F);
            CvMat *Jt_W = cvCreateMat(6, N, CV_32F);
            CvMat *Jt_W_J = cvCreateMat(6, 6, CV_32F);
            CvMat *Jt_W_J_inv = cvCreateMat(6, 6, CV_32F);
            CvMat *Jt_W_e = cvCreateMat(6, 1, CV_32F);
            CvMat *inc = cvCreateMat(6, 1, CV_32F);

            cvSetIdentity(W);
            for (int j = 0; j < N; j++)
            {
              CV_MAT_ELEM(*W, float, j, j) = 1.0f / (c + CV_MAT_ELEM(*e, float, j, 0));
            }

            cvGEMM(J, W, 1.0, NULL, 0.0, Jt_W, CV_GEMM_A_T); // J'*W
            cvGEMM(Jt_W, J, 1.0, NULL, 0.0, Jt_W_J);         // J'*W*J
            cvInvert(Jt_W_J, Jt_W_J_inv);                    // inv(J'*W*J)
            cvGEMM(Jt_W, e, 1.0, NULL, 0.0, Jt_W_e);         // J'*W*e
            cvMatMul(Jt_W_J_inv, Jt_W_e, inc);

            double inc_data[6];
            for (int j = 0; j < 6; j++)
              inc_data[j] = 2 * CV_MAT_ELEM(*inc, float, j, 0);

            TooN::Vector<6> inc_vec(inc_data);
            SE3 M_inc = SE3::exp(inc_vec);

            Matrix<4> M_prop;
            for (int r = 0; r < 4; r++)
              for (int c = 0; c < 4; c++)
                M_prop[r][c] = (double)CV_MAT_ELEM(*resultPose[d], float, r, c);

            Matrix<4> M_opt = M_prop * M_inc;

            for (int r = 0; r < 4; r++)
              for (int c = 0; c < 4; c++)
                CV_MAT_ELEM(*resultPose[d], float, r, c) = (float)M_opt[r][c];

            if (J)
              cvReleaseMat(&J);
            if (e)
              cvReleaseMat(&e);

            cvReleaseMat(&W);
            cvReleaseMat(&Jt_W);
            cvReleaseMat(&Jt_W_J);
            cvReleaseMat(&Jt_W_J_inv);
            cvReleaseMat(&Jt_W_e);
            cvReleaseMat(&inc);
          }
        }
        /*
        for(float xangle = -0.2 ; xangle <= 0.2 ; xangle += 0.2){
          for(float yangle = -0.2 ; yangle <= 0.2 ; yangle += 0.2){
            for(float zangle = -0.2 ; zangle <= 0.2 ; zangle += 0.2){
              
              rotate(rotatestate, xangle, yangle, zangle);
              cvMatMul(resultPose[d], rotatestate, resultstate);
              obj_model_->setModelviewMatrix(resultstate);
              obj_model_->findVisibleSamplePoints();
              obj_model_->getVisibleArea(height_, width_);
              obj_model_->keepOnlyContourPoints();

              int numOfVisiblePoints = obj_model_->getVisibleSamplePoints().size();
              int numofInner = 0;

              if(numOfVisiblePoints > 6 ){

                obj_model_->findNormalUsingEdgeCoordCoarseOri();
                obj_model_->findEdgeCorrespondencesCoarseOri();
                CvMat *J = NULL, *e = NULL;

                edge_tracker_->PF_getJacobianAndError(resultstate, obj_model_->getVisibleSamplePoints(), &J, &e);

                if(J == NULL){
                  continue;
                }

                numofInner = 0;
                for(int vp = 0 ; vp < obj_model_->getVisibleSamplePoints().size() ; vp++){
                  if((obj_model_->getVisibleSamplePoints())[vp].dist < match_threshold){
                    numofInner++;
                  }
                }
                
                if((float)numofInner/numOfVisiblePoints > (tempmatchrate * 0.9)){
                  if((float)numofInner/numOfVisiblePoints > tempmatchrate)
                    tempmatchrate = (float)numofInner/numOfVisiblePoints;
                }
                else{
                  continue;
                }

                int N = J->rows;

                CvMat* W = cvCreateMat(N, N, CV_32F);
                CvMat* Jt_W = cvCreateMat(6, N, CV_32F);
                CvMat* Jt_W_J = cvCreateMat(6, 6, CV_32F);
                CvMat* Jt_W_J_inv = cvCreateMat(6, 6, CV_32F);
                CvMat* Jt_W_e = cvCreateMat(6, 1, CV_32F);
                CvMat* inc = cvCreateMat(6, 1, CV_32F);

                cvSetIdentity(W);
                for(int j=0; j<N; j++)
                {
                  CV_MAT_ELEM(*W, float, j, j) = 1.0f/(c+CV_MAT_ELEM(*e, float, j, 0));
                }

                cvGEMM(J, W, 1.0, NULL, 0.0, Jt_W, CV_GEMM_A_T); // J'*W
                cvGEMM(Jt_W, J, 1.0, NULL, 0.0, Jt_W_J); // J'*W*J
                cvInvert(Jt_W_J, Jt_W_J_inv); // inv(J'*W*J)
                cvGEMM(Jt_W, e, 1.0, NULL, 0.0, Jt_W_e); // J'*W*e
                cvMatMul(Jt_W_J_inv, Jt_W_e, inc);

                double inc_data[6];
                for(int j=0; j<6; j++)
                  inc_data[j] = 2*CV_MAT_ELEM(*inc, float, j, 0);

                TooN::Vector<6> inc_vec(inc_data);
                SE3 M_inc = SE3::exp(inc_vec);

                Matrix<4> M_prop;
                for(int r=0; r<4; r++)
                  for(int c=0; c<4; c++)
                    M_prop[r][c] = (double)CV_MAT_ELEM(*resultstate, float, r, c);

                Matrix<4> M_opt = M_prop * M_inc;

                for(int r=0; r<4; r++)
                  for(int c=0; c<4; c++)
                    CV_MAT_ELEM(*resultstate, float, r, c) = (float)M_opt[r][c];

                if(J) cvReleaseMat(&J);
                if(e) cvReleaseMat(&e);

                cvReleaseMat(&W);
                cvReleaseMat(&Jt_W);
                cvReleaseMat(&Jt_W_J);
                cvReleaseMat(&Jt_W_J_inv);
                cvReleaseMat(&Jt_W_e);
                cvReleaseMat(&inc);

                obj_model_->setModelviewMatrix(resultstate);
                obj_model_->findVisibleSamplePoints();
                obj_model_->getVisibleArea(height_, width_);
                obj_model_->keepOnlyContourPoints();

                obj_model_->findNormalUsingEdgeCoordFineOri();
                obj_model_->findEdgeCorrespondencesFineOri();

                numOfVisiblePoints = obj_model_->getVisibleSamplePoints().size();
                numofInner = 0;
                for(int vp = 0 ; vp < numOfVisiblePoints ; vp++){
                  if((obj_model_->getVisibleSamplePoints())[vp].dist < match_threshold){
                    numofInner++;
                  }
                }
                
                float matchrate = (float)numofInner/numOfVisiblePoints;

                if(matchrate > maxmatch){

                  maxmatch = matchrate;
                  for(int r=0; r<4; r++){ 
                    for(int c=0; c<4; c++){
                      CV_MAT_ELEM(*maxmatchstate, float, r, c) = CV_MAT_ELEM(*resultstate, float, r, c);
                    }
                  }
                }
              }
            }
          }
        }
        

        // update new state
        for (int r = 0; r < 4; r++)
        {
          for (int c = 0; c < 4; c++)
          {
            CV_MAT_ELEM(*(resultPose[d]), float, r, c) = CV_MAT_ELEM(*maxmatchstate, float, r, c);
          }
        }
*/

        obj_model_->setModelviewMatrix(resultPose[d]);
        obj_model_->findVisibleSamplePoints();
        obj_model_->getVisibleArea(height_, width_);
        obj_model_->keepOnlyContourPoints();
        obj_model_->findNormalUsingEdgeCoord();
        obj_model_->findEdgeCorrespondences();

        if (obj_model_->getVisibleSamplePoints().size() > 0)
        {
          int detectionpoint = 1;
          cvCircle(checkimg, cvPointFrom32f((obj_model_->getVisibleSamplePoints())[0].coord2), 3, cvScalar(255), -1, CV_AA, 0);
          float lastx = (obj_model_->getVisibleSamplePoints())[0].coord2.x;
          float lasty = (obj_model_->getVisibleSamplePoints())[0].coord2.y;

          for (int vp = 1; vp < obj_model_->getVisibleSamplePoints().size(); vp++)
          {
            //if ((obj_model_->getVisibleSamplePoints())[vp].dist < match_threshold)
            float distanceinx = std::abs((obj_model_->getVisibleSamplePoints())[vp].coord2.x - lastx);
            float distanceiny = std::abs((obj_model_->getVisibleSamplePoints())[vp].coord2.y - lasty);
            if (distanceinx * distanceinx + distanceiny * distanceiny > 100 )
            {
              cvCircle(checkimg, cvPointFrom32f((obj_model_->getVisibleSamplePoints())[vp].coord2), 3, cvScalar(255), -1, CV_AA, 0);
              lastx = (obj_model_->getVisibleSamplePoints())[vp].coord2.x;
              lasty = (obj_model_->getVisibleSamplePoints())[vp].coord2.y;
              detectionpoint++;
            }
          }
          std::cout << "number of avaliable tracking points is " <<  detectionpoint << std::endl;
        }

        //std::cout << "number of visible points = " << obj_model_->getVisibleSamplePoints().size() << std::endl;
        std::cout << d << " best matching rate: " << maxmatch << std::endl;
        cvSaveImage(("check/maxmatch" + std::to_string(d) + ".png").c_str(), checkimg);
        if (maxmatch >= 0.0)
        {
          detectedStates.push_back(resultPose[d]);
        }
        else
        {
          cvReleaseMat(&(resultPose[d]));
        }
      }
      cvReleaseMat(&rotatestate);
      cvReleaseMat(&resultstate);
      cvReleaseMat(&maxmatchstate);
      cvReleaseImage(&checkimg);

      if (detectedStates.size() > 0)
      {
        init_ = false;
        return detectedStates.size();
      }
      else
      {
        init_ = true;
        return 0;
      }
    }
    else
    {
      return -1;
    }
  }

  void resetImage()
  {
    cvCvtColor(img_gray_, img_result_, CV_GRAY2BGR);
  }

protected:
  CPoseEstimationSURF *pe_surf_;
  float th_cm_;

  virtual bool initObjectModel(std::string name, int width, int height, CvMat *intrinsic, float sample_step, int maxd, bool dulledge, CEdgeTracker *edge_tracker)
  {
    TrackerBase::initObjectModel(name, width, height, intrinsic, sample_step, maxd, dulledge, edge_tracker);

    obj_model_->loadKeyframes(name);

    return (true);
  }

  bool initPoseEstimationSURF(int width, int height, std::string data_name, std::string &obj_name)
  {
    if (pe_surf_)
      delete pe_surf_;

    pe_surf_ = new CPoseEstimationSURF(
        width,
        height,
        data_name,
        obj_model_,
        cam_->getIntrinsicParams(),
        cam_->getDistortionParams(),
        obj_name);

    pe_surf_->buildKdTree(
        obj_model_->getKeyframeImages(),
        obj_model_->getKeyframePoses(),
        obj_model_->getKeyframeKeypoints2D(),
        obj_model_->getKeyframeKeypoints3D(),
        obj_model_->getKeyframeDescriptors());
    return (true);
  }

  virtual void displayResults()
  {
    ParticleFilterTracker::displayResults();

    //cvSaveImage("/home/jiaming/catkin_ws/pf_image.png", pe_surf_->getImage());
    /*
    if(display_init_result_)
    {
      cvShowImage("Initialization", pe_surf_->getImage());
      cvWaitKey(1500);
      display_init_result_ = false;
    }
    */
  }

  virtual bool initialize()
  {
    TrackerBase::initialize();
    display_init_result_ = true;
    init_ = false;
    return false;
  }

  virtual void handleKey(char key)
  {
    switch (key)
    {
    case 'r':
    case 'R':
      if (!init_)
        init_ = true;
      break;
    case 't':
    case 'T':
      if (init_)
        init_ = false;
      break;
    case 27:
      run_ = false;
      break;
    }
  }

  virtual int tracking()
  {
    // do annealing process only after (re-)initialization
    int num_anneal_level = frame_num_after_init_ == 0 ? num_annealing_layers_ : 1;

    for (int l = num_anneal_level - 1; l >= 0; l--)
    {
      // 'getEdge' returns Berkeley edge if it is available, otherwise returns NULL
      // 'extractEdge' extracts Canny edge if the fourth edge is NULL

      obj_model_->extractEdge(img_gray_, smooth_size_, th_canny_l_, th_canny_h_, cam_->getEdge(), img_mask_);
      obj_model_->extractEdgeOri(img_gray_, smooth_size_);

      // reset previous drawn image
      //if(display_)
      //  cvCvtColor(img_gray_, img_result_, CV_GRAY2BGR); // shoud be changed in better way

      if (num_anneal_level == 1)
        pf_->Propagate(noise_l_, noise_h_, true);
      else
        pf_->Propagate(alpha_[l], alpha_[l], l == num_anneal_level - 1 ? true : false);

      float currentMatchrate = 0.0;
      for (int p = 0; p < pf_->GetNumOfParticle(); p++)
      {
        // update the initial pose to object model for displaying
        obj_model_->setModelviewMatrix(pf_->GetPropState(p));
        // draw object model with visibility test
        obj_model_->findVisibleSamplePoints();
        // find normal of each sampling point
        obj_model_->findNormalUsingEdgeCoord();
        // calculate error between sampling points and nearest edge
        obj_model_->findEdgeCorrespondences();
        if (th_ransac_ > 0.0f)
        {
          obj_model_->refineEdgeCorrespondences_RANSAC(pf_->GetPropState(p), th_ransac_iter_, th_ransac_);
        }

        // consider edge sample points only
        CvMat *J = NULL, *e = NULL;
        edge_tracker_->PF_getJacobianAndError(pf_->GetPropState(p), obj_model_->getVisibleSamplePoints(), &J, &e);

        float match_threshold = 2.0 / CV_MAT_ELEM(*(pf_->GetPropState(p)), float, 2, 3);
        if (match_threshold > maxd_)
          match_threshold = maxd_;
        int numofInner = 0;
        for (int vp = 0; vp < obj_model_->getVisibleSamplePoints().size(); vp++)
        {
          if ((obj_model_->getVisibleSamplePoints())[vp].dist < match_threshold)
          {
            numofInner++;
          }
        }
        if (currentMatchrate < (float)numofInner / obj_model_->getVisibleSamplePoints().size())
          currentMatchrate = (float)numofInner / obj_model_->getVisibleSamplePoints().size();

        pf_->Update_IRLS(p, J, e, obj_model_->getNumberOfVisibleSamplePoints());
        // calculate weights
        pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(), maxd_, lamda_e_, lamda_v_);
        // release after use them
        if (J)
          cvReleaseMat(&J);
        if (e)
          cvReleaseMat(&e);

        // calculate weight (likelihood now) for optimized particles
        if (pf_->GetNumOfParticle() > 1)
        {
          // update the initial pose to object model for displaying
          obj_model_->setModelviewMatrix(pf_->GetOptState(p));
          // draw object model with visibility test
          obj_model_->findVisibleSamplePoints();
          // find normal of each sampling point
          obj_model_->findNormalUsingEdgeCoord();
          // calculate error between sampling points and nearest edge
          obj_model_->findEdgeCorrespondences();

          if (th_ransac_ > 0.0f)
            obj_model_->refineEdgeCorrespondences_RANSAC(pf_->GetOptState(p), th_ransac_iter_, th_ransac_);
          e = NULL;
          edge_tracker_->PF_getError(pf_->GetOptState(p), obj_model_->getVisibleSamplePoints(), &e);
          // Calculate weights for optimized particles
          pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(), maxd_, lamda_e_, lamda_v_, true);
          if (e)
            cvReleaseMat(&e);
        }
      }

      //std::cout << "  current match rate is " << currentMatchrate << std::endl;
      if (currentMatchrate < 0.65)
        return 1;

      // correct weights caused from considering optimized states
      if (pf_->GetNumOfParticle() > 1)
        pf_->CorrectWeights();

      // resampling
      bool valid;
      if (pf_->GetNumOfParticle() > 1)
        valid = pf_->ResampleOpt(beta_[l], num_anneal_level == 1 ? true : false, true); // and calculate particle mean
      else
        valid = pf_->Resample(beta_[l], num_anneal_level == 1 ? true : false, true); // and calculate particle mean

      if (valid) // && th_neff_ratio_*static_cast<float>(pf_->GetNumOfParticle()) < pf_->GetNeff())
      {
        mutex_.lock();
        cvCopy(pf_->GetMeanState(), pose_);
        mutex_.unlock();
      }
      else
      {
        // particle filter results is not valid
        init_ = true; // (re-)init again
        break;        // stop annealing process
      }
    }

    // reset 'previous state of particles' to 'current state of particles' right after annealing
    if (num_anneal_level > 1 && !init_)
    {
      for (int p = 0; p < pf_->GetNumOfParticle(); p++)
        cvCopy(pf_->GetState(p), pf_->GetPrevState(p));
    }
    obj_model_->getVisibleArea(height_, width_);
    return 0;
  }
};
