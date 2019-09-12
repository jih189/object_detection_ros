#pragma once

#include "tracker_pf.h"
#include <omp.h>

class TexturelessParticleFilterTracker : public ParticleFilterTracker
{
public:
  TexturelessParticleFilterTracker()
    : pe_surf_(NULL)
    , th_cm_(0.2f)
  {

  }

  virtual ~TexturelessParticleFilterTracker()
  {
  }

  virtual void clean(){
    TrackerBase::clean();
    if(pe_surf_) delete pe_surf_;
  }

  virtual bool initTracker(std::string &obj_name, std::string &cam_name, std::string &intrinsic, std::string &distortion, int width, int height, CvMat* pose_init, std::string ach_channel)
  {
    TrackerBase::initTracker(obj_name, cam_name, intrinsic, distortion, width, height, pose_init, ach_channel);

    initPoseEstimationSURF(width, height, obj_name, obj_name);

    pf_->Init(pose_init);

    return (true);
  }

  void generate_tracker(TrackerBase* cloned_tracker, CvMat* pose_init) {
    TrackerBase::generate_tracker(cloned_tracker, pose_init);
    pe_surf_ = ((TexturelessParticleFilterTracker*)cloned_tracker)->pe_surf_;
    pf_->Init(pose_init);
  }

  inline void setThresholdCM(float th) { th_cm_ = th; };
  inline float getThresholdCM() { return th_cm_; };

  void rotate(CvMat* curr_pose, float x, float y, float z){
    Mat R_x = (Mat_<float>(4,4) <<
               1,       0,        0,          0,
               0,       cos(x),   -sin(x),    0,
               0,       sin(x),   cos(x),     0,
               0,       0,        0,          1
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<float>(4,4) <<
               cos(y),    0,      sin(y),     0,
               0,         1,      0,          0,
               -sin(y),   0,      cos(y),     0,
               0,         0,      0,          1
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<float>(4,4) <<
               cos(z),    -sin(z),      0,    0,
               sin(z),    cos(z),       0,    0,
               0,         0,            1,    0,
               0,         0,            0,    1);
    cv::Mat temp = R_z * R_y * R_x;
    for(int r=0; r<4; r++){ 
      for(int c=0; c<4; c++){
        CV_MAT_ELEM(*curr_pose, float, r, c) = temp.at<float>(r,c);
      }
    }
  }

  int pose_detection(int numOfDetections, std::vector<LMDetWind> detWind, std::vector<CvMat*>& detectedStates){
    if(obj_model_->getNumOfEdgeTemplates() > 0)
    {
      pe_surf_->setImage(img_gray_);

      // calculate distance transform
      cv::Mat dt_input = cv::cvarrToMat(img_gray_);

      cv::Mat binary(dt_input.size(), dt_input.type());
      cv::Mat dt_image;

      cv::threshold(dt_input, binary, 100, 255, cv::THRESH_BINARY_INV);
      cv::distanceTransform(binary, dt_image, CV_DIST_L2, 5);
      //normalize(dt_image, dt_image, 0.0, 1.0, NORM_MINMAX);

      cv::imwrite("distance_img.png", dt_image);
      
      // estimate a set of poses based on random corresondences
      int numof_det = pe_surf_->PF_estimatePosesFDCM(th_cm_, numOfDetections, detectedStates, detWind, smooth_size_, th_canny_l_, th_canny_h_, img_mask_);
      std::cout << "number of det is " << numof_det << std::endl;

      obj_model_->extractEdge(img_gray_, smooth_size_, th_canny_l_, th_canny_h_, cam_->getEdge(), img_mask_);
      obj_model_->extractEdgeOri(img_gray_, smooth_size_);

      IplImage * checkimg = cvCreateImage(cvGetSize(img_gray_), 8, 3);
      cvCvtColor(img_gray_, checkimg, CV_GRAY2BGR);

      CvMat * teststate = cvCreateMat(4,4,CV_32F);
      CvMat * resultstate = cvCreateMat(4,4,CV_32F);
      CvMat * maxmatchstate = cvCreateMat(4,4,CV_32F);
      cvSetIdentity(maxmatchstate);

      for(int d = 0 ; d < numof_det ; d++){
        float maxmatch = 0.0;
        for(float xd = -0.2 ; xd <= 0.2 ; xd += 0.1){
          for(float yd = -0.2 ; yd <= 0.2 ; yd += 0.1){
            for(float xangle = -0.9 ; xangle <= 0.9 ; xangle += 0.3){
              for(float yangle = -0.4 ; yangle <= 0.4 ; yangle += 0.2){
                for(float zangle = -0.9 ; zangle <= 0.9 ; zangle += 0.3){
                  rotate(teststate, xangle, yangle, zangle);
                  cvMatMul(detectedStates[d], teststate, resultstate);
                  CV_MAT_ELEM(*resultstate, float, 0, 3) = CV_MAT_ELEM(*(detectedStates[d]), float, 0, 3) + xd * CV_MAT_ELEM(*(detectedStates[d]), float, 2, 3) / 10;
                  CV_MAT_ELEM(*resultstate, float, 1, 3) = CV_MAT_ELEM(*(detectedStates[d]), float, 1, 3) + yd * CV_MAT_ELEM(*(detectedStates[d]), float, 2, 3) / 10;

                  obj_model_->setModelviewMatrix(resultstate);
                  obj_model_->findVisibleSamplePoints();

                  float numofInnerDT = 0;
                  for(int vp = 0 ; vp < obj_model_->getVisibleSamplePoints().size() ; vp++){

                    if(dt_image.at<float>((obj_model_->getVisibleSamplePoints())[vp].coord2) < 2.0){
                      numofInnerDT++;
                    }
                  }
              
                  if(obj_model_->getVisibleSamplePoints().size() != 0){
                    float DTmatchrate = numofInnerDT/(obj_model_->getVisibleSamplePoints().size());
                    //std::cout << "distance transform match rate " << DTmatchrate << std::endl;
                    if(DTmatchrate > maxmatch){
                      maxmatch = DTmatchrate;
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
          }
        }

        // update new state
        for(int r=0; r<4; r++){ 
          for(int c=0; c<4; c++){
            CV_MAT_ELEM(*(detectedStates[d]), float, r, c) = CV_MAT_ELEM(*maxmatchstate, float, r, c);
          }
        }

        obj_model_->setModelviewMatrix(maxmatchstate); 
        // draw object model with visibility test
        obj_model_->findVisibleSamplePoints();
        // find normal of each sampling point
        obj_model_->findNormalUsingEdgeCoord();
        // calculate error between sampling points and nearest edge
        obj_model_->findEdgeCorrespondences();
        for(int vp = 0 ; vp < obj_model_->getVisibleSamplePoints().size() ; vp++){
          if((obj_model_->getVisibleSamplePoints())[vp].dist < 8)
            cvCircle(checkimg, cvPointFrom32f((obj_model_->getVisibleSamplePoints())[vp].coord2), 1, CV_RGB(0,255,0), -1, CV_AA, 0);
          else
            cvCircle(checkimg, cvPointFrom32f((obj_model_->getVisibleSamplePoints())[vp].coord2), 1, CV_RGB(255,0,0), -1, CV_AA, 0);
        }
      }

      cvSaveImage("maxmatch.png", checkimg);
      
      cvReleaseImage(&checkimg);
      cvReleaseMat(&teststate);
      cvReleaseMat(&resultstate);
      cvReleaseMat(&maxmatchstate);

      if(numof_det > 0)
      {
        init_ = false;
        return numof_det;
      }
      else{
        init_ = true;
        return 0;
      }
    }
    else{
      return -1;
    }
  }

  void resetImage(){
      cvCvtColor(img_gray_, img_result_, CV_GRAY2BGR);
  }

protected:
  CPoseEstimationSURF* pe_surf_;
  float th_cm_;

  virtual bool initObjectModel(std::string name, int width, int height, CvMat* intrinsic, float sample_step, int maxd, bool dulledge, CEdgeTracker* edge_tracker)
  {
    TrackerBase::initObjectModel(name, width, height, intrinsic, sample_step, maxd, dulledge, edge_tracker);

    obj_model_->loadKeyframes(name);

    return (true);
  }

  bool initPoseEstimationSURF(int width, int height, std::string data_name, std::string &obj_name)
  {
    if(pe_surf_) delete pe_surf_;
    
    pe_surf_ = new CPoseEstimationSURF(
      width, 
      height, 
      data_name, 
      obj_model_, 
      cam_->getIntrinsicParams(), 
      cam_->getDistortionParams(), 
      obj_name
    );

    pe_surf_->buildKdTree(
      obj_model_->getKeyframeImages(), 
      obj_model_->getKeyframePoses(), 
      obj_model_->getKeyframeKeypoints2D(), 
      obj_model_->getKeyframeKeypoints3D(), 
      obj_model_->getKeyframeDescriptors()
    );
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
    switch(key)
    {
    case 'r':
    case 'R':
      if(!init_)
        init_ = true;
      break;
    case 't':
    case 'T':
      if(init_)
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

    for(int l = num_anneal_level-1; l >= 0; l--)
    {
      // 'getEdge' returns Berkeley edge if it is available, otherwise returns NULL
      // 'extractEdge' extracts Canny edge if the fourth edge is NULL


      obj_model_->extractEdge(img_gray_, smooth_size_, th_canny_l_, th_canny_h_, cam_->getEdge(), img_mask_);
      obj_model_->extractEdgeOri(img_gray_, smooth_size_);

      // reset previous drawn image
      //if(display_)
      //  cvCvtColor(img_gray_, img_result_, CV_GRAY2BGR); // shoud be changed in better way

      if(num_anneal_level == 1)
        pf_->Propagate(noise_l_, noise_h_, true);
      else
        pf_->Propagate(alpha_[l], alpha_[l], l == num_anneal_level-1 ? true : false);
   
      for(int p = 0; p < pf_->GetNumOfParticle(); p++)
      {
        // update the initial pose to object model for displaying
        obj_model_->setModelviewMatrix(pf_->GetPropState(p));
        // draw object model with visibility test
        obj_model_->findVisibleSamplePoints();
        // find normal of each sampling point
        obj_model_->findNormalUsingEdgeCoord();
        // calculate error between sampling points and nearest edge
        obj_model_->findEdgeCorrespondences();
        if(th_ransac_ > 0.0f){
          obj_model_->refineEdgeCorrespondences_RANSAC(pf_->GetPropState(p), th_ransac_iter_, th_ransac_);
        }

        // consider edge sample points only
        CvMat *J = NULL, *e = NULL;
        edge_tracker_->PF_getJacobianAndError(pf_->GetPropState(p), obj_model_->getVisibleSamplePoints(), &J, &e);
        pf_->Update_IRLS(p, J, e, obj_model_->getNumberOfVisibleSamplePoints());
        // calculate weights
        pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(), maxd_, lamda_e_, lamda_v_);
        // release after use them
        if(J) cvReleaseMat(&J);
        if(e) cvReleaseMat(&e);

        // calculate weight (likelihood now) for optimized particles
        if(pf_->GetNumOfParticle() > 1)
        {
          // update the initial pose to object model for displaying
          obj_model_->setModelviewMatrix(pf_->GetOptState(p));
          // draw object model with visibility test
          obj_model_->findVisibleSamplePoints();
          // find normal of each sampling point
          obj_model_->findNormalUsingEdgeCoord();
          // calculate error between sampling points and nearest edge
          obj_model_->findEdgeCorrespondences();

          
          if(th_ransac_ > 0.0f)
            obj_model_->refineEdgeCorrespondences_RANSAC(pf_->GetOptState(p), th_ransac_iter_, th_ransac_);
          e = NULL;
          edge_tracker_->PF_getError(pf_->GetOptState(p), obj_model_->getVisibleSamplePoints(), &e);
          // Calculate weights for optimized particles
          pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(), maxd_, lamda_e_, lamda_v_, true);
          if(e) cvReleaseMat(&e);
        }
      }

      // correct weights caused from considering optimized states
      if(pf_->GetNumOfParticle() > 1)
        pf_->CorrectWeights();

      // resampling
      bool valid;
      if(pf_->GetNumOfParticle() > 1)
        valid = pf_->ResampleOpt(beta_[l], num_anneal_level == 1? true : false, true); // and calculate particle mean
      else
        valid = pf_->Resample(beta_[l], num_anneal_level == 1? true : false, true); // and calculate particle mean
      

      if(valid) // && th_neff_ratio_*static_cast<float>(pf_->GetNumOfParticle()) < pf_->GetNeff())
      {
        mutex_.lock();
        cvCopy(pf_->GetMeanState(), pose_);
        mutex_.unlock();
      }
      else
      {
        // particle filter results is not valid
        init_ = true; // (re-)init again
        break; // stop annealing process
      }
    }

    // reset 'previous state of particles' to 'current state of particles' right after annealing
    if(num_anneal_level > 1 && !init_)
    {
      for(int p=0; p<pf_->GetNumOfParticle(); p++)
        cvCopy(pf_->GetState(p), pf_->GetPrevState(p));
    }
    obj_model_->getVisibleArea(height_, width_);
    return 0;
  }
};
