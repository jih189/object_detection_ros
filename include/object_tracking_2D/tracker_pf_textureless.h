#pragma once

#include "TooN/Cholesky.h"
#include "TooN/TooN.h"  //
#include "TooN/se3.h"   // for special Euclidean group
#include "TooN/so3.h"   // for special orthogonal group
#include "TooN/wls.h"   // for weighted least square

#include <omp.h>
#include <vector>
#include "tracker_pf.h"

class TexturelessParticleFilterTracker : public ParticleFilterTracker {
 public:
  TexturelessParticleFilterTracker() : pe_surf_(NULL), th_cm_(0.2f) {}

  virtual ~TexturelessParticleFilterTracker() {}

  virtual void clean() {
    TrackerBase::clean();
    if (pe_surf_) delete pe_surf_;
  }

  CvPoint2D32f project3Dto2D(CvPoint3D32f pt3, CvMat *pose,
                             CvMat *param_intrinsic) {
    CvPoint2D32f pt2;
    CvPoint3D32f pt3_cam;
    pt3_cam.x = CV_MAT_ELEM(*pose, float, 0, 0) * pt3.x +
                CV_MAT_ELEM(*pose, float, 0, 1) * pt3.y +
                CV_MAT_ELEM(*pose, float, 0, 2) * pt3.z +
                CV_MAT_ELEM(*pose, float, 0, 3);
    pt3_cam.y = CV_MAT_ELEM(*pose, float, 1, 0) * pt3.x +
                CV_MAT_ELEM(*pose, float, 1, 1) * pt3.y +
                CV_MAT_ELEM(*pose, float, 1, 2) * pt3.z +
                CV_MAT_ELEM(*pose, float, 1, 3);
    pt3_cam.z = CV_MAT_ELEM(*pose, float, 2, 0) * pt3.x +
                CV_MAT_ELEM(*pose, float, 2, 1) * pt3.y +
                CV_MAT_ELEM(*pose, float, 2, 2) * pt3.z +
                CV_MAT_ELEM(*pose, float, 2, 3);

    float fx = CV_MAT_ELEM(*param_intrinsic, float, 0, 0);
    float fy = CV_MAT_ELEM(*param_intrinsic, float, 1, 1);
    float ux = CV_MAT_ELEM(*param_intrinsic, float, 0, 2);
    float uy = CV_MAT_ELEM(*param_intrinsic, float, 1, 2);

    // jiaming hu: adding extrinsic matrix
    float tx = 0.0;  //-148.43597135826298;
    float ty = 0.0;

    pt2.x = fx * pt3_cam.x / pt3_cam.z + ux + tx / pt3_cam.z;
    pt2.y = fy * pt3_cam.y / pt3_cam.z + uy + ty / pt3_cam.z;

    float th_d = 10000.0;
    if (pt2.x < -th_d || pt2.x > th_d || pt2.y < -th_d || pt2.y > th_d) {
      pt2.x = -1;
      pt2.y = -1;
    }

    return pt2;
  }

  virtual bool initTracker(std::string &template_directory,
                           std::string &cam_name, std::string &intrinsic,
                           std::string &distortion, int width, int height,
                           CvMat *pose_init, std::string ach_channel) {
    size_t i = template_directory.rfind('/', template_directory.length());
    // the file name should be the object name
    std::string object_name =
        template_directory.substr(i + 1, template_directory.length() - i);

    TrackerBase::initTracker(template_directory, cam_name, intrinsic,
                             distortion, width, height, pose_init, ach_channel);
    initPoseEstimationSURF(width, height, template_directory, object_name);
    // pf_->Init(pose_init);

    return (true);
  }

  void generate_tracker(TrackerBase *cloned_tracker, CvMat *pose_init) {
    TrackerBase::generate_tracker(cloned_tracker, pose_init);
    pe_surf_ = ((TexturelessParticleFilterTracker *)cloned_tracker)->pe_surf_;
    // pf_->Init(pose_init);
  }

  inline void setThresholdCM(float th) { th_cm_ = th; };
  inline float getThresholdCM() { return th_cm_; };

  void rotate(CvMat *curr_pose, float x, float y, float z) {
    Mat R_x = (Mat_<float>(4, 4) << 1, 0, 0, 0, 0, cos(x), -sin(x), 0, 0,
               sin(x), cos(x), 0, 0, 0, 0, 1);

    // Calculate rotation about y axis
    Mat R_y = (Mat_<float>(4, 4) << cos(y), 0, sin(y), 0, 0, 1, 0, 0, -sin(y),
               0, cos(y), 0, 0, 0, 0, 1);

    // Calculate rotation about z axis
    Mat R_z = (Mat_<float>(4, 4) << cos(z), -sin(z), 0, 0, sin(z), cos(z), 0, 0,
               0, 0, 1, 0, 0, 0, 0, 1);
    cv::Mat temp = R_z * R_y * R_x;
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        CV_MAT_ELEM(*curr_pose, float, r, c) = temp.at<float>(r, c);
      }
    }
  }

  int pose_detection(int numOfDetections, std::vector<LMDetWind> detWind,
                     std::vector<CvMat *> &detectedStates) {
    if (obj_model_->getNumOfEdgeTemplates() > 0) {
      Timer timer;
      timer.start();

      epnp ePnP;
      ePnP.set_internal_parameters(
          CV_MAT_ELEM(*(cam_->getIntrinsicParams()), float, 0, 2),
          CV_MAT_ELEM(*(cam_->getIntrinsicParams()), float, 1, 2),
          CV_MAT_ELEM(*(cam_->getIntrinsicParams()), float, 0, 0),
          CV_MAT_ELEM(*(cam_->getIntrinsicParams()), float, 1, 1));

      pe_surf_->setImage(img_gray_);

      std::vector<CvMat *> resultPose;

      // estimate a set of poses based on random corresondences
      int numof_det = pe_surf_->PF_estimatePosesFDCM(
          th_cm_, numOfDetections, resultPose, detWind, smooth_size_,
          th_canny_l_, th_canny_h_, img_mask_);

      if (numof_det <= 0) {
        return numof_det;
      }

      // for (int d = 0; d < numof_det; d++)
      // {
      //   IplImage *imageTemp = cvCreateImage(cvSize(img_gray_->width,
      //   img_gray_->height), 8, 3);
      //   cvCvtColor(img_gray_, imageTemp, CV_GRAY2BGR);
      //   obj_model_->displayPoseLine(imageTemp, resultPose[d], CV_RGB(255, 0,
      //   0), 1, false);
      //   cv::imshow("after FDCM" + std::to_string(d),
      //   cv::cvarrToMat(imageTemp));
      //   cvReleaseImage(&imageTemp);
      // }

      img_edge_ = obj_model_->extractEdge(img_gray_, smooth_size_, th_canny_l_,
                                          th_canny_h_, NULL, img_mask_);
      cv::imshow("edge", cv::cvarrToMat(img_edge_));
      obj_model_->extractEdgeOri(img_gray_, smooth_size_);

      // cv::Mat dt;
      // cv::Mat invertImage(img_edge_->height, img_edge_->width, CV_8UC1);

      // cv::bitwise_not(cv::cvarrToMat(img_edge_), invertImage);
      // cv::distanceTransform(invertImage, dt, CV_DIST_L2, 3);

      // timer.printTimeMilliSec("preprocess");
      // IplImage *imageTemp =
      //     cvCreateImage(cvSize(img_gray_->width, img_gray_->height), 8, 3);
      // cvCvtColor(img_gray_, imageTemp, CV_GRAY2BGR);

      for (int d = 0; d < numof_det; d++) {
        // obj_model_->displayPoseLine(img_mask_, resultPose[d], cvScalar(0), 1,
        // false);

        pf_->Init(resultPose[d]);
        pf_->Propagate(noise_l_, noise_h_, true);

        // Timer dtimer;
        // dtimer.start();
        // for (int p = 0; p < pf_->GetNumOfParticle(); p++)
        for (int p = 0; p < 1; p++) {
          // printPose(pf_->GetPropState(p));
          obj_model_->setModelviewMatrix(pf_->GetPropState(p));
          obj_model_->findVisibleSamplePoints();
          // obj_model_->keepOnlyContourPoints();
          // cvCvtColor(img_gray_, imageTemp, CV_GRAY2BGR);
          // obj_model_->displayPoseLine(imageTemp, pf_->GetPropState(p),
          // CV_RGB(255, 0, 0), 1, false);

          // Timer ptimer;
          // ptimer.start();

          // obj_model_->refindMatching(dt, ePnP);

          // pf_->setPropagate(p, obj_model_->getPose());
          // obj_model_->findVisibleSamplePoints();
          // obj_model_->keepOnlyContourPoints();

          obj_model_->findNormalUsingEdgeCoordFineOri();
          obj_model_->findEdgeCorrespondencesFineOri();
          // obj_model_->drawPointsAndErrorFineOri(imageTemp);
          // cv::imshow("error" + std::to_string(p), cv::cvarrToMat(imageTemp));

          // if (th_ransac_ > 0.0f)
          // {
          //  obj_model_->refineEdgeCorrespondences_RANSAC(pf_->GetPropState(p),
          //  th_ransac_iter_, th_ransac_);
          // }

          CvMat *J = NULL, *e = NULL;

          edge_tracker_->PF_getJacobianAndError(
              obj_model_->getPose(), obj_model_->getVisibleSamplePoints(), &J,
              &e);

          pf_->Update_IRLS(p, J, e,
                           obj_model_->getNumberOfVisibleSamplePoints());

          // calculate weights
          pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(),
                                maxd_, lamda_e_, lamda_v_);
          // if (e != NULL)
          // {
          //   obj_model_->displayPoseLine(imageTemp, pf_->GetOptState(p),
          //   CV_RGB(0, 255, 0), 1, false);
          //   cv::imshow("particle" + std::to_string(p),
          //   cv::cvarrToMat(imageTemp));
          //   obj_model_->drawPointsAndErrorFineOri(imageTemp);
          //   cv::imshow("error" + std::to_string(p) + " with " +
          //   std::to_string(d), cv::cvarrToMat(imageTemp));
          // }

          // release after use them
          if (J) cvReleaseMat(&J);
          if (e) cvReleaseMat(&e);

          if (pf_->GetNumOfParticle() > 1) {
            e = NULL;

            if (CV_MAT_ELEM(*(pf_->GetOptState(p)), float, 2, 3) > 0.0 &&
                CV_MAT_ELEM(*(pf_->GetOptState(p)), float, 2, 3) < 2.0) {
              // update the initial pose to object model for displaying
              obj_model_->setModelviewMatrix(pf_->GetOptState(p));
              // draw object model with visibility test
              obj_model_->findVisibleSamplePoints();
              // find normal of each sampling point
              obj_model_->findNormalUsingEdgeCoord();
              // calculate error between sampling points and nearest edge
              obj_model_->findEdgeCorrespondences();
              edge_tracker_->PF_getError(pf_->GetOptState(p),
                                         obj_model_->getVisibleSamplePoints(),
                                         &e);
            }
            // Calculate weights for optimized particles
            pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(),
                                  maxd_, lamda_e_, lamda_v_, true);
            if (e) cvReleaseMat(&e);
          }
        }

        if (pf_->GetNumOfParticle() > 1) pf_->CorrectWeights();

        // resampling
        bool valid;
        if (pf_->GetNumOfParticle() > 1)
          valid = pf_->ResampleOpt(beta_[0], true,
                                   true);  // and calculate particle mean
        else
          valid = pf_->Resample(beta_[0], true,
                                true);  // and calculate particle mean

        if (valid)  // &&
        // th_neff_ratio_*static_cast<float>(pf_->GetNumOfParticle())
        // < pf_->GetNeff())
        {
          mutex_.lock();
          cvCopy(pf_->GetMeanState(), resultPose[d]);
          mutex_.unlock();
        }

        detectedStates.push_back(resultPose[d]);
        // dtimer.printTimeMilliSec("each detection");
      }

      double maxmatchrate = 0.0;
      int mind = 0;

      timer.printTimeMilliSec("object detection");
      IplImage *imageTemp = cvCloneImage(img_result_);
      for (int d = 0; d < numof_det; d++) {
        CvMat *e = NULL;

        obj_model_->setModelviewMatrix(resultPose[d]);
        obj_model_->findVisibleSamplePoints();
        obj_model_->findNormalUsingEdgeCoordFineOri();
        obj_model_->findEdgeCorrespondencesFineOri();
        // obj_model_->drawPointsAndErrorFineOri(imageTemp);
        edge_tracker_->PF_getError(resultPose[d],
                                   obj_model_->getVisibleSamplePoints(), &e);
        // double currentcost = 0.0;
        double matchpoint = 0;
        if (e) {
          for (int ei = 0; ei < e->rows; ei++) {
            if (CV_MAT_ELEM(*e, float, ei, 0) <= 2) {
              matchpoint++;
            }
            // currentcost += CV_MAT_ELEM(*e, float, ei, 0);
          }
          std::cout << d << "match rate = " << matchpoint / e->rows << std::endl;
          if (matchpoint / e->rows > maxmatchrate) {
            maxmatchrate = matchpoint / e->rows;
            mind = d;
          }
        }
        // double currcost = obj_model_->getCMCost(dt);

        // cv::putText(cv::cvarrToMat(imageTemp),
        //             "cost: " + std::to_string(currcost), cvPoint(30, 30),
        //             FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 0, 0), 1,
        //             CV_AA);

        // imshow("result" + std::to_string(d), cv::cvarrToMat(imageTemp));
        // std::cout << "result " << std::to_string(d)
        //           << " with result = " << currcost << std::endl;

        if (e) cvReleaseMat(&e);
      }

      if (mind != 0) {
        CvMat *temp = resultPose[mind];
        resultPose[mind] = resultPose[0];
        resultPose[0] = temp;
      }
      obj_model_->setModelviewMatrix(resultPose[0]);
      obj_model_->findVisibleSamplePoints();
      obj_model_->findNormalUsingEdgeCoordFineOri();
      obj_model_->findEdgeCorrespondencesFineOri();
      obj_model_->displayPoseLine(imageTemp, resultPose[0], CV_RGB(255, 0, 0), 1, false);
      imshow("result", cv::cvarrToMat(imageTemp));

      cv::waitKey(0);
      cv::destroyAllWindows();

      cvReleaseImage(&imageTemp);

      if (detectedStates.size() > 0) {
        init_ = false;
        return detectedStates.size();
      } else {
        init_ = true;
        return 0;
      }
    } else {
      return -1;
    }
  }

  void resetImage() { cvCvtColor(img_gray_, img_result_, CV_GRAY2BGR); }

 protected:
  CPoseEstimationSURF *pe_surf_;
  float th_cm_;

  virtual bool initObjectModel(std::string name, int width, int height,
                               CvMat *intrinsic, float sample_step, int maxd,
                               bool dulledge, CEdgeTracker *edge_tracker) {
    TrackerBase::initObjectModel(name, width, height, intrinsic, sample_step,
                                 maxd, dulledge, edge_tracker);

    return (true);
  }

  bool initPoseEstimationSURF(int width, int height, std::string data_name,
                              std::string &obj_name) {
    if (pe_surf_) delete pe_surf_;

    pe_surf_ = new CPoseEstimationSURF(width, height, data_name, obj_model_,
                                       cam_->getIntrinsicParams(),
                                       cam_->getDistortionParams(), obj_name);

    return (true);
  }

  virtual void displayResults() {
    ParticleFilterTracker::displayResults();

    // cvSaveImage("/home/jiaming/catkin_ws/pf_image.png",
    // pe_surf_->getImage());
    /*
    if(display_init_result_)
    {
      cvShowImage("Initialization", pe_surf_->getImage());
      cvWaitKey(1500);
      display_init_result_ = false;
    }
    */
  }

  virtual bool initialize() {
    TrackerBase::initialize();
    display_init_result_ = true;
    init_ = false;
    return false;
  }

  virtual void handleKey(char key) {
    switch (key) {
      case 'r':
      case 'R':
        if (!init_) init_ = true;
        break;
      case 't':
      case 'T':
        if (init_) init_ = false;
        break;
      case 27:
        run_ = false;
        break;
    }
  }

  virtual int tracking() {
    // do annealing process only after (re-)initialization
    int num_anneal_level =
        frame_num_after_init_ == 0 ? num_annealing_layers_ : 1;

    for (int l = num_anneal_level - 1; l >= 0; l--) {
      // 'getEdge' returns Berkeley edge if it is available, otherwise returns
      // NULL
      // 'extractEdge' extracts Canny edge if the fourth edge is NULL
      img_edge_ =
          obj_model_->extractEdge(img_gray_, smooth_size_, th_canny_l_,
                                  th_canny_h_, cam_->getEdge(), img_mask_);
      obj_model_->extractEdgeOri(img_gray_, smooth_size_);

      if (num_anneal_level == 1)
        pf_->Propagate(noise_l_, noise_h_, true);
      else
        pf_->Propagate(alpha_[l], alpha_[l],
                       l == num_anneal_level - 1 ? true : false);

      for (int p = 0; p < pf_->GetNumOfParticle(); p++) {
        // update the initial pose to object model for displaying
        obj_model_->setModelviewMatrix(pf_->GetPropState(p));
        // draw object model with visibility test
        obj_model_->findVisibleSamplePoints();
        obj_model_->keepOnlyContourPoints();
        // find normal of each sampling point
        obj_model_->findNormalUsingEdgeCoord();
        // calculate error between sampling points and nearest edge
        obj_model_->findEdgeCorrespondences();

        /*
        if (th_ransac_ > 0.0f)
        {
          obj_model_->refineEdgeCorrespondences_RANSAC(pf_->GetPropState(p),
        th_ransac_iter_, th_ransac_);
        }
*/
        // consider edge sample points only
        CvMat *J = NULL, *e = NULL;

        edge_tracker_->PF_getJacobianAndError(
            pf_->GetPropState(p), obj_model_->getVisibleSamplePoints(), &J, &e);

        pf_->Update_IRLS(
            p, J, e,
            obj_model_->getNumberOfVisibleSamplePoints());  // update OptState

        // calculate weights
        pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(), maxd_,
                              lamda_e_, lamda_v_);
        // release after use them
        if (J) cvReleaseMat(&J);
        if (e) cvReleaseMat(&e);

        // calculate weight (likelihood now) for optimized particles
        if (pf_->GetNumOfParticle() > 1) {
          e = NULL;
          // check whether this opt pose state is valid
          if (CV_MAT_ELEM(*(pf_->GetOptState(p)), float, 2, 3) > 0.001 &&
              CV_MAT_ELEM(*(pf_->GetOptState(p)), float, 2, 3) < 1.0) {
            // update the initial pose to object model for displaying
            obj_model_->setModelviewMatrix(pf_->GetOptState(p));
            // draw object model with visibility test
            obj_model_->findVisibleSamplePoints();
            obj_model_->keepOnlyContourPoints();
            // find normal of each sampling point
            obj_model_->findNormalUsingEdgeCoord();
            // calculate error between sampling points and nearest edge
            obj_model_->findEdgeCorrespondences();
            /*
            if (th_ransac_ > 0.0f)
              obj_model_->refineEdgeCorrespondences_RANSAC(pf_->GetOptState(p),
            th_ransac_iter_, th_ransac_);
              */

            edge_tracker_->PF_getError(
                pf_->GetOptState(p), obj_model_->getVisibleSamplePoints(), &e);
          }

          // Calculate weights for optimized particles
          pf_->calculateWeights(p, e, obj_model_->getVisibleSamplePoints(),
                                maxd_, lamda_e_, lamda_v_, true);
          if (e) cvReleaseMat(&e);
        }
      }

      // correct weights caused from considering optimized states
      if (pf_->GetNumOfParticle() > 1) pf_->CorrectWeights();

      // resampling
      bool valid;
      if (pf_->GetNumOfParticle() > 1)
        valid = pf_->ResampleOpt(beta_[l], num_anneal_level == 1 ? true : false,
                                 true);  // and calculate particle mean
      else
        valid = pf_->Resample(beta_[l], num_anneal_level == 1 ? true : false,
                              true);  // and calculate particle mean

      if (valid)  // &&
                  // th_neff_ratio_*static_cast<float>(pf_->GetNumOfParticle())
                  // < pf_->GetNeff())
      {
        mutex_.lock();
        cvCopy(pf_->GetMeanState(), pose_);
        mutex_.unlock();

        if (CV_MAT_ELEM(*pose_, float, 2, 3) <= 0.0) {
          init_ = true;
          return 1;
        }

        obj_model_->setModelviewMatrix(pose_);
        obj_model_->findVisibleSamplePoints();

        obj_model_->keepOnlyContourPoints();
        // find normal of each sampling point
        obj_model_->findNormalUsingEdgeCoord();
        // calculate error between sampling points and nearest edge
        obj_model_->findEdgeCorrespondences();
        float match_threshold = 2.0 / CV_MAT_ELEM(*pose_, float, 2, 3);
        if (match_threshold > maxd_) match_threshold = maxd_;
        int numofInner = 0;
        for (int vp = 0; vp < obj_model_->getVisibleSamplePoints().size();
             vp++) {
          if ((obj_model_->getVisibleSamplePoints())[vp].dist <
              match_threshold) {
            numofInner++;
          }
        }
        float currentMatchrate =
            (float)numofInner / obj_model_->getVisibleSamplePoints().size();
        // std::cout << "matching rate: " << currentMatchrate << std::endl;
        // std::cout << " number of visible points: " <<
        // obj_model_->getVisibleSamplePoints().size() << std::endl;
        if (obj_model_->getVisibleSamplePoints().size() > 10 &&
            currentMatchrate < 0.37) {
          init_ = true;
          return 1;
        }
      } else {
        // particle filter results is not valid
        init_ = true;  // (re-)init again
        return 1;      // stop annealing process
      }
    }

    // reset 'previous state of particles' to 'current state of particles' right
    // after annealing
    if (num_anneal_level > 1 && !init_) {
      for (int p = 0; p < pf_->GetNumOfParticle(); p++)
        cvCopy(pf_->GetState(p), pf_->GetPrevState(p));
    }

    return 0;
  }
};
