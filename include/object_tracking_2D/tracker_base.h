#pragma once

#include <boost/signals2/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <omp.h>        // openmp
#include <iostream>
//#include <direct.h> // mkdir()

#include "Camera.h"
#include "PoseEstimationSURF.h"
#include "ObjectModel.h"
#include "EdgeTracker.h"
#include "Timer.h"

// tensorflow headers
#include <utility>
#include <vector>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "utils.h"

//#include <imageReceiver.h>
//#include "/home/acosgun/repos/ach/include/ach.h"
//#include "sns.h"
//#include <ach.h>
//#include <sns.h>

using namespace cv;
using boost::asio::ip::tcp;

// tensorflow using
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

class TrackerBase
{
public:
  TrackerBase()
    : cam_(NULL)
    , edge_tracker_(NULL)
    , obj_model_(NULL)
    , th_canny_l_(100) //100
    , th_canny_h_(200) //120t
    , sample_step_(0.005f)
    , maxd_(32)
    , dulledge_(false)
    , ransac_th_(0.0f)
    , limityrot_(false)
    , run_(true)
    , init_(true)
    , th_valid_sample_points_ratio_(0.1)
    , img_input_(NULL)
    , img_gray_(NULL)
    , img_gray_tracking(NULL)
    , img_result_(NULL)
    , img_edge_(NULL)
    , img_mask_(NULL)
    , display_(true)    , display_result_(true)
    , display_init_result_(false)
    , display_edge_result_(true)
    , display_grayscale_image_(false)
    , smooth_size_(1)
    , obj_name_("")
    , frame_num_(0)
    , frame_num_after_init_(0)
    , hsvFilt(NULL)
    , net_(false)
    , save_rslt_txt_(false)
    , save_rslt_img_(false)
    , str_result_path_("result")
    , time_tracking_(0.f)
    , time_init_(0.f)
    , time_run_(0.f)
    , saveKeyframe_(false)
  {
    pose_ = cvCreateMat(4, 4, CV_32F);
    pose_init_ = cvCreateMat(4, 4, CV_32F);
    cvSetIdentity(pose_);
    covariance_ = cvCreateMat(6, 6, CV_32F);
    cvSetIdentity(covariance_);
    displayOpenCVInfo();
  }
  
  virtual ~TrackerBase()
  {
    cvReleaseMat(&pose_);
    cvReleaseMat(&pose_init_);
    cvReleaseMat(&covariance_);
  }

  virtual void clean(){
    //if(cam_)            delete cam_;
    if(edge_tracker_)   delete edge_tracker_;
    if(obj_model_)      delete obj_model_;
    // 'img_input_' is aleady released
    if(img_gray_)       cvReleaseImage(&img_gray_);
    if(img_gray_tracking)       cvReleaseImage(&img_gray_tracking);
    if(img_result_)     cvReleaseImage(&img_result_);
    if(img_edge_)       cvReleaseImage(&img_edge_);
    if(img_mask_)       cvReleaseImage(&img_mask_);

    //cvReleaseMat(&pose_);
    //cvReleaseMat(&pose_init_);
    //cvReleaseMat(&covariance_);
    if(ofs_pose_.is_open())      ofs_pose_.close();
    if(ofs_time_.is_open())      ofs_time_.close();
  }

  virtual bool initTracker(std::string &obj_name, std::string &cam_name, std::string &intrinsic, std::string &distortion, int width, int height, CvMat* pose_init, std::string ach_channel)
  {
    if(ach_channel != "none")
			use_ach_ = true;
		else 
			use_ach_ = false;

    //ACH
    if(use_ach_)
      {
	//sns_chan_open( &channel, "obj", NULL );
	//rec.Init(ach_channel.c_str(), width, height);
      }

    obj_name_ = obj_name;
    width_ = width;
    height_ = height;
    initCamera(cam_name, intrinsic, distortion, width, height);
    initEdgeTracker(width, height, cam_->getIntrinsicParams(), maxd_, limityrot_);
/*
    std::cout << "jiaming check Intrinsic\n";
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 3; j++){
        std::cout << CV_MAT_ELEM(*(cam_->getIntrinsicParams()), float, i, j) << " ";
      }
      std::cout << std::endl;
    }
*/  
    initObjectModel(obj_name, width, height, cam_->getIntrinsicParams(), sample_step_, maxd_, dulledge_, edge_tracker_);
    initImages(width, height);

    if(save_rslt_txt_)
    {
      if(!ofs_pose_.is_open())
        ofs_pose_.open((str_result_path_ + std::string("/pose.txt")).c_str());
      if(!ofs_time_.is_open())
        ofs_time_.open((str_result_path_ + std::string("/time.txt")).c_str());
    }

    cvCopy(pose_init, pose_init_);
    cvCopy(pose_init_, pose_);

    // jiaming hu: initialize tensorflow
    // Set dirs variables
    /*
    string ROOTDIR = "/home/jiaming/catkin_ws/";
    string LABELS = "labels_map.pbtxt";
    string GRAPH = "frozen_inference_graph.pb";

    // Set input & output nodes names
    inputLayer = "image_tensor:0";
    outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

    // Load and initialize the model from .pb file
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR - " << loadGraphStatus;./
        return false;
    } else
        LOG(INFO) << "loadGraph(): frozen graph loaded " << endl;


    // Load labels map from .pbtxt file
    labelsMap = std::map<int,std::string>();
    Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
    if (!readLabelsMapStatus.ok()) {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return false;
    } else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;
    thresholdScore = 0.5;
    thresholdIOU = 0.8;


    shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim((int64)height);
    shape.AddDim((int64)width);
    shape.AddDim(3);
    */
    return (true);
  }

  // jiaming hu: generate_tracker the tracker template
  virtual void generate_tracker(TrackerBase* cloned_tracker, CvMat* pose_init){
    use_ach_ = false;
    obj_name_ = cloned_tracker->obj_name_;
    width_ = cloned_tracker->width_;
    height_ = cloned_tracker->height_;

    cam_ = cloned_tracker->cam_;
    edge_tracker_ = cloned_tracker->edge_tracker_;
    obj_model_ = cloned_tracker->obj_model_;
    img_gray_ = cloned_tracker->img_gray_;
    img_result_ = cloned_tracker->img_result_;
    img_edge_ = cloned_tracker->img_edge_;
    img_mask_ = cloned_tracker->img_mask_;

    cvCopy(pose_init, pose_init_);
    cvCopy(pose_init_, pose_);
  }

  void sendPoseACH(CvMat* p)
  {
/*		if(use_ach_)
		{
			printf("Sending Pose Over ACH\n");
    
			struct sns_msg_tf *msg = sns_msg_tf_local_alloc(1);
			double R[9];
			for( size_t col = 0; col < 3; col++ ) 
	  	{
	    	for( size_t row = 0; row < 3; row ++ ) 
	      {
					AA_MATREF(R,3,row,col) = CV_MAT_ELEM(*p,float,row,col);
	      }
	  	}
			aa_tf_rotmat2quat( R, msg->tf[0].r.data );

			for( size_t i = 0; i < 3; i++ ) //translation
	  	{
	    	msg->tf[0].v.data[i] = CV_MAT_ELEM(*p,float,i,3);
	  	}

			enum ach_status r = sns_msg_tf_put( &channel, msg );
			if( ACH_OK != r ) fprintf(stderr, "Could not put message: %s\n", ach_result_to_string(r));

			aa_mem_region_local_pop( msg );
		}*/
    return;
  }

  void printPose(CvMat* p)
  {
    for(int cur_row = 0; cur_row<4; cur_row++)
		{
			printf("[%1.3f %1.3f %1.3f %1.3f]\n", CV_MAT_ELEM(*p, float, cur_row, 0), CV_MAT_ELEM(*p, float, cur_row, 1) ,CV_MAT_ELEM(*p, float, cur_row, 2), CV_MAT_ELEM(*p, float, cur_row, 3));
    }
    return;
  }

  cv::Mat filterOutColor(Mat inputImage, int iLowH, int iLowS, int iLowV, int iHighH, int iHighS, int iHighV)
  {
    Mat imgHSV;
    Mat imgThresholded;
    cvtColor(inputImage, imgHSV, COLOR_BGR2HSV);
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

    return imgThresholded;
  }

  void run()
  {
    if(display_)
    {
      cvNamedWindow("Result");
      cvNamedWindow("Edge");
      cvNamedWindow("Initialization");
    }

    while(run_)
    {
      // Wait if network flag is on
      if(net_)
      {
        handleKey(cvWaitKey(1));
        continue;
      }

      // capture or load an image

   //   std::cout<<"initialization and run "<<init_<<run_<<std::endl;
      if(!getImage(use_ach_))
        break;
      
      //timer_.start();


      
      if(init_) initialize();

      //time_init_ = timer_.printTimeMilliSec("initializing");

      // filter color from image
      if(hsvFilt!=NULL)
      {
        IplImage* tmp = new IplImage(filterOutColor(img_input_, hsvFilt[0], hsvFilt[1], hsvFilt[2], hsvFilt[3], hsvFilt[4], hsvFilt[5]));
        img_gray_tracking = cvCloneImage(tmp);
        //cvCvtColor(img_gray_tracking, img_result_, CV_GRAY2RGB);
      }
      else
      {
        img_gray_tracking = cvCloneImage(img_gray_);
      }

      // do processing
      //timer_.start();

      if(!init_)
	{
	  if(!use_tracking_)
	    {
	      init_ = true; //NO TRACKING if true
	    }
	  tracking();	  
	}

      //time_tracking_ = timer_.printTimeMilliSec("tracking");

      if(display_)
        displayResults();

      if(save_rslt_txt_)
        saveResultText();

      if(save_rslt_img_)
        saveResultImage();

      // release image if it's loaded from file
      if(!cam_->IsCamera()) cvReleaseImage(&img_input_);

      char key = (char)cvWaitKey(1); // get keyboard input
      handleKey(key);

      frame_num_++;
      frame_num_after_init_++;
    }

    if(display_)
    {
      cvDestroyWindow("Result");
      cvDestroyWindow("Edge");
      cvDestroyWindow("Initialization");
    }
  }

  void saveResultText()
  {
    if(ofs_pose_.is_open())
    {
      for(int r = 0; r < 4; r++)
        for(int c = 0; c < 4; c++)
          ofs_pose_ << std::fixed << CV_MAT_ELEM(*pose_, float, r, c) << '\t';
      ofs_pose_ << std::endl; 
    }
    else
      std::cerr << "ofs_pose_ is not opened." << std::endl;

    ofs_time_ << setprecision(8) << time_init_ + time_tracking_ << std::endl;
  }

  void saveResultImage()
  {
    std::stringstream ss;
    ss << str_result_path_ << "/" << "track" << std::setw(4) << std::setfill('0') << frame_num_ << ".jpg";
    cvSaveImage(ss.str().c_str(), img_result_);
  }

  typedef struct _NET_THREAD_PARAM
  {
    CvMat* pose;
    bool* net;
    bool* run;
    bool* init;
    boost::signals2::mutex* mutex;
  }NET_THREAD_PARAM;

  inline int    getWidth()                          { return width_;          }
  inline int    getHeight()                         { return height_;         }
  inline int    getCannyLow()                       { return th_canny_l_;     }
  inline void   setCannyLow(int th)                 { th_canny_l_ = th;       }
  inline int    getCannyHigh()                      { return th_canny_h_;     }
  inline void   setCannyHigh(int th)                { th_canny_h_ = th;       }
  inline void   setValidSamplePointsRatio(double th){th_valid_sample_points_ratio_ = th;}
  inline double getValidSamplePointsRatio()         {return th_valid_sample_points_ratio_;}
  inline float  getSampleStep()                     { return sample_step_;    }
  inline void   setSampleStep(float ss)             { sample_step_ = ss;      }
  inline int    getMaxSearchDistance()              { return maxd_;           }
  inline void   setMaxSearchDistance(int d)         { maxd_ = d;              }
  inline bool   getConsideringDullEdges()           { return dulledge_;       }
  inline void   setConsideringDullEdges(bool tf)    { dulledge_ = tf;         }
  inline bool   getDisplay()                        { return display_;        }
  inline void   setDisplay(bool tf)                 { display_ = tf;          }
  inline bool   getNetworkMode()                    { return net_;            }
  inline void   setNetworkMode(bool tf)             { net_ = tf;              } 
  inline bool   getSaveResultText()                 { return save_rslt_txt_;  }
  inline void   setSaveResultText(bool tf)          { save_rslt_txt_ = tf;    }
  inline bool   getSaveResultImage()                { return save_rslt_img_;  }
  inline void   setSaveResultImage(bool tf)         { save_rslt_img_ = tf;    }
  inline CvMat*   getPose()                         { return pose_;           }
  inline CvMat*   getCovariance()                   { return covariance_;                        }
  inline void   setPose(CvMat* pose)                { pose_ = pose;           }
  inline IplImage*   getResultImage()               { return img_result_;     }
  inline IplImage*   getEdgeImage()                 { return img_edge_;       }
  inline IplImage*   getMaskImage()                 { return img_mask_;       }
  inline void   setMinKeypointMatches(int d)        { min_keypoint_matches = d; }
  inline void   setTracking(bool use_tracking)      { use_tracking_ = use_tracking; }
  inline std::string& getSaveResultPath()           { return str_result_path_; }
  inline void   saveKeyframe()                      { saveKeyframe_ = true;}
  bool init_;
  bool label_;
  bool setSaveResultPath(std::string& path)
  { 
    if(mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST)
    {
      std::cerr << "Cannot create " << path << " directory for saving results." << std::endl;
      return false;
    }
    str_result_path_ = path;
    return true;
  }
  virtual int tracking() = 0;

  void setColorFilter(int* thresholds)
  {
    hsvFilt = thresholds;
  }

  bool imageFilter(cv::Mat image){
      // jiaming: tensorflow processing
      // clean edge image
      //cvRectangle(img_edge_, cvPoint(0,0), cvPoint(image.cols,image.rows), CV_RGB(0,0,0), -1);
      cvRectangle(img_mask_, cvPoint(0,0), cvPoint(image.cols,image.rows), CV_RGB(0,0,0), -1);

      tensor = Tensor(tensorflow::DT_FLOAT, shape);
      try{
          Status readTensorStatus = readTensorFromMat(image, tensor);
          if(!readTensorStatus.ok()) {
              LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
              return false;
          }
      } catch(exception& e){
          std::cout << e.what() << std::endl;
      }

      // Run the graph on tensor
      outputs.clear();
      Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
      if (!runStatus.ok()) {
          LOG(ERROR) << "Running model failed: " << runStatus;
          return false;
      }

      // Extract results from the outputs vector
      tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
      tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
      tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
      tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();
      vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);

      for (size_t i = 0; i < goodIdxs.size(); i++){
          CvPoint tl = cvPoint((int) (boxes(0, goodIdxs.at(i), 1) * image.cols), (int) (boxes(0, goodIdxs.at(i), 0) * image.rows));
          CvPoint br = cvPoint((int) (boxes(0, goodIdxs.at(i), 3) * image.cols), (int) (boxes(0, goodIdxs.at(i), 2) * image.rows));
          cvRectangle(img_mask_, tl, br, CV_RGB(255,255,255), -1);
      }

      return true;
  }

  bool setImage(cv::Mat image)
  {
      //imageFilter(image);
      cvSetZero(img_mask_);
      IplImage copy = image;
      img_input_ = static_cast<IplImage *>(&copy);

      cvCvtColor(img_input_, img_gray_, CV_RGB2GRAY);
      cvCvtColor(img_gray_, img_result_, CV_GRAY2RGB);
      return true;
  }

  virtual void renderResults()
  {
    obj_model_->displayPoseLine(img_result_, pose_, CV_RGB(255, 0, 0), 1, false);
    obj_model_->displaySamplePointsAndErrors(img_edge_);
  }

  virtual void displayResults()
  {
    renderResults();
    cvShowImage("Result", img_result_);
    cvShowImage("Edge", img_edge_);
  }

  virtual bool initialize()
  {
    if(!init_)
    {
      std::cerr << "Initialization flag is not set." << std::endl;
    }

    frame_num_after_init_ = 0;
    return false;
  }



protected:
  void displayOpenCVInfo()
  {
    /*char* libraries;
    char* modules;
    //cvGetModuleInfo(0, (const char**)&libraries, (const char**)&modules);

    std::cout << "Libraries: " << libraries << std::endl;
    std::cout << "Modules: " << modules << std::endl;*/
  }

  bool initCamera(std::string &cam_name, std::string &intrinsic, std::string &distortion, int width, int height)
  {
    if(cam_) delete cam_;
    if(cam_name.compare("normal") == 0 || cam_name.compare("fire-i") == 0 || cam_name.compare("flea") == 0 || cam_name.compare("openni") == 0 || cam_name.compare("ach") == 0)
      cam_ = new CCamera(cam_name, intrinsic, distortion, width, height);
    else{ // assume 'cam_name' is a path to an image sequence
      std::string jpg = "jpg";
      cam_ = new CCamera(cam_name, false, 0, intrinsic, distortion, jpg);
    }
    return (true);
  }

  bool initEdgeTracker(int width, int height, CvMat* intrinsic, int maxd, bool limityrot)
  {
    if(edge_tracker_) delete edge_tracker_;
    edge_tracker_ = new CEdgeTracker(width, height, intrinsic, maxd, limityrot);
    return (true);
  }

  virtual bool initObjectModel(std::string name, int width, int height, CvMat* intrinsic, float sample_step, int maxd, bool dulledge, CEdgeTracker* edge_tracker)
  {
    if(obj_model_) delete obj_model_;
    obj_model_ = new CObjectModel(name, width, height, intrinsic, sample_step, maxd, dulledge, edge_tracker);
    obj_model_->loadObjectCADModel(name);
    return (true);
  }
  
  bool initImages(int width, int height)
  {
    if(img_gray_) cvReleaseImage(&img_gray_);
    img_gray_ = cvCreateImage(cvSize(width, height), 8, 1);
    if(img_result_) cvReleaseImage(&img_result_);
    img_result_ = cvCreateImage(cvSize(width, height), 8, 3);
    if(img_edge_) cvReleaseImage(&img_edge_);
    img_edge_ = cvCreateImage(cvSize(width, height), 8, 3);
    if(img_mask_) cvReleaseImage(&img_mask_);
    img_mask_ = cvCreateImage(cvSize(width, height), 8, 1);

    return (true);
  }


  bool getImage(bool ach)
  {
    if(!ach)
    {
      img_input_ = cam_->getImage();
      if(img_input_ == NULL) return false;
      if(img_input_->nChannels == 1)
      {
        cvCopy(img_input_, img_gray_);
        cvCvtColor(img_gray_, img_result_, CV_GRAY2RGB);
      }
      else
      {
        cvCvtColor(img_input_, img_gray_, CV_RGB2GRAY);
        //cvCopy(img_input_, img_result_);
        cvCvtColor(img_gray_, img_result_, CV_GRAY2RGB);
      }
    }
    else
    {
			// receive image over ach
			/*image = rec.receiveImage();

			// HACK: check this part!
			Mat image_scaled;
			resize(image, image_scaled, Size(width_, height_),0,0,INTER_CUBIC);

			IplImage copy = image_scaled;
			img_input_ = static_cast<IplImage *>(&copy);

			cvCvtColor(img_input_, img_gray_, CV_RGB2GRAY);
			cvCvtColor(img_gray_, img_result_, CV_GRAY2RGB);*/
    }

    return true;
  }

  virtual void handleKey(char key) = 0;



    std::string handleRequest(std::string& req)
  {
    if(req.substr(0, 4).compare("1004") == 0) // request current pose (SE(3) in meter)
    {
      net_ = false;
      mutex_.lock();
      std::stringstream ss;
      ss << "1 "; // means success
      for(int i=0; i<16; i++)
        ss << std::fixed << std::setprecision(6) << CV_MAT_ELEM(*pose_, float, i/4, i%4) << " ";
      ss << std::endl;
      mutex_.unlock();
      return ss.str();
    }
    else if(req.substr(0, 4).compare("1005") == 0) // start without initialization
    {
      net_ = false;
      init_ = false;
      return std::string("1\n");
    }
    else if(req.substr(0, 4).compare("2000") == 0) // request re-initialization
    {
      init_ = true;
      return std::string("1\n");
    }
    else if(req.substr(0, 4).compare("3000") == 0) // request to end tracker
    {
      run_ = false;
      return std::string("1\n");
    }

    return std::string("");
  }

  void networkFunc()
  {
    std::cout << "Starting up a network server" << std::endl;

    try
    {
      boost::asio::io_service io_service;

      tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 1701));

      tcp::socket socket(io_service);
      acceptor.accept(socket);

      for (;;)
      {
        boost::array<char, 128> buf;
        boost::system::error_code error;
        size_t len = socket.read_some(boost::asio::buffer(buf), error);
        if (error == boost::asio::error::eof)
          break; // Connection closed cleanly by peer.
        else if (error)
          throw boost::system::system_error(error); // Some other error.

        std::stringstream ss;
        ss.write(buf.data(), len);
				std::string tmp = ss.str(); 
        std::string msg = handleRequest(tmp);

        boost::system::error_code ignored_error;
        boost::asio::write(socket, boost::asio::buffer(msg), boost::asio::transfer_all(), ignored_error);
      }
    }
    catch (std::exception& e)
    {
      std::cerr << e.what() << std::endl;
    }

    return;
  }

  CCamera* cam_;
  CEdgeTracker* edge_tracker_;
  CObjectModel* obj_model_;

  int width_;
  int height_;
  bool run_;
  bool display_init_result_;
  double th_valid_sample_points_ratio_;
  std::string obj_name_;
  int frame_num_;
  int frame_num_after_init_;
  bool net_;
  bool save_rslt_txt_;  // save estimated pose and time result in txt file
  bool save_rslt_img_;  // save result image as jpg file
  std::ofstream ofs_pose_;
  std::ofstream ofs_time_;
  std::string str_result_path_;
  
  int th_canny_l_;
  int th_canny_h_;
  float sample_step_;
  int maxd_;
  bool dulledge_;
  float ransac_th_;
  bool limityrot_;
  bool display_;
  bool display_result_;
  bool display_edge_result_;
  bool display_grayscale_image_;

  int smooth_size_;

  IplImage* img_input_;
  IplImage* img_gray_;
  IplImage* img_gray_tracking;
  IplImage* img_mask_;
  IplImage* img_result_;
  IplImage* img_edge_;

  CvMat* pose_;
  CvMat* pose_init_;
  CvMat* covariance_;


  boost::signals2::mutex mutex_; // For syncronization between main function and network thread
  Timer timer_;
  double time_tracking_;
  double time_init_;
  double time_run_;

  int min_keypoint_matches;
  //ach_channel_t channel;
  bool use_ach_;
  bool use_tracking_;
  bool saveKeyframe_;
  //ImageReceiver rec;
  cv::Mat image;

  int* hsvFilt;

  // tensorflow variables
  std::map<int, std::string> labelsMap;
  Tensor tensor;
  std::vector<Tensor> outputs;
  double thresholdScore;
  double thresholdIOU;
  tensorflow::TensorShape shape;
  std::unique_ptr<tensorflow::Session> session;
  string inputLayer;
  vector<string> outputLayer;
};
