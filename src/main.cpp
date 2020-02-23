#include <iostream>
#include <string>

#include "object_tracking_2D/tracker_irls.h"
#include "object_tracking_2D/tracker_pf_texture.h"
#include "object_tracking_2D/tracker_pf_textureless.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace po = boost::program_options;

int main(int argc, char **argv)
{
  std::string tracker_name;
  std::string template_directory;
  std::string input;
  std::string intrinsic;
  std::string distortion;
  int width;
  int height;
  float sample_step;
  bool dull_edge;
  float th_cm;
  int n; // number of particles
  bool display;
  bool net;
  bool save_rslt_txt;
  bool save_rslt_img;
  std::string str_result_path;
  CvMat *pose_init;
  pose_init = cvCreateMat(4, 4, CV_32F);
  cvSetIdentity(pose_init);
  CV_MAT_ELEM(*pose_init, float, 2, 3) = 0.5f; // 0.5 meter in front of camera
  std::string pose_init_str;
  int min_keypoint_matches;
  std::string ach_channel;
  bool use_tracking;

  po::options_description desc("Tracker options");
  desc.add_options()("help,h", "produce help message")("tracker,t", po::value<std::string>(&tracker_name)->default_value("irls"), "name tracker (irls, pf, pf_textureless)")("obj-name,o", po::value<std::string>(&template_directory), "name of traget object")("input,i", po::value<std::string>(&input), "name of camera (e.g. flea) or image sequence path (e.g. dir/seq1/)")("sample-step,s", po::value<float>(&sample_step)->default_value(0.005f), "sample step")("num-particle,n", po::value<int>(&n)->default_value(1), "number of particles")("save-txt", po::value<bool>(&save_rslt_txt)->default_value(false), "save results in text file")("save-img", po::value<bool>(&save_rslt_img)->default_value(false), "save results in image files")("save-path", po::value<std::string>(&str_result_path), "save results in image files")

      ("display", po::value<bool>(&display)->default_value(true), "display results or not")("network", po::value<bool>(&net)->default_value(false), "use network mode or not")("width", po::value<int>(&width)->default_value(640), "width")("height", po::value<int>(&height)->default_value(480), "height")("intrinsic", po::value<std::string>(&intrinsic)->default_value("Intrinsics_normal.xml"), "intrinsic parameters")("distortion", po::value<std::string>(&distortion)->default_value("Distortion_normal.xml"), "distortion parameters")("dull_edge", po::value<bool>(&dull_edge)->default_value(false), "consider dull edges")("th_cm", po::value<float>(&th_cm)->default_value(0.2f), "threshold of chamfer matching")("init_pose", po::value<std::string>(&pose_init_str), "init pose")
      // AKAN
      ("min_keypoint_matches,m", po::value<int>(&min_keypoint_matches)->default_value(20), "min number of keypoint matches to start tracking")("use_ach_channel", po::value<std::string>(&ach_channel)->default_value("none"), "Use specific ach channel with given name")("use_tracking", po::value<bool>(&use_tracking)->default_value(true), "Enable tracking after detection of object");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (argc < 2 || vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 1;
  }

  if (input == "ach")
  {
    //sns_init();
  }

  if (template_directory.empty())
  {
    std::cerr << "obj-name should be specified." << std::endl;
    return 1;
  }

  if (save_rslt_txt || save_rslt_img)
  {
    if (str_result_path.empty())
    {
      std::cerr << "No save-path specified when using either save-txt or save-img" << std::endl;
      return 1;
    }
  }

  if (vm.count("input"))
  {
    std::cout << "input: " << vm["input"].as<std::string>() << std::endl;
  }

  if (vm.count("obj-name"))
  {
    std::cout << "obj-name: " << vm["obj-name"].as<std::string>() << std::endl;
  }

  if (vm.count("n"))
  {
    std::cout << "n: " << vm["n"].as<int>() << std::endl;
  }

  if (vm.count("init_pose"))
  {
    std::cout << "init_pose: ";
    std::vector<std::string> tokens;
    boost::split(tokens, pose_init_str, boost::is_any_of(","), boost::token_compress_on);

    for (int i = 0; i < tokens.size(); i++)
      std::cout << tokens[i] << " ";
    std::cout << std::endl;

    if (tokens.size() != 16)
      std::cerr << "Not enough number of data in pose. 16 values are required!" << std::endl;
    {
      // oeverwrite pose_init
      for (int i = 0; i < 16; i++)
        pose_init->data.fl[i] = std::atof(tokens[i].c_str());
    }
  }

  /*
  VideoCapture cap(0);

  // Check if camera opened successfully
  if (!cap.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  */

  Mat inputImg = imread(input, CV_LOAD_IMAGE_COLOR);

  TrackerBase *origin_tracker_;
  origin_tracker_ = new TexturelessParticleFilterTracker();
  ((TexturelessParticleFilterTracker *)origin_tracker_)->setNumParticle(n);
  ((TexturelessParticleFilterTracker *)origin_tracker_)->setThresholdCM(th_cm);
  ((TexturelessParticleFilterTracker *)origin_tracker_)->initParticleFilter();

  origin_tracker_->setSampleStep(sample_step);
  origin_tracker_->setMaxSearchDistance(5);
  origin_tracker_->setDisplay(display);
  origin_tracker_->setNetworkMode(false);
  origin_tracker_->setConsideringDullEdges(dull_edge);
  origin_tracker_->setTracking(true);
  origin_tracker_->setValidSamplePointsRatio(0.85);
  origin_tracker_->setMinKeypointMatches(12);

  // set the parameters of edge detection
  origin_tracker_->setCannyHigh(20);
  origin_tracker_->setCannyLow(15);

  input = "ach";
  origin_tracker_->initTracker(template_directory, input, intrinsic, distortion, width, height, pose_init, ach_channel);

  int numofframe = 0;

  bool istrack = false;
  int numOfTracking = 0;
  int lifetime[10];
  int numOfDetections = 0;
  std::vector<TrackerBase *> trackers;

  origin_tracker_->setImage(inputImg);
  std::vector<LMDetWind> detWind;
  std::vector<CvMat *> detectedState;
  Timer timer;
  timer.start();
  numOfDetections = ((TexturelessParticleFilterTracker *)origin_tracker_)->pose_detection(5, detWind, detectedState);
  timer.printTimeMilliSec("object detection");

  /*
  
  while (1)
  {
    Timer timer;
    timer.start();
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
    if (istrack == false)
      fastNlMeansDenoisingColored(frame, frame, 9, 9);
    
    numofframe++;

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    origin_tracker_->setImage(frame);

    if (istrack == false)
    {
      for (int li = 0; li < 10; li++)
        lifetime[li] = 0;

      // clean the trackers
      while (!trackers.empty())
      {
        trackers.pop_back();
      }

      origin_tracker_->init_ = true;
      std::vector<LMDetWind> detWind;
      std::vector<CvMat *> detectedState;

      numOfDetections = ((TexturelessParticleFilterTracker *)origin_tracker_)->pose_detection(5, detWind, detectedState);
      timer.printTimeMilliSec("pose detection");
      if (numOfDetections > 0)
      {
        istrack = false; // todo
        numOfTracking = numOfDetections;

        for (int nt = 0; nt < numOfDetections; nt++)
        {
          trackers.push_back(new TexturelessParticleFilterTracker());
          ((TexturelessParticleFilterTracker *)trackers[nt])->setNumParticle(n);
          ((TexturelessParticleFilterTracker *)trackers[nt])->setThresholdCM(th_cm);
          ((TexturelessParticleFilterTracker *)trackers[nt])->initParticleFilter();
          trackers[nt]->setSampleStep(sample_step);
          trackers[nt]->setMaxSearchDistance(8);
          trackers[nt]->setDisplay(display);
          trackers[nt]->setNetworkMode(false);
          trackers[nt]->setConsideringDullEdges(dull_edge);
          trackers[nt]->setTracking(true);
          trackers[nt]->setCannyHigh(55);
          trackers[nt]->setCannyLow(30);
          try
          {
            ((TexturelessParticleFilterTracker *)trackers[nt])->generate_tracker(origin_tracker_, detectedState[nt]);
            lifetime[nt] = 1;
            trackers[nt]->renderResults();
          }
          catch (exception &e)
          {
            std::cout << e.what() << std::endl;
          }
        }
      }
    }
    else
    {
      for (int nt = 0; nt < min(numOfDetections, 10); nt++)
      {
        if (lifetime[nt] <= 0)
        {
          continue;
        }
        int trackresult = trackers[nt]->tracking();
        if (trackresult != 0)
        {
          lifetime[nt]--;
          numOfTracking--;
          continue;
        }
        trackers[nt]->renderResults();
      }
      if (numOfTracking <= 0)
      {
        istrack = false;
      }
      timer.printTimeMilliSec("object tracking");
    }
    Mat img_edge = origin_tracker_->getEdgeImage();
    Mat img_result = origin_tracker_->getResultImage();
    Mat img_mask = origin_tracker_->getMaskImage();

    // Display the resulting frame
    imshow("Frame", frame);
    //imshow("Edge", img_edge);
    imshow("Result", img_result);
    imshow("Mask", img_mask);

    // Press  ESC on keyboard to exit
    char c = (char)waitKey(25);
    if (c == 27)
      break;
  }

  // When everything done, release the video capture object
  cap.release();
  */
  //Mat img_edge = origin_tracker_->getEdgeImage();
  // Mat img_result = origin_tracker_->getResultImage();
  //Mat img_mask = origin_tracker_->getMaskImage();

  // Display the resulting frame
  //imshow("Frame", inputImg);
  //imshow("Edge", img_edge);
  // imshow("Result", img_result);
  //imshow("Mask", img_mask);
  waitKey(0);
  // Closes all the frames
  destroyAllWindows();

  origin_tracker_->clean();
  delete origin_tracker_;
  return (0);
}