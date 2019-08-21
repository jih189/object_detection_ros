#include <string>

//#include "tracker_irls.h"
//#include "tracker_pf_texture.h"
//#include "tracker_pf_textureless.h"

#include <boost/signals2/mutex.hpp> 
#include <boost/thread.hpp>
#include <boost/array.hpp>
#include <boost/asio.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <omp.h>        // openmp
#include <iostream>
//#include <direct.h> // mkdir(), creating a directory




#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <limits.h>

#include "object_tracking_2D/ObjectModel.h"

#include "Fitline/LFLineFitter.h"

namespace po = boost::program_options;

CvPoint2D32f project3Dto2D(CvPoint3D32f pt3, CvMat* pose, CvMat* param_intrinsic)
{
  CvPoint2D32f pt2;
  CvPoint3D32f pt3_cam;
  pt3_cam.x = CV_MAT_ELEM(*pose, float, 0, 0)*pt3.x + CV_MAT_ELEM(*pose, float, 0, 1)*pt3.y + CV_MAT_ELEM(*pose, float, 0, 2)*pt3.z + CV_MAT_ELEM(*pose, float, 0, 3);
  pt3_cam.y = CV_MAT_ELEM(*pose, float, 1, 0)*pt3.x + CV_MAT_ELEM(*pose, float, 1, 1)*pt3.y + CV_MAT_ELEM(*pose, float, 1, 2)*pt3.z + CV_MAT_ELEM(*pose, float, 1, 3);
  pt3_cam.z = CV_MAT_ELEM(*pose, float, 2, 0)*pt3.x + CV_MAT_ELEM(*pose, float, 2, 1)*pt3.y + CV_MAT_ELEM(*pose, float, 2, 2)*pt3.z + CV_MAT_ELEM(*pose, float, 2, 3);

  float fx = CV_MAT_ELEM(*param_intrinsic, float, 0, 0);
  float fy = CV_MAT_ELEM(*param_intrinsic, float, 1, 1);
  float ux = CV_MAT_ELEM(*param_intrinsic, float, 0, 2);
  float uy = CV_MAT_ELEM(*param_intrinsic, float, 1, 2);

  // jiaming hu: adding extrinsic matrix
  float tx = 0.0;//-148.43597135826298;
  float ty = 0.0;

  pt2.x = fx*pt3_cam.x/pt3_cam.z + ux + tx / pt3_cam.z;
  pt2.y = fy*pt3_cam.y/pt3_cam.z + uy + ty / pt3_cam.z;

  float th_d = 10000.0;
  if(pt2.x < -th_d || pt2.x > th_d || pt2.y < -th_d || pt2.y > th_d)
  {
    pt2.x = -1;
    pt2.y = -1;
  }

  return pt2;
}

CvRect drawModel(IplImage* img, std::vector<CvPoint3D32f> ep1, std::vector<CvPoint3D32f> ep2, CvMat* pose, CvMat* param_intrinsic, CvScalar color)
{
  float widthf = static_cast<float>(img->width), heightf = static_cast<float>(img->height);
  CvPoint2D32f pt21, pt22;
  float l = numeric_limits<float>::max(), r = 0.0f, t = numeric_limits<float>::max(), b = 0.0f;
  for(int i=0; i<ep1.size(); i++)
  {
    pt21 = project3Dto2D(ep1[i], pose, param_intrinsic);
    pt22 = project3Dto2D(ep2[i], pose, param_intrinsic);
    cvLine(img, cvPointFrom32f(pt21), cvPointFrom32f(pt22), color, 1, 8);
    if(pt21.x < l) l = pt21.x;
    if(pt21.x > r) r = pt21.x;
    if(pt21.y < t) t = pt21.y;
    if(pt21.y > b) b = pt21.y;

    if(pt22.x < l) l = pt22.x;
    if(pt22.x > r) r = pt22.x;
    if(pt22.y < t) t = pt22.y;
    if(pt22.y > b) b = pt22.y;
  }

  l = max(0.f, l);
  r = min(widthf - 1.f, r);
  t = max(0.f, t);
  b = min(heightf - 1.f, b);
  return cvRect(static_cast<int>(l), static_cast<int>(t), static_cast<int>(r-l+1.f), static_cast<int>(b-t+1.f));
}

int main(int argc, char **argv)
{
  LFLineFitter lf;
  lf.Init();
  lf.Configure("para_template_line_fitter.txt");
  for(int num = 0 ; num < 49 ; num++){
    char buf[100];

    sprintf( buf, "/home/jiaming/catkin_ws/src/object_tracking_2d_ros/data/megabloks/edge_template%03d.png", num);
    std::cout << buf << std::endl;

    // fit lines
    IplImage *img_result= cvLoadImage(buf, 0);
    lf.FitLine(img_result);
    sprintf(buf, "/home/jiaming/catkin_ws/src/object_tracking_2d_ros/data/megabloks/megabloks_edge_template_line%03d.png", num);
    lf.DisplayEdgeMap(img_result, buf);
    cvReleaseImage(&img_result);

    
    sprintf(buf, "/home/jiaming/catkin_ws/src/object_tracking_2d_ros/data/megabloks/megabloks_edge_template%03d.txt", num);
    lf.SaveEdgeMap(buf);
  }

/*
  
    
      // save edge template
      cvSetImageROI(img_result, bound);
      sprintf(buf, "/edge_template%03d.png", int_not);
      cvSaveImage((str_result_path + buf).c_str(), img_result);

      // fit lines
      IplImage* img = cvLoadImage((str_result_path + buf).c_str(), 0);
      lf.FitLine(img);
      sprintf(buf, "/%s_edge_template_line%03d.png", obj_name.c_str(), int_not);
      lf.DisplayEdgeMap(img, (str_result_path + buf).c_str());
      cvReleaseImage(&img);

      sprintf(buf, "/%s_edge_template%03d.txt", obj_name.c_str(), int_not);
      lf.SaveEdgeMap((str_result_path + buf).c_str());
      cvResetImageROI(img_result);

      // save template pose
      sprintf(buf, "/edge_template_pose%03d.xml", int_not);
      cvSave((str_result_path + buf).c_str(), pose);

      int_not++;
      break;
    }
    cvWaitKey(10);
  }

  cvDestroyWindow("Edge");
  cvReleaseMat(&pose);
  cvReleaseImage(&img_result);
  */
  return (0);
}
