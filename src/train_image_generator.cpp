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

CvRect drawModel(IplImage* img, std::vector<CvPoint3D32f> ep1, std::vector<CvPoint3D32f> ep2, CvMat* pose, CvMat* param_intrinsic, CvScalar color, CvPoint3D32f originPoint)
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

void rotate(CvMat* curr_pose, float x, float y, float z){
    cv::Mat R_x = (cv::Mat_<float>(4,4) <<
               1,       0,        0,          0,
               0,       cos(x),   -sin(x),    0,
               0,       sin(x),   cos(x),     0,
               0,       0,        0,          1
               );
     
    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<float>(4,4) <<
               cos(y),    0,      sin(y),     0,
               0,         1,      0,          0,
               -sin(y),   0,      cos(y),     0,
               0,         0,      0,          1
               );
     
    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<float>(4,4) <<
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

int main(int argc, char **argv)
{
  std::string obj_name;
  std::string intrinsic;
  int width;
  int height;
  float sample_step;
  bool dull_edge;
  std::string str_result_path;
  std::string str_param_linfit;
  float depth;

  po::options_description desc("\nTemplate generator keys:\nk: save current template\nw: translate further\ns: translate closer\nq,e,a,d,z,d: rotates in 3 axes\n\nTemplate generator options");
  desc.add_options()
    ("help,h", "produce help message")
    ("obj-name,o", po::value<std::string>(&obj_name), "name of traget object")
    ("sample-step,s", po::value<float>(&sample_step)->default_value(0.005f), "sample step")
    ("depth,d", po::value<float>(&depth)->default_value(1.0f), "distance between object and camera")
    ("param-linefit,l", po::value<std::string>(&str_param_linfit)->default_value(std::string("para_template_line_fitter.txt")), "set parameters for line fitting")
    ("save-path,p", po::value<std::string>(&str_result_path)->default_value(std::string("train_images")), "set result path")
    
    ("width", po::value<int>(&width)->default_value(640), "width")
    ("height", po::value<int>(&height)->default_value(480), "height")
    ("intrinsic", po::value<std::string>(&intrinsic)->default_value("Intrinsics_normal.xml"), "intrinsic parameters")
    ("dull_edge", po::value<bool>(&dull_edge)->default_value(true), "consider dull edges")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(argc < 2 || vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 1;
  }

  if(obj_name.empty())
  {
    std::cerr << "obj-name should be specified." << std::endl;
    return 1;
  }

  if(vm.count("obj-name"))
  {
    std::cout << "obj-name: " << vm["obj-name"].as<std::string>() << std::endl;
  }
 
  if(vm.count("save-path"))
  {
    if(mkdir(str_result_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST)
    {
      std::cerr << "Cannot create " << str_result_path << " directory for saving results." << std::endl;
      return false;
    }
  }

  IplImage* img_result = cvCreateImage(cvSize(width, height), 8, 1);
  CvMat* pose = cvCreateMat(4, 4, CV_32F);
  char key = 0;
  float modelPosition[3];
  float modelAngle[3];
  float matrixModel[16];
  modelPosition[0] = 0.0;
  modelPosition[1] = 0.0;
  modelPosition[2] = depth;
  modelAngle[0] = 0.0;
  modelAngle[1] = 0.0;
  modelAngle[2] = 0.0;

  CvMat* param_intrinsic = (CvMat*)cvLoad(intrinsic.c_str());


  // Create object model instance
  int maxd = 16;
  std::string tempname = obj_name + ".obj";
  CObjectModel cObjModel(tempname, width, height, param_intrinsic, sample_step, maxd, dull_edge, NULL);

  cvSetIdentity(pose);
  CV_MAT_ELEM(*pose, float, 2, 3) = depth;

  // init line fitter
  //LFLineFitter lf;
  //lf.Init();
  //lf.Configure("para_template_line_fitter.txt");


  vector<int> ma0Arr;
  vector<int> ma1Arr;
  vector<int> ma2Arr;

  ma0Arr.push_back(0);
  ma1Arr.push_back(0);
  ma2Arr.push_back(0);
  
  // todo
  for(int ma0 = -45; ma0 >= -135 ; ma0 -= 45){
    for(int ma2 = 0 ; ma2 < 360 ; ma2 += 45){
      ma0Arr.push_back(ma0);
      ma1Arr.push_back(0);
      ma2Arr.push_back(ma2);
    }
  }

  ma0Arr.push_back(180);
  ma1Arr.push_back(0);
  ma2Arr.push_back(0);

  int indexnum = 0;
  while(indexnum < ma0Arr.size()){
    modelAngle[0] = (float)ma0Arr[indexnum];
    modelAngle[1] = (float)ma1Arr[indexnum];
    modelAngle[2] = (float)ma2Arr[indexnum];
    

    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    // In order to get the modeling matrix only, reset GL_MODELVIEW matrix
    glLoadIdentity();
    // transform the object
    // From now, all transform will be for modeling matrix only. (transform from object space to world space)
    glTranslatef(modelPosition[0], modelPosition[1], modelPosition[2]);
    glRotatef(modelAngle[0], 1, 0, 0);
    glRotatef(modelAngle[1], 0, 1, 0);
    glRotatef(modelAngle[2], 0, 0, 1);

    // save modeling matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, matrixModel);

    CvMat* poset = cvCreateMat(4, 4, CV_32F);
    CvMat* rotatestate = cvCreateMat(4,4,CV_32F);
    CvMat* resultstate = cvCreateMat(4,4,CV_32F);


    memcpy(poset->data.fl, matrixModel, sizeof(float)*16);
    cvTranspose(poset, pose);
    cvReleaseMat(&poset);
    glPopMatrix();

    for(int rotz = 0 ; rotz < 8 ; rotz++ ){
      cvSet(img_result, cvScalar(0)); // reset image
      
      rotate(rotatestate, 0.0, 0.0, 0.785 * rotz);
      cvMatMul(rotatestate, pose, resultstate);

/*
      std::cout << "rotate\n";
      for(int r=0; r<4; r++){ 
        for(int c=0; c<4; c++){
          std::cout << CV_MAT_ELEM(*rotatestate, float, r, c) << " ";
        }
        std::cout << std::endl;
      }
*/
      CV_MAT_ELEM(*resultstate, float, 0, 3) = CV_MAT_ELEM(*pose, float, 0, 3);
      CV_MAT_ELEM(*resultstate, float, 1, 3) = CV_MAT_ELEM(*pose, float, 1, 3);
      CV_MAT_ELEM(*resultstate, float, 2, 3) = CV_MAT_ELEM(*pose, float, 2, 3);

      // Draw object model with visibility test
      cObjModel.setModelviewMatrix(resultstate); // update the initial pose to object model for displaying    
      cObjModel.findVisibleSamplePoints(); // draw object model with visibility test


      // Find visible edges
      std::vector<CvPoint3D32f> ep1, ep2;
    
      std::vector<CObjectModel::SamplePoint>& vsp = cObjModel.getVisibleSamplePoints();
      
      // determine two end points in each common edge_mem sample points
      int edge_mem = vsp[0].edge_mem;
      ep1.push_back(vsp[0].coord3);
      int i;
      for(i=0; i<int(vsp.size()); i++)
      {
        if(edge_mem != vsp[i].edge_mem){
          // new point, so add end/starting edge point
          ep2.push_back(vsp[i-1].coord3);
          ep1.push_back(vsp[i].coord3);
          // update new edge_mem value
          edge_mem = vsp[i].edge_mem;
        }
      }
      ep2.push_back(vsp[i-1].coord3);

      CvPoint3D32f originPoint={modelPosition[0],modelPosition[1],modelPosition[2]};
      CvRect bound = drawModel(img_result, ep1, ep2, resultstate, param_intrinsic, CV_RGB(255, 255, 255), originPoint);
      bound.x -= 2;
      bound.y -= 2;
      bound.width += 4;
      bound.height += 4;
      char buf[50];
      // jiaming hu: draw the point of origin point
      float fx = CV_MAT_ELEM(*param_intrinsic, float, 0, 0);
      float fy = CV_MAT_ELEM(*param_intrinsic, float, 1, 1);
      float ux = CV_MAT_ELEM(*param_intrinsic, float, 0, 2);
      float uy = CV_MAT_ELEM(*param_intrinsic, float, 1, 2);

      CvPoint2D32f ptori;
      ptori.x = fx*originPoint.x/originPoint.z + ux;
      ptori.y = fy*originPoint.y/originPoint.z + uy;

      // save center point
      ofstream centerfile;
      sprintf(buf, "/pose_position%03d.txt", indexnum * 8 + rotz);
      centerfile.open((str_result_path + buf).c_str());
      centerfile << ptori.x - bound.x << " " << ptori.y - bound.y << std::endl;
      centerfile.close();

      // save edge template
      cvSetImageROI(img_result, bound);
      sprintf(buf, "/edge_template%03d.png", indexnum * 8 + rotz);
      cvSaveImage((str_result_path + buf).c_str(), img_result);

      cvResetImageROI(img_result);

      // save template pose
      sprintf(buf, "/edge_template_pose%03d.xml", indexnum * 8 + rotz);
      cvSave((str_result_path + buf).c_str(), resultstate);
    }
    
    cvReleaseMat(&rotatestate);
    cvReleaseMat(&resultstate);
    indexnum++;
  }

  cvReleaseMat(&pose);
  cvReleaseImage(&img_result);
  
  return (0);
}
