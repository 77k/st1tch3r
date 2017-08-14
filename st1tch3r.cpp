/**
 * snaske@77k.eu
 */

#include <boost/asio.hpp>
#include <boost/regex.hpp>
#include <string>
#include <stack>
#include <boost/array.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/logic/tribool.hpp>
#include <signal.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <boost/lexical_cast.hpp>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"




int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2)
        {
            std::cerr << "Usage: st1tch3r image1 image2 ... imagen" 
		      << " "
		      << std::endl;
            return 1;
        }
	std::cout << "argc = " << argc << std::endl;
	std::vector< std::string > image_filenames;
	std::vector< cv::String > image_names;
	for(int i = 1; i <  argc; i++)
	{
		image_filenames.push_back(argv[i]);
		image_names.push_back(cv::String(argv[i]));
	}

	std::vector<cv::Mat> images;
	std::vector<cv::detail::ImageFeatures> features;
	cv::Mat tmp_image;
	cv::detail::ImageFeatures tmp_features;
	auto clock_start = clock();
	auto feature_detector = cv::makePtr<cv::detail::SurfFeaturesFinder>();
	for(auto s: image_filenames)
	{
		std::cout << "trying to load: " << s << std::endl;
		tmp_image = cv::imread(s);
		if(tmp_image.empty())
		{
			std::cerr << "Can not load: " << s << std::endl;
		}else{
			(*feature_detector)(tmp_image, tmp_features);
			std::cout << "Features found in " << s << " : " << tmp_features.keypoints.size() << std::endl;
			images.push_back(tmp_image);
			tmp_features.img_idx = images.size() - 1;
			features.push_back(tmp_features);
			feature_detector->collectGarbage();
		};
	}
	
	std::cout << (double(clock() - clock_start))/CLOCKS_PER_SEC;
	std::cout << " seconds needed for feature detection" << std::endl;
	clock_start = clock();
	cv::Ptr<cv::detail::FeaturesMatcher> matcher;
	std::vector<cv::detail::MatchesInfo> pairwise_matches;
	matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(false, 0.65);
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	std::cout << (double(clock() - clock_start))/CLOCKS_PER_SEC;
	std::cout << " seconds needed for feature and image matching" << std::endl;

	std::cout << cv::detail::matchesGraphAsString(image_names, pairwise_matches, 1.0);

	cv::Ptr<cv::detail::Estimator> estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();
	std::vector<cv::detail::CameraParams> cameras;
	if(!(*estimator)(features, pairwise_matches, cameras))
	{
		std::cout << "estimation failed" << std::endl;
		return -1;
	}

	for(auto &c: cameras)
	{
		cv::Mat R;
		c.R.convertTo(R, CV_32F);
		c.R = R;
	}

	cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
	adjuster->setConfThresh(1.0);
	cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
	for(int i = 0; i < 3; i++) refine_mask(0, i) = 1;
	for(int i = 1; i < 3; i++) refine_mask(1, i) = 1;
	adjuster->setRefinementMask(refine_mask);
	if(!(*adjuster)(features, pairwise_matches, cameras))
	{
		std::cout << "Camera Parameter adjustment failed." << std::endl;
		return -1;
	}
	std::vector< double > focals;
	for(auto &c: cameras) focals.push_back(c.focal);
	std::sort(std::begin(focals), std::end(focals));
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	int num_images = images.size();
	
	std::vector<cv::Point> corners(num_images);
	std::vector<cv::UMat> masks_warped(num_images);
	std::vector<cv::UMat> images_warped(num_images);
	std::vector<cv::Size> sizes(num_images);
	std::vector<cv::UMat> masks(num_images);

	for(int i = 0; i < num_images; i++)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(cv::Scalar::all(255));
	}

	cv::Ptr< cv::WarperCreator > warper_creator = cv::makePtr< cv::PlaneWarper >();
	cv::Ptr< cv::detail::RotationWarper > warper = warper_creator->create(warped_image_scale);
	for (int i = 0; i < num_images; ++i)
	{
		cv::Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)1;
		K(0,0) *= swa; K(0,2) *= swa;
		K(1,1) *= swa; K(1,2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
	}

	int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
	cv::Ptr<cv::detail::ExposureCompensator> compensator = cv::detail::ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	cv::Ptr<cv::detail::SeamFinder> seam_finder;
	

    }
    catch (std::exception& e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "He's Dead Jim" << std::endl;
    }
    return 0;

}
