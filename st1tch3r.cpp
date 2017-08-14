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

bool preview = false;
bool try_cuda = false;
double work_megapix = 2.0;
double seam_megapix = 2;
double compose_megapix = -1;
float conf_thresh = 1.f;
bool do_wave_correct = true;
bool save_graph = false;
float match_conf = 0.3f;
float blend_strength = 5;
bool timelapse = false;



int main(int argc, char* argv[])
{

   double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
	double seam_work_aspect = 1;
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
	std::string output(argv[1]);
	std::cout << "write to: " << output << std::endl;
	std::vector< std::string > image_filenames;
	std::vector< cv::String > image_names;
	for(int i = 2; i <  argc; i++)
	{
		image_filenames.push_back(argv[i]);
		image_names.push_back(cv::String(argv[i]));
	}

	std::vector<cv::Mat> images;
	std::vector<cv::Size> full_image_sizes;
	std::vector<cv::detail::ImageFeatures> features;
	cv::Mat tmp_image, image;
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
			if (work_megapix < 0)
			{
				image = tmp_image;
				work_scale = 1;
				is_work_scale_set = true;
			}
			else
			{
				if (!is_work_scale_set)
				{
					work_scale =  std::min(1.0, sqrt(work_megapix * 1e6 / tmp_image.size().area()));
					is_work_scale_set = true;
				}
				resize(tmp_image, image, cv::Size(), work_scale, work_scale);
			}
			if (!is_seam_scale_set)
			{
				seam_scale = std::min(1.0, sqrt(seam_megapix * 1e6 / tmp_image.size().area()));
				seam_work_aspect = seam_scale / work_scale;
				is_seam_scale_set = true;
			}

			cv::resize(tmp_image, image, cv::Size(), seam_scale, seam_scale);	
			full_image_sizes.push_back(tmp_image.size());
			(*feature_detector)(image, tmp_features);
			std::cout << "Features found in " << s << " : " << tmp_features.keypoints.size() << std::endl;
			images.push_back(image);
			tmp_features.img_idx = images.size() - 1;
			features.push_back(tmp_features);
			feature_detector->collectGarbage();
		};
	}
	image.release();
	tmp_image.release();	
	std::cout << (double(clock() - clock_start))/CLOCKS_PER_SEC;
	std::cout << " seconds needed for feature detection" << std::endl;
	clock_start = clock();
	cv::Ptr<cv::detail::FeaturesMatcher> matcher;
	std::vector<cv::detail::MatchesInfo> pairwise_matches;
	matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(false, 0.35);
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	std::cout << (double(clock() - clock_start))/CLOCKS_PER_SEC;
	std::cout << " seconds needed for feature and image matching" << std::endl;
	clock_start = clock();
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
	//for(int i = 0; i < 3; i++) refine_mask(0, i) = 1;
	//for(int i = 1; i < 3; i++) refine_mask(1, i) = 1;

	refine_mask(0,0) = 1;
        refine_mask(0,1) = 1;
        refine_mask(0,2) = 1;
        refine_mask(1,1) = 1;
        refine_mask(1,2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if(!(*adjuster)(features, pairwise_matches, cameras))
	{
		std::cout << "Camera Parameter adjustment failed." << std::endl;
		return -1;
	}
	
	std::cout << (double(clock() - clock_start))/CLOCKS_PER_SEC;
	std::cout << " seconds needed for bundle adjust" << std::endl;
	clock_start = clock();

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
	cv::Ptr< cv::detail::RotationWarper > warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
	for (int i = 0; i < num_images; ++i)
	{
		cv::Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float) seam_work_aspect;
		K(0,0) *= swa; K(0,2) *= swa;
		K(1,1) *= swa; K(1,2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
	}

	std::cout << (double(clock() - clock_start))/CLOCKS_PER_SEC;
	std::cout << " seconds needed for warping" << std::endl;
	clock_start = clock();
	std::vector<cv::UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
	{
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	}

	int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
	cv::Ptr<cv::detail::ExposureCompensator> compensator = cv::detail::ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr< cv::detail::GraphCutSeamFinder >(cv::detail::GraphCutSeamFinderBase::COST_COLOR);

	

	seam_finder->find(images_warped_f, corners, masks_warped);


	std::cout << (double(clock() - clock_start))/CLOCKS_PER_SEC;
	std::cout << " seconds needed for feature and seam finding" << std::endl;
	clock_start = clock();

	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	cv::Mat img_warped, img_warped_s;
	cv::Mat dilated_mask, seam_mask, mask, mask_warped;
	cv::Ptr<cv::detail::Blender> blender;
	cv::Ptr<cv::detail::Timelapser> timelapser;

	float compose_megapix = -1;
	bool is_compose_scale_set = false;
	double compose_work_aspect = 1.0;
	for(int image_idx = 0; image_idx < num_images; image_idx++)
	{
		std::cout << "compositing image #" << image_idx << std::endl;
		tmp_image = cv::imread(image_names[image_idx]);
		if(!is_compose_scale_set)
		{
			if(compose_megapix > 0)
			{
				compose_scale = std::min(1.0, sqrt(compose_megapix * 1e6 / tmp_image.size().area()));
			}
			is_compose_scale_set = true;
	//		compose_scale = 0.1;
			compose_work_aspect = compose_scale / work_scale;

			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			for (int i = 0; i < num_images; ++i)
			{
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				cv::Size sz = full_image_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_image_sizes[i].width * compose_scale);
					sz.height = cvRound(full_image_sizes[i].height * compose_scale);
				}

				cv::Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				cv::Rect roi = warper->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (abs(compose_scale - 1) > 1e-1)
			cv::resize(tmp_image, image, cv::Size(), compose_scale, compose_scale);
		else
			image =  tmp_image;
		tmp_image.release();
		cv::Size img_size = image.size();

		cv::Mat K;
		cameras[image_idx].K().convertTo(K, CV_32F);

		warper->warp(image, K, cameras[image_idx].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, img_warped);

		mask.create(img_size, CV_8U);
		mask.setTo(cv::Scalar::all(255));
		warper->warp(mask, K, cameras[image_idx].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);

		compensator->apply(image_idx, corners[image_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		image.release();
		mask.release();
		if(!blender)
		{
			cv::dilate(masks_warped[image_idx], dilated_mask, cv::Mat());
			cv::resize(dilated_mask, seam_mask, mask_warped.size());
			mask_warped = seam_mask & mask_warped;
			float blend_strength = 5.0;
			blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
			cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if(blend_width < 1.0)
			{
				blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, false);
			} else if (true)
			{
				blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
				cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
			}

			blender->prepare(corners, sizes);
		}
		blender->feed(img_warped_s, mask_warped, corners[image_idx]);
	}
	cv::Mat result, result_mask;
	blender->blend(result, result_mask);
	cv::imwrite(output, result);
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
