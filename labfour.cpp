/**
 * Computer Vision - Final Project
 * Name         : Wishnuputra Dhanu
 * Student ID   : 2013067
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "panoramic_utils.h"

// a struct to hold SIFT features
struct Features{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

// Image folder name, file extension, and features type
const cv::String IMAGE_FOLDER = "kitchen"; // "venezia", "kitchen"
const cv::String IMAGE_EXTENSION = "bmp"; // "png", "bmp"
const cv::String FEATURES_TYPE = "SIFT"; // "SIFT", "ORB"

// Ratio for refining the matches
// when using SIFT, RATIO_THRESHOLD = 3.0f
// when using ORB, RATIO_THRESHOLD = 6.0f
const float RATIO_THRESHOLD = 3.0f;

// Function header for STEP-1 until STEP-5 consecutively
std::vector<cv::Mat> projectImageIntoCylinder(cv::String imageFolder);
std::vector<Features> extractFeatures(std::vector<cv::Mat>& listOfProjectedImages, cv::String featuresType);
std::vector<std::vector<cv::DMatch>> findMatchingFeatures(std::vector<cv::Mat>& listOfProjectedImages,
                                             std::vector<Features>& listOfFeatures);
std::vector<cv::Mat> estimateTranslation(std::vector<cv::Mat>& listOfProjectedImages,
                            std::vector<Features>& listOfFeatures,
                            std::vector<std::vector<cv::DMatch>>& listOfMatches);
void buildPanoramicImage(std::vector<cv::Mat>& listOfProjectedImages,
                         std::vector<cv::Mat>& listOfHomography);

// Helper function to find the minimum distance between matches
int findMinimumDistance(std::vector<cv::DMatch>& matches);


int main(int argc, char** argv)
{
    // STEP-1 Project images into cylinder
    std::vector<cv::Mat> listOfProjectedImages;
    listOfProjectedImages = projectImageIntoCylinder(IMAGE_FOLDER);

    // STEP-2 Extract feature using SIFT or ORB
    std::vector<Features> listOfFeatures;
    listOfFeatures = extractFeatures(listOfProjectedImages, FEATURES_TYPE);

    // STEP-3 Find matching features
    std::vector<std::vector<cv::DMatch>> listOfMatches;
    listOfMatches = findMatchingFeatures(listOfProjectedImages, listOfFeatures);

    // STEP-4 Estimate translation between couples of adjacent images
    std::vector<cv::Mat> listOfHomography;
    listOfHomography = estimateTranslation(listOfProjectedImages,listOfFeatures,listOfMatches);

    // STEP-5 Build panoramic image
    buildPanoramicImage(listOfProjectedImages, listOfHomography);

    cv::destroyAllWindows();
    return 0;
}

/**
 * This function will find the minimum distance that is recorded in
 * the matches array.
 * @param matches
 * @return
 */
int findMinimumDistance(std::vector<cv::DMatch>& matches)
{
    int min_distance = matches[0].distance;
    for (int i = 0; i < matches.size(); ++i)
    {
        if (matches[i].distance < min_distance)
        {
            min_distance = matches[i].distance;
        }
    }
    return min_distance;
}

/**
 * This function will read all images in the selected folder and then
 * projected them using cylindrical projection. Then this function will
 * return an array of the projected images.
 */
std::vector<cv::Mat> projectImageIntoCylinder(cv::String imageFolder)
{
    // Read file names in the image folder
    std::vector<cv::String> imageFileNames;
    cv::String directory = imageFolder + "/i*." + IMAGE_EXTENSION;
    cv::glob(directory, imageFileNames);

    // Project all images in the folder
    cv::Mat originalImage;
    cv::Mat projectedImage;
    std::vector<cv::Mat> listOfProjectedImages(imageFileNames.size());
    for (int i = 0; i < imageFileNames.size(); ++i) {
        originalImage = cv::imread(imageFileNames[i]);
        projectedImage = PanoramicUtils::cylindricalProj(originalImage, 30);
        listOfProjectedImages[i] = projectedImage;
    }

    return listOfProjectedImages;
}

/**
 * This function will extract the features on each images using the specified featuresType.
 * Then it will save all the features inside a vector and return it.
 * @param listOfProjectedImages
 * @return
 */
std::vector<Features> extractFeatures(std::vector<cv::Mat>& listOfProjectedImages, cv::String featuresType)
{
    std::vector<Features> listOfFeatures;
    int numberOfImages = listOfProjectedImages.size();
    cv::Ptr<cv::SIFT> sift;
    cv::Ptr<cv::ORB> orb;

    if (featuresType == "SIFT") {
        sift = cv::SIFT::create();
    } else if (featuresType == "ORB") {
        orb = cv::ORB::create();
    }

    for (int i = 0; i < numberOfImages; ++i) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        cv::Mat img = listOfProjectedImages[i];
        if (featuresType == "SIFT") {
            sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        } else if (featuresType == "ORB") {
            orb->detectAndCompute(img,cv::noArray(), keypoints,descriptors);
        }

        Features features;
        features.keypoints = keypoints;
        features.descriptors = descriptors;

        listOfFeatures.push_back(features);
    }

    return listOfFeatures;
}

/**
 * This function will find matching features between a pair of images using the struct Features.
 * Then it will refine the matching features based on the ratio threshold. This means only
 * the matching features with distance less than ratio * min_distance that will be saved.
 * It will then save all the refined matching features in a vector and return it at the end.
 * @param listOfProjectedImages
 * @param listOfFeatures
 * @return
 */
std::vector<std::vector<cv::DMatch>> findMatchingFeatures(std::vector<cv::Mat>& listOfProjectedImages,
                                             std::vector<Features>& listOfFeatures)
{
    std::vector<std::vector<cv::DMatch>> listOfMatches;

    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::BFMatcher> matcher;

    if (FEATURES_TYPE == "SIFT") {
        matcher = cv::BFMatcher::create(cv::NORM_L2, false);
    } else if (FEATURES_TYPE == "ORB") {
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    }

    int numberOfImages = listOfProjectedImages.size();

    for (int i = 0; i < numberOfImages - 1; ++i)
    {
        std::vector<cv::DMatch> good_matches;
        Features features1 = listOfFeatures[i];
        Features features2 = listOfFeatures[i + 1];
        cv::Mat descriptors1 = features1.descriptors;
        cv::Mat descriptors2 = features2.descriptors;
        std::vector<cv::KeyPoint> keypoints1 = features1.keypoints;
        std::vector<cv::KeyPoint> keypoints2 = features2.keypoints;
        cv::Mat img1 = listOfProjectedImages[i];
        cv::Mat img2 = listOfProjectedImages[i + 1];

        matcher->match(descriptors1, descriptors2, matches);

        //-- Refine matches by selecting matches with distance less than ratio * min_distance
        int min_distance = findMinimumDistance(matches);
        for (int j = 0; j < matches.size(); j++)
        {
            if (matches[j].distance < RATIO_THRESHOLD * min_distance)
            {
                good_matches.push_back(matches[j]);
            }
        }

        listOfMatches.push_back(good_matches);
    }

    return listOfMatches;
}

/**
 * This function will estimate the translation matrix H using the findHomography method.
 * Then it will save all the translation matrices in a vector and return it.
 * @param listOfProjectedImages
 * @param listOfFeatures
 * @param listOfMatches
 * @return
 */
std::vector<cv::Mat> estimateTranslation(std::vector<cv::Mat>& listOfProjectedImages,
                            std::vector<Features>& listOfFeatures,
                            std::vector<std::vector<cv::DMatch>>& listOfMatches)
{
    std::vector<cv::Mat> listOfHomography;  // List of homography matrix

    int numberOfImages = listOfProjectedImages.size();
    for (int i = 0; i < numberOfImages - 1; ++i)
    {
        //-- Localize the object
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;

        std::vector<cv::DMatch> good_matches;
        good_matches = listOfMatches[i];
        Features features1 = listOfFeatures[i];
        Features features2 = listOfFeatures[i + 1];
        std::vector<cv::KeyPoint> keypoints1 = features1.keypoints;
        std::vector<cv::KeyPoint> keypoints2 = features2.keypoints;
        cv::Mat img1 = listOfProjectedImages[i];
        cv::Mat img2 = listOfProjectedImages[i + 1];

        cv::Mat img_matches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        for( size_t j = 0; j < good_matches.size(); j++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints1[ good_matches[j].queryIdx ].pt );
            scene.push_back( keypoints2[ good_matches[j].trainIdx ].pt );
        }

        cv::Mat H = findHomography( obj, scene, cv::RANSAC );
        listOfHomography.push_back(H);

        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cv::Point2f(0, 0);
        obj_corners[1] = cv::Point2f( (float)img1.cols, 0 );
        obj_corners[2] = cv::Point2f( (float)img1.cols, (float)img1.rows );
        obj_corners[3] = cv::Point2f( 0, (float)img1.rows );
        std::vector<cv::Point2f> scene_corners(4);

        cv::perspectiveTransform( obj_corners, scene_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + cv::Point2f((float)img1.cols, 0),
              scene_corners[1] + cv::Point2f((float)img1.cols, 0), cv::Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + cv::Point2f((float)img1.cols, 0),
              scene_corners[2] + cv::Point2f((float)img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + cv::Point2f((float)img1.cols, 0),
              scene_corners[3] + cv::Point2f((float)img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + cv::Point2f((float)img1.cols, 0),
              scene_corners[0] + cv::Point2f((float)img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );

        //-- Show detected matches
        cv::imshow("Good Matches & Object detection", img_matches );
        cv::waitKey();
    }

    return listOfHomography;
}

/**
 * This function will stitch all the images using the translation matrix H and warpPerspective method.
 * After the panoramic image is constructed, it will equalize the histogram to enhance the contrast.
 * @param listOfProjectedImages
 * @param listOfHomography
 */
void buildPanoramicImage(std::vector<cv::Mat>& listOfProjectedImages,
                         std::vector<cv::Mat>& listOfHomography)
{
    int numberOfImages = listOfProjectedImages.size();

    cv::Mat H = listOfHomography[0];
    cv::Mat img1 = listOfProjectedImages[0];
    cv::Mat img_stitch;

    for (int i = 0; i < numberOfImages - 1; ++i)
    {
        H = listOfHomography[i];
        cv::Mat img2 = listOfProjectedImages[i + 1];

        int x_translation = -H.at<double>(0,2);

        // Use only components of H matrix which are related to the image translation
        if (FEATURES_TYPE == "SIFT") {
            H.at<double>(0,0) = 1;
            H.at<double>(0,1) = 0;
            H.at<double>(0,2) = - img2.cols - H.at<double>(0,2);
            H.at<double>(1,0) = 0;
//            H.at<double>(1,1) = 1;
//            H.at<double>(1,2) = 0;
            H.at<double>(2,0) = 0;
//            H.at<double>(2,1) = 0;
//            H.at<double>(2,2) = 1;
        } else if (FEATURES_TYPE == "ORB") {
            H.at<double>(0,0) = 1;
            H.at<double>(0,1) = 0;
            H.at<double>(0,2) = - img2.cols - H.at<double>(0,2);
            H.at<double>(1,0) = 0;
            H.at<double>(1,1) = 1;
            H.at<double>(1,2) = 0;
            H.at<double>(2,0) = 0;
            H.at<double>(2,1) = 0;
            H.at<double>(2,2) = 1;
        }

        cv::Mat img2_result;
        cv::warpPerspective(img2, img2_result, H, cv::Size(2 * img2.cols, img2.rows));

        cv::Range rowRange = cv::Range::all();
        cv::Range colRange = cv::Range(0, x_translation);
        img2_result = img2_result.operator()(rowRange, colRange);

        cv::hconcat(img1, img2_result, img_stitch);
        img1 = img_stitch;
        imshow("img_stitch", img_stitch);

        cv::waitKey();
    }

    // Crop the image to hide the black background
    int y_translation = std::abs(H.at<double>(1,2));
    cv::Range rowRange = cv::Range(y_translation,img_stitch.rows - y_translation);
    cv::Range colRange = cv::Range::all();
    img_stitch = img_stitch.operator()(rowRange, colRange);

    // Equalize the histogram of the stitched image
    cv::Mat final_image;
    cv::equalizeHist(img_stitch,final_image);
    imshow("Final Image", final_image);
    imwrite("final_image.png",final_image);

    cv::waitKey();
}

/**
 * This function will project an image with cylindrical projection
 * @param image
 * @param angle
 * @return
 */
cv::Mat PanoramicUtils::cylindricalProj(
        const cv::Mat& image,
        const double angle)
{
    cv::Mat tmp, result;
    cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
    result = tmp.clone();


    double alpha(angle / 180 * CV_PI);
    double d((image.cols / 2.0) / tan(alpha));
    double r(d / cos(alpha));
    double d_by_r(d / r);
    int half_height_image(image.rows / 2);
    int half_width_image(image.cols / 2);

    for (int x = -half_width_image + 1,
                 x_end = half_width_image; x < x_end; ++x)
    {
        for (int y = -half_height_image + 1,
                     y_end = half_height_image; y < y_end; ++y)
        {
            double x1(d * tan(x / r));
            double y1(y * d_by_r / cos(x / r));

            if (x1 < half_width_image &&
                x1 > -half_width_image + 1 &&
                y1 < half_height_image &&
                y1 > -half_height_image + 1)
            {
                result.at<uchar>(y + half_height_image, x + half_width_image)
                        = tmp.at<uchar>(round(y1 + half_height_image),
                                        round(x1 + half_width_image));
            }
        }
    }

    return result;
}


