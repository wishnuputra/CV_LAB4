#include <iostream>
#include <opencv2/opencv.hpp>
#include "panoramic_utils.h"

// class to hold SIFT features
struct siftFeatures{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

// Function header for STEP-1 until STEP-5 consecutively
std::vector<cv::Mat> projectImageIntoCylinder(cv::String folderName);
std::vector<siftFeatures> extractSiftFeatures(std::vector<cv::Mat>& listOfProjectedImages);
std::vector<std::vector<cv::DMatch>> findMatchingFeatures(std::vector<cv::Mat>& listOfProjectedImages,
                                             std::vector<siftFeatures>& listOfFeatures);
std::vector<cv::Mat> estimateTranslation(std::vector<cv::Mat>& listOfProjectedImages,
                            std::vector<siftFeatures>& listOfFeatures,
                            std::vector<std::vector<cv::DMatch>>& listOfMatches);
void buildPanoramicImage(std::vector<cv::Mat>& listOfProjectedImages,
                         std::vector<cv::Mat>& listOfHomography);


// Helper function to find the minimum distance between matches
int findMinimumDistance(std::vector<cv::DMatch>& matches);

int main(int argc, char** argv)
{
    // STEP-1 Project images into cylinder
    std::vector<cv::Mat> listOfProjectedImages;
    cv::String folderName = "kitchen";
    listOfProjectedImages = projectImageIntoCylinder(folderName);

    // STEP-2 Extract feature using SIFT
    std::vector<siftFeatures> listOfFeatures;
    listOfFeatures = extractSiftFeatures(listOfProjectedImages);

    // STEP-3 Find matching features
    std::vector<std::vector<cv::DMatch>> listOfMatches;
    listOfMatches = findMatchingFeatures(listOfProjectedImages, listOfFeatures);
//    std::cout << "list of matches size: " << listOfMatches.size() << std::endl;

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
    int index = 0;
    for (int i = 0; i < matches.size(); ++i)
    {
//        std::cout << "distance " << i << ": " << matches[i].distance << std::endl;
        if (matches[i].distance < min_distance)
        {
            min_distance = matches[i].distance;
            index = i;
        }
    }
//    std::cout << "min distance at index " << index << ": " << min_distance << std::endl;
    return min_distance;
}

/**
 * This function will read all images in the selected folder and then
 * projected them using cylindrical projection. Then this function will
 * return an array of the projected images.
 * @param folderName
 * @return
 */
std::vector<cv::Mat> projectImageIntoCylinder(cv::String folderName)
{
    // Read file names in the image folder
    std::vector<cv::String> imageFileNames;
    cv::String directory = folderName + "/i*.bmp";
    cv::glob(directory, imageFileNames);

//    std::cout << imageFileNames[0] << std::endl;

    cv::Mat originalImage;
    cv::Mat projectedImage;
    std::vector<cv::Mat> listOfProjectedImages(imageFileNames.size());
    for (int i = 0; i < imageFileNames.size(); ++i) {
        originalImage = cv::imread(imageFileNames[i]);
//        cv::imshow("img1",originalImage);
//        cv::waitKey();
        projectedImage = PanoramicUtils::cylindricalProj(originalImage, 30);
        listOfProjectedImages[i] = projectedImage;
    }

    return listOfProjectedImages;
};

std::vector<siftFeatures> extractSiftFeatures(std::vector<cv::Mat>& listOfProjectedImages)
{
    std::vector<siftFeatures> listOfFeatures;
    int numberOfImages = listOfProjectedImages.size();
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    for (int i = 0; i < numberOfImages; ++i) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        cv::Mat img = listOfProjectedImages[i];
        sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

        siftFeatures features;
        features.keypoints = keypoints;
        features.descriptors = descriptors;

        listOfFeatures.push_back(features);
    }

    return listOfFeatures;
}

std::vector<std::vector<cv::DMatch>> findMatchingFeatures(std::vector<cv::Mat>& listOfProjectedImages,
                                             std::vector<siftFeatures>& listOfFeatures)
{
    std::vector<std::vector<cv::DMatch>> listOfMatches;

    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);
    int numberOfImages = listOfProjectedImages.size();

    for (int i = 0; i < numberOfImages - 1; ++i)
    {
        std::vector<cv::DMatch> good_matches;
        siftFeatures features1 = listOfFeatures[i];
        siftFeatures features2 = listOfFeatures[i + 1];
        cv::Mat descriptors1 = features1.descriptors;
        cv::Mat descriptors2 = features2.descriptors;
        std::vector<cv::KeyPoint> keypoints1 = features1.keypoints;
        std::vector<cv::KeyPoint> keypoints2 = features2.keypoints;
        cv::Mat img1 = listOfProjectedImages[i];
        cv::Mat img2 = listOfProjectedImages[i + 1];

        matcher->match(descriptors1, descriptors2, matches);

        //-- Refine matches by selecting matches with distance less than ratio * min_distance
        int min_distance = findMinimumDistance(matches);
        const float ratio_thresh = 3.0f;

//        std::cout << "matches size: " << matches.size() << std::endl;

        for (int j = 0; j < matches.size(); j++)
        {
            if (matches[j].distance < ratio_thresh * min_distance)
            {
                good_matches.push_back(matches[j]);
//                std::cout << "good matches at " << j << ": " << good_matches.size() << std::endl;
            }
        }

//        std::cout << "good matches size: " << good_matches.size() << std::endl;

        listOfMatches.push_back(good_matches);

//        cv::Mat img_matches;
//        cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
//                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//
//        cv::imshow("Matches", img_matches);
//        cv::waitKey();
    }

    return listOfMatches;
}


std::vector<cv::Mat> estimateTranslation(std::vector<cv::Mat>& listOfProjectedImages,
                            std::vector<siftFeatures>& listOfFeatures,
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
        siftFeatures features1 = listOfFeatures[i];
        siftFeatures features2 = listOfFeatures[i + 1];
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


void buildPanoramicImage(std::vector<cv::Mat>& listOfProjectedImages,
                         std::vector<cv::Mat>& listOfHomography)
{
    int numberOfImages = listOfProjectedImages.size();

    // Prepare a large canvas to hold all the stitched images
    cv::Mat image = listOfProjectedImages[0];
    cv::Mat panoramicImage(image.rows, numberOfImages * image.cols, image.type());

    cv::Mat img1 = listOfProjectedImages[0];

    for (int i = 0; i < numberOfImages - 1; ++i)
    {
        cv::Mat H = listOfHomography[i];
        cv::Mat img2 = listOfProjectedImages[i + 1];

        H.at<double>(0,2) = - img1.cols - H.at<double>(0,2);
        cv::Mat img2_result;
        cv::warpPerspective(img2, img2_result, H, cv::Size(2 * img2.cols, img2.rows));

        cv::Mat result(img1.rows, 2 * img1.cols, img1.type());

        cv::Mat leftResult = result(cv::Rect(0, 0, img1.cols, img1.rows));
        cv::Mat rightResult = result(cv::Rect(img1.cols, 0, img2.cols, img2.rows));

        cv::Mat image1 = img1(cv::Rect(0, 0, img1.cols, img1.rows));
        cv::Mat image2 = img2_result(cv::Rect(0, 0, img2.cols, img2.rows));

        image1.copyTo(leftResult); //Img1 will be on the left of result
        image2.copyTo(rightResult); //Img2 will be on the right of result

        cv::Range rowRange = cv::Range::all();
        int offset =  - H.at<double>(0,2);
        std::cout << offset << std::endl;
        cv::Range colRange = cv::Range(0, img1.cols + offset/2);
        result = result.operator()(rowRange, colRange);
        imshow("Final Image", result);
        img1 = result;
        cv::waitKey();
    }
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


