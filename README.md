## Project Overview

For this project, we will be trying to take in video data from soccer games and classify which teams the players are on in real time. To do this, we are using an already trained Neural Network, called YoloV3, which is capable of detecting people and figuring out bounding boxes for the people. For each frame of video footage put into the neural network, we will get image frames corresponding to detected people as output.

To actually classify the individual frames as players on a certain team, we initially thought to use a neural network that would be capable of learning the underlying model over time as it was trained. However, we soon found that it would be impossible for us to extract and label-by-hand tens of thousands of player frame data, since there is no existing database on classified soccer players. Therefore, we decided to use the Expectation Maximization algorithm to classify the players. This was done by using a model consisting of average RGB color intensity in each channel of the image frame, after it was cropped to include only the jersey and shorts of the player. 

### Neural Network

The neural network used for this project is a *pytorch* adaptation of the YoloV3 object detector.  The network was implemented by following the tutorial at: https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch. YoloV3 is a fully convolutional 32 layer neural network that makes heavy use of residual blocks and skip connections.  The network outputs 3 different matrices with each element containing a vector that contains the likelihood of various objects at each spot.  The vector also contains the bounding box that is overload on the image.  

The output of the network is used for both the Motion Estimation and the Team Detection.  For Motion Estimation, the coordinates of each player is output and the optical flow is calculated around that point.  For Team Detection, the output is limited to people, and the image contained by the bounding box is classified by an EM algorithm.  The overlay on the image is then modified to include the team information.

*Note: The network is designed to work with Cuda.  CPU evaluation works without any necessary modifications, but the performance is significantly lacking.*

**How to run Motion Estimation**

The `video_optical.py` file is used to overlay the motion estimates of each player onto the frames.  An example of running this file would be `python3 video_optical.py —video "../Game Data/Liverpool vs. Roma Part 2/vlc-record-2018-04-29-21h47m03s-V1.mkv-.ts"`. Any of the included files in Gama Data can be tested via this file.  

**How to run Team Detection**

The `video_player.py` file is used to overlay identification bounding boxes of each player onto the frames.  To run this file, the command should look as follows `python3 video_player.py —video "../Game Data/Liverpool vs. Roma Part 2/vlc-record-2018-04-29-21h47m03s-V1.mkv-.ts"`

### Expectation Maximization

The code for the Expectation Maximization logic, which creates the classification model, is located in the **EM Classification Files**. This model will be trained to classify 6 types of players (4 teams and 2 different dressed referees), as well as one extra type to contain frames with overlapping players and mis-cropped players. To do this, the normal EM algorithm leads to too many local maxima convergences, so we implemented a Short Run EM algorithm. This will run N trials (~50) of the EM algorithm, capping the number of iterations at only few steps (~3). Then, it will take the best of those trials, in terms of log likelihood, and run that using the original EM algorithm until it converges. We then have another wrapper that will call the above procedure X times (~5) and pick the best model in terms of log likelihood out of those. This proved to work very well to classify our 7 classes of data, and it was able to correctly classify almost every player tested on, using a model of average RGB channel intensity of a cropped portion of the player image corresponding to just the jersey and shorts. We tested other models: one used 3 additional dimensions corresponding to deviation of color intensity in each channel, another used 1 model to represent average deviation of intensity across all channels, and another used histogram-equalized frames to try to normalize the color spectrum. However, none of these models worked as well, so we simply scrapped them and did not include it as part of the report.

### Player Movement Analysis Using Optical Flow

After classifying each player on the field as being part of a certain team, we then performed a point-wise optical flow analysis on the positioning of each player to calculate their velocities. This is also located in **EM Classification Files**. To do this, we used a Sobel-Kernel gradient to determine average X and Y movement derivatives. For actually detecting player movement, the tricky part is that the camera is moving, many times in a direction away from the player, so we cannot simply use optical flow or else the movement of the camera across the field will be a large confounder on player movement. Therefore, to get around this, we took measurements of the optical flow at multiple (~10-20) data points contained on each player, and then took more data points (>20) right outside of the player's bounding box and compared the average flow in each region. Since the surrounding region of the player would capture the movement of only the camera, and the region of the player would capture the movement of both the player and the camera, we were able to simply subtract these estimates to get a rough vector of movement for the player. This data was then sent back to the Neural Network and overlaid on the output frame to show the velocity vector for each classified player.

### Results

Overall, the EM/NN combo algorithm worked very well, and we were able to class players for the 6 types very consistently in real time, or at least in slow motion (when running on a Nvidia GTX 970m). 

 