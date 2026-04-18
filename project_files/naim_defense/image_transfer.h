
// image types
// RGB_IMAGE   is 3 bytes per pixel
// GREY_IMAGE  is 1 bytes per pixel
// LABEL_IMAGE is 2 bytes per pixel
#define RGB_IMAGE 1
#define GREY_IMAGE 2
#define LABEL_IMAGE 3 

// define a variable type that holds a single byte of data
typedef unsigned char ibyte;
typedef unsigned short int i2byte;
typedef unsigned long int i4byte;

// define a structure that stores an image
typedef struct {
	int type;   // image type
	i2byte height; // image height
	i2byte width;  // image width
	ibyte *pdata; // pointer to image data
	i2byte nlabels;  // number of labels (used for LABEL_IMAGE type)
} image;

// size of shared memory block
// 170 MB (10,000 x 5,650 pixel -- RGB image)
const unsigned long int SMAX = 170000000; 

// NODE_A = image transfer lib program
// NODE_B = image_view program
const i2byte NODE_A = 1, NODE_B = 2;

///////

int activate_vision();

int activate_camera(int cam_number, int height, int width);

int acquire_image(image &a, int cam_number);

int deactivate_vision();

int stop_camera(int cam_number);

int start_camera(int cam_number);

///////

int view_rgb_image(image &a, int mode = 0);

// try the different modes below to see what gives the best performance
// if you find the image_view.exe display is delayed or jerks

// mode = 0 (default -- ie when no argument is given)
// - view_rgb_image() waits until image_view.exe is ready for a new image
// - this tends to sync fps of the program with fps of image_view.exe
// resulting in a more smooth display in some situations
// - in other situations it might be less smooth with more delays

// mode = 1
// - view_rgb_image() doesn't wait until image_view.exe is ready
// - if it's not ready, the functions just returns -- you can try 
// again later (eg the next iteration in a while loop)
// - the function doesn't stop the program waiting for image_view.exe, 
// so the performance might be better in some cases with more smooth output

// mode = 2
// - similar to mode 0 but it doesn't use Sleep when waiting, 
// which could improve performance in some situations

int save_rgb_image(char *file_name, image &a);

int load_rgb_image(char *file_name, image &a);

int set_rgb_image(char *file_name, image &a);

int allocate_image(image &a);

int free_image(image &a);

// video I/O functions ///////

// restrictions:
// 1) only one input and output file can be opened at a time
// 2) an input video file must currently be opened in order to open and write 
// to an output file

int open_video_input(char *file_name, double &t_duration, int &width, int &height);

int close_video_input();

int open_video_output(char *file_name, 
	int target_data_rate_bps = 100000000);

// 100000000 bps (100,000 kbps) = 12:1 compression 
// for a 1080p, 24 fps video which should give no 
// significant losses -- even for very challenging cases

int close_video_output();

int read_video_input(image &a, double &t_sample);

int position_video_input(double t_position);

int write_video_output(image &a, double t_sample);
