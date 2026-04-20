
#include <cstdio>
#include <iostream>
#include <fstream>

#include <Windows.h>

using namespace std; 

#define KEY(c) ( GetAsyncKeyState((int)(c)) & (SHORT)0x8000 )

#include "image_transfer.h"

// include this header file for computer vision functions
#include "vision.h"

#include "robot.h"

#include "vision_simulation.h"

#include "timer.h"

#include "shared_memory.h"

// global variables ///////////

robot_system *S1;

image rgb_robot, rgb_opponent, rgb_background;
image rgb_obstacle[N_MAX];
image binary1, binary2, rgb;

char *p_shared; // pointer to shared memory

///////////////////////////////

int activate_simulation(double width, double height,
	double x_obs[], double y_obs[], int N_obs,
	char robot_file[], char opponent_file[], char background_file[],
	char obstacle_file[][S_MAX], double D, double Lx, double Ly, 
	double Ax, double Ay, double alpha_max, int n_robot)
{
	int i, k;
	double *pd;	
	int *pi;

	S1 = new robot_system(D,Lx,Ly,Ax,Ay,alpha_max,n_robot);

	if( S1 == NULL ) {
		cout << "\nmemory allocation error in activate_simulation()";
		return 1;
	}
	
	S1->width = width;
	S1->height = height;

	S1->N_obs = N_obs;

	for(i=0;i<S1->N_obs;i++) {
		S1->x_obs[i] = x_obs[i];
		S1->y_obs[i] = y_obs[i];
	}

	// get image size, dynamically allocate images, and load from file
	
	set_rgb_image(robot_file,rgb_robot);
	set_rgb_image(opponent_file,rgb_opponent);
	set_rgb_image(background_file,rgb_background);
	
	for(i=0;i<S1->N_obs;i++) {
		set_rgb_image(obstacle_file[i],rgb_obstacle[i]);
	}

	binary1.type = GREY_IMAGE;
	binary1.width = rgb_background.width;
	binary1.height = rgb_background.height;
	allocate_image(binary1);	
		
	binary2.type = GREY_IMAGE;
	binary2.width = rgb_background.width;
	binary2.height = rgb_background.height;
	allocate_image(binary2);			
		
	set_rgb_image(background_file,rgb);	

	// setup shared memory for 2 player option ///////////
	
	int n_shared = 1000; // size of shared memory block (bytes)
	char name[] = "shared_memory_v"; // name of shared memory block
	
	// create / access shared memory
	p_shared = shared_memory(name,n_shared);
	
	// player 1
	k = 0;
	pi = (int *)(p_shared + k);
	*pi = 0; pi++; // sample
	*pi = 0; pi++; // laser
	pd = (double *)pi;
	*pd = 0.0; pd++; // theta
	*pd = 150; pd++; // x
	*pd = 150; pd++; // y
	*pd = 0.0; pd++; // alpha

	// player 2
	k = 500;
	pi = (int *)(p_shared + k);
	*pi = 0; pi++; // sample
	*pi = 0; pi++; // laser
	pd = (double *)pi;
	*pd = 0.0; pd++; // theta
	*pd = 300; pd++; // x
	*pd = 250; pd++; // y
	*pd = 0.0; pd++; // alpha

	return 0;
}


int wait_for_player()
{
	int k, *pi;	
	
	k = 900;
	pi = (int *)(p_shared + k);
	
	// initialize start flag
	*pi = 0;
	
	cout << "\n\nwaiting for player to join ...\n";
	
	// wait for player to join
	while(1) {
		if( *pi == 1 ) break;
		Sleep(1);
	}
	
	return 0;
}


int join_player()
{
	int k, *pi;	
	
	k = 900;
	pi = (int *)(p_shared + k);
	
	cout << "\n\njoining other player ...\n";	
	
	// indicate player is ready to join
	*pi = 1;
	
	return 0;
}


int deactivate_simulation()
{
	int i;

	// free the image memory before the program completes
	free_image(rgb_robot);
	free_image(rgb_opponent);
	free_image(rgb_background);
	
	for(i=0;i<S1->N_obs;i++) {
		free_image(rgb_obstacle[i]);
	}
	
	free_image(binary1);
	free_image(binary2);	
	free_image(rgb);	
	
	// safe delete
	if( S1 != NULL ) {
		delete S1;
		S1 = NULL;
	} else {
		cout << "\nerror: NULL pointer in deactivate_simulation()";
	}
	
	return 0;
}

// TODO: interpolate inputs if large delays ?
	
int set_inputs(int pw_l, int pw_r, int pw_laser, int laser, 
				double max_speed)
{
	// * note it's the responsibility of the user to call this function
	// or the inputs will stay constant in the simulation
	
	// set inputs for the simulation -- no smoothing / interpolation
	// since actual system doesn't do that

	// need to set max_speed before setting other inputs
	// since inputs depend on that parameter
	S1->P[1]->v_max = max_speed;	

	// set robot inputs
	S1->P[1]->set_inputs(pw_l,pw_r,pw_laser,laser);
	
	// note the opponent input will be set automatically later
	
	return 0;
}


int set_opponent_inputs(int pw_l, int pw_r, int pw_laser, int laser, 
				double max_speed)
{
	// manually set opponent inputs for the simulation
	// -- good for testing your program

	if( S1->N_robot > 1 ) {
		
		// set opponent parameters
		S1->P[2]->v_max = max_speed;	

		// set opponent inputs
		S1->P[2]->set_inputs(pw_l,pw_r,pw_laser,laser);
		
	}
	
	return 0;
}


int acquire_image_sim(image &rgb)
// assume this function is called frequenlty since it performs
// real-time simulation of the robots
{
	int k;
	double x, y, theta, ia, ja, ib, jb;
	static int init = 0;
	static double tc0;
	double tc;
	double dt = 1.0e-4; // simulation time step
	
	// laser start times
	static double t_laser_start = 0.0, t_laser_start_o = 0.0;	
	
	// previous state of laser inputs
	static int laser_previous = 0, laser_previous_o = 0;
	static int laser_fired = 0, laser_fired_o = 0;

	// laser time durations (s)
	double t_laser;
	double laser_duration = 1.0;
	
	double theta_s, x_s, y_s, alpha_s, *pd;	
	int laser_s, sample_s = 0, *pi;

	int blend;

	// TODO: use subpixel rendering -- ie double for (i,j), etc.
	
	// TODO: time stamp result with simulation and or clock time

	// set initial clock time
	if( !init ) {
		tc0 = high_resolution_time();
	}

	// read current clock time
	tc = high_resolution_time() - tc0;

	if( laser_fired ) {
		t_laser = high_resolution_time() - t_laser_start;
		if( t_laser > laser_duration ) {
			laser_fired = 0; // turn off laser when done
		}
	}
	
	if( laser_fired_o ) {
		t_laser = high_resolution_time() - t_laser_start_o;
		if( t_laser > laser_duration ) {
			laser_fired_o = 0; // turn off laser when done
		}
	}

	if( !init ) {
		// don't simulate for the first call -- ie draw ICs
		init = 1;
	} else {
		// real-time simulation of robots
		// -- simulate robots with time step dt until simulation time = clock time
		while( S1->t < tc ) S1->sim_step(dt);
	}

	// read/write shared memory to get/set the state of the 
	// opponent/robot for 2 player mode //////////////////////

	// mode = 1 - two player mode, player #1 (data block #1)
	// mode = 2 - two player mode, player #2 (data block #2)
	
	if( (S1->mode == 1) || (S1->mode == 2) ) {
		
		// read opponent state from shared memory block ////
		
		if( S1->mode == 1 ) {
			k = 500; // byte number for player 2 data block
		} else {
			k = 0; // byte number for player 1 data block
		}
		
		// start of memory block to write robot data	
		pi = (int *)(p_shared + k);
		
		sample_s = *pi; pi++; // sample
		laser_s = *pi; pi++; // laser
		pd = (double *)pi;
		theta_s = *pd; pd++; // theta
		x_s = *pd; pd++; // x
		y_s = *pd; pd++; // y
		alpha_s = *pd; pd++; // alpha			
				
		// set opponent states to read data
		S1->P[2]->x[1] = theta_s;	
		S1->P[2]->x[2] = x_s;
		S1->P[2]->x[3] = y_s;
		S1->P[2]->x[4] = alpha_s;	
		S1->P[2]->laser = laser_s;
				
		// write robot state to shared memory block ////////
		
		if( S1->mode == 1 ) {
			k = 0; // byte number for player 1 data block
		} else {
			k = 500; // byte number for player 2 data block
		}	
		
		// start of memory block to write robot data	
		pi = (int *)(p_shared + k);		
		
		// get robot data to write
		theta_s = S1->P[1]->x[1];	
		x_s     = S1->P[1]->x[2];
		y_s     = S1->P[1]->x[3];
		alpha_s = S1->P[1]->x[4];	
		laser_s = S1->P[1]->laser;	
		
		*pi = sample_s; pi++; // sample
		*pi = laser_s; pi++; // laser
		pd = (double *)pi;
		*pd = theta_s; pd++; // theta
		*pd = x_s; pd++; // x
		*pd = y_s; pd++; // y
		*pd = alpha_s; pd++; // alpha	
		
	}
	
	// construct image from simulation results /////////
	
	// copy background into result
	copy(rgb_background,rgb);

	// add obstacles to image /////////////////
	
	for(k=0;k<S1->N_obs;k++) {

		// obstacle image center point
		ia = 0.5*rgb_obstacle[k].width;
		ja = 0.5*rgb_obstacle[k].height;	

		// get pixel location of obstacle center
		ib = S1->x_obs[k];
		jb = S1->y_obs[k];
	
		theta = 0.0;
		blend = 1;
		
		draw_image(rgb_obstacle[k],theta,ia,ja,rgb_background,
					ib,jb,rgb,blend);

	}

	// add opponent to image ///////////////////
	
	if( S1->N_robot > 1 ) {
		
		theta = S1->P[2]->x[1] - 3.14159/2;
		x 	  = S1->P[2]->x[2] - S1->P[2]->xa;
		y 	  = S1->P[2]->x[3] - S1->P[2]->ya;
	
		// opponent image center point
		ia = 0.5*rgb_opponent.width;
		ja = 0.5*rgb_opponent.height;
	
		// get pixel location of opponent center
		ib = x;
		jb = y;

		blend = 1;
		
		draw_image(rgb_opponent,theta,ia,ja,rgb_background,ib,jb,rgb,blend);
	}
	
	// add robot to image //////////////////////
	
	theta = S1->P[1]->x[1] - 3.14159/2;	
	x 	  = S1->P[1]->x[2] - S1->P[1]->xa;
	y 	  = S1->P[1]->x[3] - S1->P[1]->ya;
	
	// robot image center point
	ia = 0.5*rgb_robot.width;
	ja = 0.5*rgb_robot.height;
	
	// get pixel location of robot center
	ib = x;
	jb = y;		

//	theta = 3.14159/4; // for testing

	blend = 1;

	draw_image(rgb_robot,theta,ia,ja,rgb_background,ib,jb,rgb,blend);

	// for testing /////////////
/*
	save_rgb_image("output7.bmp",rgb);

	cout << "\noutput file complete.\n";
	Sleep(1000);
	exit(0);
*/	

	// draw laser when fired /////////////////////////

	// check if laser was fired (laser went from 0 to 1)
	// -- ie rising edge on laser
	if(	S1->P[1]->laser && !laser_previous ) {
		t_laser_start = high_resolution_time();
		laser_fired = 1;
		S1->t = tc + laser_duration; // freeze simulation
		cout << "\nlaser fired !";
	}
	
	// update previous laser state
	laser_previous = S1->P[1]->laser;
	
	// draw laser if fired
	if( laser_fired ) {
		draw_laser(S1->P[1],rgb);
	}
	
/*
	// this code is for case where laser doesn't freeze simulation
	// turn laser off after duration
	t_laser = S1->t - t_laser_start;
	if( t_laser > laser_duration ) {
		laser_fired = 0;
	}
*/
	
	// fire opponent laser if needed
	if( S1->N_robot > 1 ) {
		
		// check if laser was fired (laser went from 0 to 1)
		// -- ie rising edge on laser
		if(	S1->P[2]->laser && !laser_previous_o ) {
			t_laser_start_o = high_resolution_time();
			laser_fired_o = 1;
			S1->t = tc + laser_duration; // freeze simulation			
			cout << "\nopponent laser fired !";
		}
	
		// update previous laser state
		laser_previous_o = S1->P[2]->laser;
	
		// draw laser if fired
		if( laser_fired_o ) {
			draw_laser(S1->P[2],rgb);
		}
				
	}
	
	return 0;
}


int draw_image(image &a, double theta, double ia, double ja, 
			   image &b, double ib, double jb, image &output,
			   int blend)			   
{
	int i, j, k, R, G, B, width;
	ibyte *pa, *pb, *pout;
	double iL, jL; // local coord
	double ic, jc; // image center coord
	double cos_th, sin_th;
	int i1, j1, i2, j2, sum;
	
	ibyte *pbin1, *pbin2, *p, *prgb;
	ibyte *q1, *q2, *q3, *q4, *q5;
	int width_bin, height_bin;
	
	// variables for bilinear interpolation
	ibyte *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9;
	int B1, G1, R1, B2, G2, R2, B3, G3, R3, B4, G4, R4;	
	double Ba, Ga, Ra, Bb, Gb, Rb, ip, jp;
//	int flag;

	int width_a, height_a, width_b, height_b, width_out, height_out;
	int imin, imax, jmin, jmax;
	double wl_imin, wl_imax, wl_jmin, wl_jmax;
	double wg_imin, wg_imax, wg_jmin, wg_jmax;
	double wl_i[5], wl_j[5], wg_i[5], wg_j[5];

	// TODO: add #define DEBUG which checks bounds

	// TODO: check for image compatibility

	cos_th = cos(theta);
	sin_th = sin(theta);
	
	pa	 = a.pdata;
	pb	 = b.pdata;	
	pout = output.pdata;
	
	pbin1 = binary1.pdata;
	pbin2 = binary2.pdata;	
	
	// TODO: check background and b have same size

	width_a = a.width;
	height_a = a.height;	
	width_b = width_out = output.width;
	height_b = height_out = output.height;

	width_bin = binary1.width;
	height_bin = binary1.height;

	// TODO: check for special case of abs(theta) < 1.0e-7
	// to speedup and reduce roundoff error

	// find arrays that describle the pixels in rotated window of image a

	// image a window in local coord
	wl_imin = 0.0 - ia;
	wl_imax = width_a - 1.0 - ia;
	wl_jmin = 0.0 - ja;
	wl_jmax = height_a - 1.0 - ja;

	wl_i[1] = wl_imin; wl_j[1] = wl_jmin;
	wl_i[2] = wl_imax; wl_j[2] = wl_jmin;
	wl_i[3] = wl_imin; wl_j[3] = wl_jmax;
	wl_i[4] = wl_imax; wl_j[4] = wl_jmax;

	// image a window in global coord -- fully envelopes rotated window
	wg_imin = 1.0e7; wg_imax = -1.0e7;
	wg_jmin = 1.0e7; wg_jmax = -1.0e7;

	for(k=1;k<=4;k++) {
		wg_i[k] = cos_th * wl_i[k] - sin_th * wl_j[k] + ib;
		wg_j[k] = sin_th * wl_i[k] + cos_th * wl_j[k] + jb;
		if( wg_i[k] > wg_imax ) wg_imax = wg_i[k];
		if( wg_i[k] < wg_imin ) wg_imin = wg_i[k];
		if( wg_j[k] > wg_jmax ) wg_jmax = wg_j[k];
		if( wg_j[k] < wg_jmin ) wg_jmin = wg_j[k];
	}

	// convert to int making sure to cover any fraction
	imin = (int)wg_imin;
	imax = (int)(wg_imax + 1 - 1.0e-10);
	jmin = (int)wg_jmin;
	jmax = (int)(wg_jmax + 1 - 1.0e-10);

	// correct if global window is out of bounds
	if( imin < 0 ) imin = 0;
	if( imax > width_out-1 ) imax = width_out-1;
	if( jmin < 0 ) jmin = 0;
	if( jmax > height_out-1 ) jmax = height_out-1;

	// TODO check for imin > width_out - 1, etc. ?
	// -- for loops are protected 

	if( blend ) {
	
//	for(j=0;j<height_bin;j++) { // for testing
//		for(i=0;i<width_bin;i++) {
	
	for(j=jmin;j<=jmax;j++) {		
		for(i=imin;i<=imax;i++) {	
			p1  = pbin1 + j*width_bin + i;		
			*p1 = 0;
			p2  = pbin2 + j*width_bin + i;		
			*p2 = 0;			
		}
	}
	
	} // end if blend

	// calculate each pixel in output image using linear interpolation
	// with respect to original image
	for(j=jmin;j<=jmax;j++) {		
	
		for(i=imin;i<=imax;i++) {
		
			// calculate local coord il, jl
			// pg = R*pl,  pg = [i,j], pl = [il,jl]	
			// pl = inv(R)*pg = Rt*pg
			// R  = [cos(th) -sin(th)]
			//      [sin(th)  cos(th)]
			// Rt = [ cos(th) sin(th)]
			//      [-sin(th) cos(th)]
			
			// center coords
			ic = i - ib;
			jc = j - jb;
		
			// convert global to local coord
			iL =  cos_th*ic + sin_th*jc;
			jL = -sin_th*ic + cos_th*jc;

			i1 = (int)(iL + ia);
			i2 = i1 + 1;
			
			j1 = (int)(jL + ja);
			j2 = j1 + 1;			

			// check if (i1,j1), etc. are within range of image a
			if( (i1 >= 0) && (i1 < width_a-1) && 
				(j1 >= 0) && (j1 < height_a-1) ) {		

			// bilinear interpolation ///////////////////////

			// set neighborhood pointers to interpolation points
			// p3 p4
			// p1 p2
			
			// TODO: remove i,j calcs and use small increments like conv
			p1 = pa + 3*( j1*width_a + i1 ); // (i1,j1)			
			p2 = pa + 3*( j1*width_a + i2 ); // (i2,j1)
			p3 = pa + 3*( j2*width_a + i1 ); // (i1,j2)
			p4 = pa + 3*( j2*width_a + i2 ); // (i2,j2)
			
			B1 = *p1; G1 = *(p1+1); R1 = *(p1+2);
			B2 = *p2; G2 = *(p2+1); R2 = *(p2+2);	
			B3 = *p3; G3 = *(p3+1); R3 = *(p3+2);
			B4 = *p4; G4 = *(p4+1); R4 = *(p4+2);				
			
			sum = B1 + G1 + R1 + B2 + G2 + R2 + B3 + G3 + R3 + B4 + G4 + R4;		
			
			// interpolate if at least one interpolation point is not zero
			if( sum > 0 ) {
			
			// replace black interpolation points with background points		
	
	// TODO: subexpression optimization, etc. to improve efficiency

	// TODO: more accurate expressions for i and j for substitutions ?

//			flag = 0; // for testing
			if ( B1 + G1 + R1 == 0 ) {
				p1 = pb + 3*( j*width_b + i );		
				B1 = *p1; G1 = *(p1+1); R1 = *(p1+2);
//				flag = 1;
			}
			
			if ( B2 + G2 + R2 == 0 ) {
				p2 = pb + 3*( j*width_b + i );	
				B2 = *p2; G2 = *(p2+1); R2 = *(p2+2);	
//				flag = 1;				
			}

			if ( B3 + G3 + R3 == 0 ) {
				p3 = pb + 3*( j*width_b + i );		
				B3 = *p3; G3 = *(p3+1); R3 = *(p3+2);
//				flag = 1;
			}

			if ( B4 + G4 + R4 == 0 ) {
				p4 = pb + 3*( j*width_b + i );
				B4 = *p4; G4 = *(p4+1); R4 = *(p4+2);	
//				flag = 1;			
			}			

/*			
			if( flag ) {
				p1 = pb + 3*( j*width_b + i + 1);	
				*p1 = 0;
				*(p1+1) = 255;
				*(p1+2) = 0;			
			}
*/			
			/////////////////////////////////////////////////////
			
			// find RGB for point a, b using linear interpolation, eg
			// p3 Rb p4
			// p1 Ra p2

			// interpolation point
			// TODO: subexpression optimization
			// TODO: show window function for testing window algo
			// -- also include rotated windows and sampled points

			ip = iL + ia;
			jp = jL + ja;

			Ra = R1 + (R2 - R1)*(ip - i1);
			Rb = R3 + (R4 - R3)*(ip - i1);		
			
			Ga = G1 + (G2 - G1)*(ip - i1);
			Gb = G3 + (G4 - G3)*(ip - i1);	
			
			Ba = B1 + (B2 - B1)*(ip - i1);
			Bb = B3 + (B4 - B3)*(ip - i1);	

			// interpolate from point a to point b
			// p3 Rb p4
			// p1 Ra p2
			
			R = (int)( Ra + (Rb - Ra)*(jp - j1) );
			G = (int)( Ga + (Gb - Ga)*(jp - j1) );
			B = (int)( Ba + (Bb - Ba)*(jp - j1) );
			
			////////////////////////////////////////////////////

			// get background pixel
			p1 = pb + 3*( j*width_b + i );		
			B1 = *p1; G1 = *(p1+1); R1 = *(p1+2);
				
			// draw pixel if not a background pixel
//			if( !( (R == R1) && (G == G1) && (B == B1) ) ) {	
				p1 = pout + 3*( j*width_out + i );
				*p1	    = B;
				*(p1+1) = G;
				*(p1+2) = R;
				
				if( blend ) {
					// mark object pixels in binary image 1
					p2  = pbin1 + j*width_bin + i;		
					*p2 = 255;
				}
//			}
		
			} // end if sum > 0
			
			} // end if in range

		} // end for i
		
	} // end for j

	// skip the remainder if no blending
	if( blend == 0 ) return 0;
	
	// calculate edges based on 3x3 neighbourhood
	// p7 p8 p9
	// p4 p5 p6
	// p1 p2 p3
	
	for(j=jmin+1;j<jmax;j++) {	
	
		for(i=imin+1;i<imax;i++) {		
			
			p5  = pbin1 + j*width_bin + i;

			// if center pixel is not zero
			if( *p5 ) {		
			
				p1 = p5 - width_bin - 1;
				p2 = p5 - width_bin;
				p3 = p5 - width_bin + 1;
				p4 = p5 - 1;
				p6 = p5 + 1;
				p7 = p5 + width_bin - 1;
				p8 = p5 + width_bin;
				p9 = p5 + width_bin + 1;			
			
				// if at least one neighbour pixel is zero
				if( *p1==0 || *p2==0 || *p3==0 || *p4==0 || 
				    *p6==0 || *p7==0 || *p8==0 || *p9==0 ) {
					q1  = pbin2 + j*width_bin + i;	
					*q1 = 255;
				} // end if
				
			} // end if
			
		} // for i

	} // for j

	// for testing purposes ////////
//	copy(binary1,rgb);
//	save_rgb_image("output_binary1.bmp",rgb);

//	dialate(binary2,binary1);

	// dialate binary image2 -> put result in binary image1
	//    p3
	// p4 p1 p2
	// 	  p5
	for(j=jmin+1;j<jmax;j++) {	
	
		for(i=imin+1;i<imax;i++) {	
		
			p1  = pbin2 + j*width_bin + i;
			p2 = p1 + 1;
			p3 = p1 + width_bin;
			p4 = p1 - 1;
			p5 = p1 - width_bin;
			
			if( *p1 || *p2 || *p3 || *p4 || *p5 ) {
				q1  = pbin1 + j*width_bin + i;			
				*q1 = 255;
			} else {
				q1  = pbin1 + j*width_bin + i;			
				*q1 = 0;
			}

		} // for i

	} // for j

	// for testing purposes ////////
//	copy(binary1,rgb);
//	save_rgb_image("output_binary2.bmp",rgb);	

	// output blended output pixels into rgb
	// and then copy them into the output image
	// -- this avoids making the output image also an input
	prgb = rgb.pdata;
	width = 3*width_out;

	for(j=jmin+1;j<jmax;j++) {		
	
		for(i=imin+1;i<imax;i++) {		
		
			q1 = pbin1 + j*width_bin + i;	
			
			if( *q1 ) {
				
				R = G = B = 0;

				// p7 p8 p9
				// p4 p5 p6
				// p1 p2 p3

				p5 = pout + 3*( j*width_out + i );
				
				p1 = p5 - width - 3;
				p2 = p5 - width;
				p3 = p5 - width + 3;
				p4 = p5 - 3;
				p6 = p5 + 3;
				p7 = p5 + width - 3;
				p8 = p5 + width;
				p9 = p5 + width + 3;

				B += *p1;
				G += *(p1+1);
				R += *(p1+2);	
				
				B += 2*(*p2);
				G += 2*(*(p2+1));
				R += 2*(*(p2+2));					
				
				B += *p3;
				G += *(p3+1);
				R += *(p3+2);					

				B += 2*(*p4);
				G += 2*(*(p4+1));
				R += 2*(*(p4+2));	
				
				B += 4*(*p5);
				G += 4*(*(p5+1));
				R += 4*(*(p5+2));					
				
				B += 2*(*p6);
				G += 2*(*(p6+1));
				R += 2*(*(p6+2));
	
				B += *p7;
				G += *(p7+1);
				R += *(p7+2);	
				
				B += 2*(*p8);
				G += 2*(*(p8+1));
				R += 2*(*(p8+2));					

				B += *p9;
				G += *(p9+1);
				R += *(p9+2);

				// put blended edge pixels in rgb
				
				p = prgb + 3*( j*width_out + i );
				
				*p     = B / 16;
				*(p+1) = G / 16;
				*(p+2) = R / 16;
				
			} // if( *q1 )
			
		} // for i
		
	} // for j

	// copy rgb edge pixels into output image
	for(j=jmin+1;j<jmax;j++) {
		for(i=imin+1;i<imax;i++) {
			q1 = pbin1 + j*width_bin + i;
			if( *q1 ) {
				p1 = prgb + 3*( j*width_out + i );
				p2 = pout + 3*( j*width_out + i );	
				*p2	    = *p1;     // B
				*(p2+1) = *(p1+1); // G
				*(p2+2) = *(p1+2); // R
			}
		}
	}

/*	
	// for testing purposes ////////
	save_rgb_image("output.bmp",output);
	
	cout << "\ntesting images saved";
	cout << "\npress enter to continue ...";	
	getchar();
*/
	
	return 0;
}


// TODO: should check get_image for simple cases
// - low constant velocity motion
// - up and down simple case of integral pixel -- don't use transform
// - program a compare function for two images -- good for testing, etc.

int get_image(image &a, image &b, double ib, double jb, double theta)
{
	int i, j, R, G, B;
	ibyte *pa, *pb;
	double ic, jc;
	double cos_th, sin_th;
	int i1, j1;
	
	// variables for bilinear interpolation
	ibyte *p1, *p2, *p3, *p4;
	int B1, G1, R1, B2, G2, R2, B3, G3, R3, B4, G4, R4;	
	double Ba, Ga, Ra, Bb, Gb, Rb;
	int width_a, height_a, width_b, height_b;
	double ig, jg;

	// TODO: add #define DEBUG which checks bounds

	// TODO: check for image compatibility

	cos_th = cos(theta);
	sin_th = sin(theta);
	
	pa	 = a.pdata;
	pb	 = b.pdata;

	width_a  = a.width;
	height_a = a.height;	
	width_b  = b.width;
	height_b = b.height;

	// TODO: check for special case of abs(theta) < 1.0e-7
	// to speedup and reduce roundoff error

	// calculate each pixel in output image using linear interpolation
	// with respect to original image a

	double ic0, jc0, di, dj;

	ic0 = 0.5*(width_b-1);
	jc0 = 0.5*(height_b-1);

	for(j=0;j<height_b;j++) {		
	
		for(i=0;i<width_b;i++) {
		
			// calculate global coord ig, jg
			// pg = T + R*pl,  pg = [ig,jg], pl = [il,jl]	
			// R  = [cos(th) -sin(th)]
			//      [sin(th)  cos(th)]
		
			// image center coords
			// TODO: check -1 term in draw_image
			ic = i - ic0;
			jc = j - jc0;

			// convert from local coord (image b) to global coord (image a)		
			ig =  ib + cos_th*ic - sin_th*jc;
			jg =  jb + sin_th*ic + cos_th*jc;

			i1 = (int)ig;
			j1 = (int)jg;		

			// check if (i1,j1), etc. are within range of image a
			if( (i1 > 1) && (i1 < width_a-3) && 
				(j1 > 1) && (j1 < height_a-3) ) {		

				// bilinear interpolation ///////////////////////

				// set neighborhood pointers to interpolation points
				// p3 p4
				// p1 p2
				p1 = pa + 3*( j1*width_a + i1 ); // (i1,j1)			
				p2 = p1 + 3;
				p3 = p1 + 3*width_a;
				p4 = p3 + 3;
			
				B1 = *p1; G1 = *(p1+1); R1 = *(p1+2);
				B2 = *p2; G2 = *(p2+1); R2 = *(p2+2);	
				B3 = *p3; G3 = *(p3+1); R3 = *(p3+2);
				B4 = *p4; G4 = *(p4+1); R4 = *(p4+2);				
			
				// TODO: subexpression optimization, etc. to improve efficiency

				// TODO: more accurate expressions for i and j for substitutions ?
			
				// find RGB for point a, b using linear interpolation, eg
				// p3 Rb p4
				// p1 Ra p2

				// interpolation point
				// TODO: subexpression optimization
				// TODO: show window function for testing window algo
				// -- also include rotated windows and sampled points

				di = ig - i1;
				dj = jg - j1;

				Ra = R1 + (R2 - R1)*di;
				Rb = R3 + (R4 - R3)*di;		
			
				Ga = G1 + (G2 - G1)*di;
				Gb = G3 + (G4 - G3)*di;	
			
				Ba = B1 + (B2 - B1)*di;
				Bb = B3 + (B4 - B3)*di;	

				// interpolate from point a to point b
				// p3 Rb p4
				// p1 Ra p2
			
				R = (int)( Ra + (Rb - Ra)*dj );
				G = (int)( Ga + (Gb - Ga)*dj );
				B = (int)( Ba + (Bb - Ba)*dj );
			
				// draw pixel into image b
				p1 = pb + 3*( j*width_b + i );
				*p1	    = (ibyte)B;
				*(p1+1) = (ibyte)G;
				*(p1+2) = (ibyte)R;

			} else { // out of range of image a

				// draw black pixel into image b
				R = G = B = 0;
				p1 = pb + 3*( j*width_b + i );
				*p1	    = (ibyte)B;
				*(p1+1) = (ibyte)G;
				*(p1+2) = (ibyte)R;

			} // end if (i,j) in range of image a

		} // end for i
		
	} // end for j

	return 0;
}


int set_simulation_mode(int mode)
{
	// mode = 0 - single player mode (manual opponent)
	// mode = 1 - two player mode, player #1
	// mode = 2 - two player mode, player #2
	S1->mode = mode;

	return 0;
}

int set_robot_position(double x, double y, double theta)
{
	S1->P[1]->x[1] = theta;	
	S1->P[1]->x[2] = x;
	S1->P[1]->x[3] = y;

	// calculate robot output
	S1->P[1]->calculate_outputs();	

	return 0;
}


int set_opponent_position(double x, double y, double theta)
{
	if( S1->N_robot > 1 ) {
		S1->P[2]->x[1] = theta;	
		S1->P[2]->x[2] = x;
		S1->P[2]->x[3] = y;

		// calculate robot output
		S1->P[2]->calculate_outputs();		
	}
	
	return 0;
}


robot_system::robot_system(double D, double Lx, double Ly, 
	double Ax, double Ay, double alpha_max, int n_robot)
{
	int i;
	double x0, y0, theta0, vmax;
	double max_speed, opponent_max_speed;
	int pw_l, pw_r, pw_laser, laser;
	
	// initial time
	t = 0.0;
	
	N_robot = n_robot;
	
	width  = 640;
	height = 480;

	N_obs = 0;

	// robot max wheel speed (pixels/s)
	max_speed = 100.0;

	// opponent max wheel speed (pixels/s)
	opponent_max_speed = max_speed;

	// array of pointers to robot objects
	for(i=1;i<=N_robot;i++) {
		
		if(i == 1) { // robot
			x0 = 0.3*width;
			y0 = 0.5*height;
			theta0 = 0.0;
			vmax = max_speed;
		} else if (i == 2) { // opponent
			x0 = 0.7*width;
			y0 = 0.5*height;
			theta0 = 3.14159;
			vmax = opponent_max_speed;			
		} else { // default / other robots
			x0 = 0.5*width;
			y0 = 0.5*height;
			theta0 = 3.14159/2;			
			vmax = opponent_max_speed;
		}
		
		P[i] = new robot(x0,y0,theta0,vmax);
		
		if( P[i] == NULL ) {
			cout << "\nmemory allocation error in robot_system()";
			return;
		}
		
	}

	// default actuator values
	pw_l = 1500; // us
	pw_r = 1500; // us
	pw_laser = 1500; // us
	laser = 0; // 0 or 1

	// set robot initial inputs and parameters
	for(i=1;i<=N_robot;i++) {
		P[i]->set_inputs(pw_l,pw_r,pw_laser,laser);
		P[i]->D = 121.0; // distance between wheels (pixels)
		P[i]->Lx = 31.0; // laser / gripper position (pixels)
		P[i]->Ly = 0.0; // laser / gripper position (pixels)
		P[i]->Ax = 37.0; // position of robot center relative to image center
		P[i]->Ay = 0.0; // position of robot center relative to image center		
		P[i]->alpha_max = alpha_max; // max range of laser / gripper (rad)
	}

	// take robot parameters from activate_simulation function
	P[1]->D = D; // distance between wheels (pixels)
	P[1]->Lx = Lx; // laser / gripper position (pixels)
	P[1]->Ly = Ly; // laser / gripper position (pixels)
	P[1]->Ax = Ax; // position of robot center relative to image center
	P[1]->Ay = Ay; // position of robot center relative to image center	
	P[1]->alpha_max = alpha_max; // max range of laser / gripper (rad)

	// calculate robot outputs
	for(i=1;i<=N_robot;i++) {
		P[i]->calculate_outputs();
	}

	mode = 0; // default is single player mode (manual opponent)
		
}


robot_system::~robot_system()
// class destructor
// free dynamic memory, etc.
{
	int i;
	
	// array of pointers to robot objects
	for(i=1;i<=N_robot;i++) {
		// safe delete
		if( P[i] != NULL ) {
			delete P[i];
			P[i] = NULL;
		} else {
			cout << "\nerror: NULL pointer in ~robot_system()";
		}
	}
	
}


void robot_system::sim_step(double dt)
{
	int i;	

	// * assume the inputs have already been set for each robot

	// simulate each robot in the system for one time step	
	for(i=1;i<=N_robot;i++) {
		P[i]->sim_step(dt);
	}
	
	// increment robot_system time
	t += dt;
	
}


int draw_laser(robot *P, image &rgb)
{
	double x0, y0, theta, r, dr;
	int i, j, R, G, B;

	dr = 1;

	// set laser colour
	R = 0;
	G = 255;
	B = 0;

	// get start point of the laser
	x0 = P->xg;
	y0 = P->yg;
	
	// robot theta + laser alpha
	theta = P->x[1] + P->x[4];	
	
	for(r=0;r<5000;r+=dr) {
		
		i = (int)( x0 + r*cos(theta) );
		j = (int)( y0 + r*sin(theta) );
		
		// stop loop when (i,j) goes out of range / off screen
		if( i < 3 ) break;
		if( i > rgb.width-3 ) break;
		if( j < 3 ) break;
		if( j > rgb.height-3 ) break;	
		
		draw_point_rgb_laser(rgb,i,j,R,G,B);
		
	}
	
	return 0;
}


int draw_point_rgb_laser(image &rgb, int ip, int jp, int R, int G, int B)
{
	ibyte *p;
	int i,j,w=1,pixel;

	// initialize pointer
	p  = rgb.pdata;

	if ( rgb.type != RGB_IMAGE ) {
		cout << "\nerror in draw_point_RGB: input type not valid!\n";
		return 1;
	}

	// limit out of range (i,j) values
	// NOTE: this part is important to avoid wild pointers
	if( ip < w ) ip = w;
	if( ip > rgb.width-w-1 ) ip = rgb.width-w-1;
	if( jp < w ) jp = w;
	if( jp > rgb.height-w-1 ) jp = rgb.height-w-1;

	for(i=-w;i<=w;i++) {
		for(j=-w;j<=w;j++) {
			pixel = rgb.width*(jp+j)+(ip+i);
			p[3*pixel]   = B;
			p[3*pixel+1] = G;
			p[3*pixel+2] = R;
		}
	}

	return 0;
}

