
// WASD keyboard control of an ESP32 + L298N robot over Bluetooth.
// Stripped-down version of
//   lectures_examples_prof/bluetooth_robot_1.1_vision/windows_program/program.cpp
// with vision removed (vision will replace the keyboard reads later for the
// project, the rest of the protocol stays identical).
//
// Outgoing protocol: 9600 baud, [255, th_left, th_right, th_aux] every 30 ms.
// 90 = stop, >90 forward, <90 backward, range 0..180.

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <windows.h>

#include "serial_com.h"
#include "timer.h"

// EDIT THIS to match the OUTGOING Bluetooth COM port for the paired ESP32.
// Find it in Device Manager -> Ports (COM & LPT) after pairing.
static char COM_PORT[] = "COM5";

// macro that checks for a key press, lifted from the prof's example
#define KEY(c) ( GetAsyncKeyState((int)(c)) & (SHORT)0x8000 )

using namespace std;

int main()
{
	HANDLE h1;
	const int NMAX = 64;
	char buffer_out[NMAX];
	int n, dth, speed;
	int th_left, th_right, th_aux;

	unsigned char start_char = 255;
	// start character for message -- don't use this value for anything else
	// note: this represents -1 as a char variable

	// 9600 bps to match the ESP32 BluetoothSerial side
	speed = 0;

	cout << "\nESP32 BT robot - keyboard drive";
	cout << "\nopening " << COM_PORT << " at 9600 baud ...";
	if( open_serial(COM_PORT, h1, speed) ) {
		cout << "\nfailed to open " << COM_PORT;
		cout << "\npress any key to exit"; cin.get();
		return 1;
	}

	cout << "\npress space key to continue";
	while( !KEY(VK_SPACE) ) Sleep(1);

	// send start message to ESP32 (matches prof's wake handshake)
	n = 1;
	buffer_out[0] = 's';
	if( serial_send(buffer_out, n, h1) ) {
		cout << "\nfailed to send wake byte";
		close_serial(h1);
		return 1;
	}
	Sleep(100);

	cout << "\n\ncontrols:";
	cout << "\n  W / S      forward / backward";
	cout << "\n  A / D      turn left / right";
	cout << "\n  SPACE      stop";
	cout << "\n  X          exit";
	cout << "\n";

	// stop (90 deg = neutral)
	th_left = th_right = th_aux = 90;

	// per-tick increment - prof used 1, same idea: input is integrated
	// over time so holding a key ramps speed/turn
	dth = 30;  // bigger step here since L298N is bang-bang with deadband

	while(1) {
		// default each tick to stop, then bump based on keys
		// (this gives crisp "release to stop" feel for a bang-bang drive)
		th_left = 90;
		th_right = 90;

		if( KEY('W') ) {        // forward: both wheels forward
			th_left  = 90 + dth;
			th_right = 90 + dth;
		}
		if( KEY('S') ) {        // backward: both wheels backward
			th_left  = 90 - dth;
			th_right = 90 - dth;
		}
		if( KEY('A') ) {        // turn left: left back, right forward
			th_left  = 90 - dth;
			th_right = 90 + dth;
		}
		if( KEY('D') ) {        // turn right: left forward, right back
			th_left  = 90 + dth;
			th_right = 90 - dth;
		}
		if( KEY(VK_SPACE) ) {   // hard stop
			th_left = th_right = 90;
		}

		// clamp before sending (prof's pattern)
		if( th_left  < 0 )   th_left  = 0;
		if( th_left  > 180 ) th_left  = 180;
		if( th_right < 0 )   th_right = 0;
		if( th_right > 180 ) th_right = 180;

		buffer_out[0] = start_char;
		buffer_out[1] = (unsigned char)th_left;
		buffer_out[2] = (unsigned char)th_right;
		buffer_out[3] = (unsigned char)th_aux;

		n = 4;
		if( serial_send(buffer_out, n, h1) ) {
			cout << "\nserial send error.\npress c key to continue\n";
			while( !KEY('C') ) Sleep(1);
			break;
		}

		// match prof's 30 fps cadence so the ESP32 input buffer doesn't overflow
		Sleep(30);

		if( KEY('X') ) break;
	}

	// stop the robot before closing
	buffer_out[0] = start_char;
	buffer_out[1] = (unsigned char)90;
	buffer_out[2] = (unsigned char)90;
	buffer_out[3] = (unsigned char)90;
	serial_send(buffer_out, 4, h1);
	Sleep(50);

	close_serial(h1);

	cout << "\ndone.\npress any key to exit"; cin.get();
	return 0;
}
