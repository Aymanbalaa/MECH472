
// copied from lectures_examples_prof/bluetooth_robot_1.1_vision/windows_program/serial_com.cpp
// no modifications - this is the prof's reference implementation.

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <windows.h>

#include "serial_com.h"

using namespace std;

int open_serial(char *port_name, HANDLE &h, int speed)
{
	DCB param = {0};
	COMMTIMEOUTS CommTimeouts;
	char str[30] = "\\\\.\\";

	// typical Arduino speed settings
	DWORD BR[] = {CBR_9600, CBR_115200, 250000, 1000000, 2000000,
		5000000, 12000000, 480000000 };

	cout << "\nInitializing serial port and resetting Arduino.\n\nPlease wait ...";

	// need the following form for COM > 9, "\\\\.\\COM10"
	// this form also works for COM < 10
	strcat(str,port_name);

    h = CreateFile(str, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
						FILE_ATTRIBUTE_NORMAL, NULL);

    if( h == INVALID_HANDLE_VALUE ) {
       cout << "\nerror opening serial port";
	   return 1; // error
	}

	// get current serial port parameters
	if( !GetCommState(h,&param) ) {
		cout << "\nerror: could not get serial port parameters";
		return 1;
	}

	param.StopBits    = ONESTOPBIT;
	param.Parity      = NOPARITY;
	param.ByteSize    = 8;

	if( speed < 0 || speed > 8 ) {
		cout << "\nerror: speed out of range [0 to 8]";
		return 1;
	}

	param.BaudRate = BR[speed];

	param.fDtrControl = DTR_CONTROL_ENABLE; // reset arduino with serial connection

	if( !SetCommState(h,&param) ) {
		cout << "\nerror: serial port parameters could not be set";
		return 1; // error
	}

	// set serial port time-out parameters for a total timeout of 10s
	CommTimeouts.ReadIntervalTimeout = 0; // not used
	CommTimeouts.ReadTotalTimeoutMultiplier = 0; // total time-out per byte
	CommTimeouts.ReadTotalTimeoutConstant = 10000; // add constant to get total time-out
	CommTimeouts.WriteTotalTimeoutMultiplier = 0; // not used
	CommTimeouts.WriteTotalTimeoutConstant = 0; // not used

	if( !SetCommTimeouts(h,&CommTimeouts) ) {
		cout << "\nerror: serial port time-out parameters could not be set";
		return 1; // error
	}

	PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR);
	Sleep(100);

	PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR);
	Sleep(3000); // wait for Arduino to fully reset

	cout << "\n\nSerial port and Arduino are ready.\n";

	return 0; // OK
}


int close_serial(HANDLE &h)
{
	if( h == INVALID_HANDLE_VALUE ) {
		cout << "\nerror closing serial port";
		return 1; // error
	}

	CloseHandle(h);
	h = INVALID_HANDLE_VALUE;

	return 0; // OK
}


int serial_recv(char *buffer, int n, HANDLE h)
{
    DWORD nrecv;

	if( !ReadFile(h, (LPVOID)buffer, (DWORD)n, &nrecv, NULL) ) {
		cout << "\nerror: serial port read error";
		return 1; // error
	}

	if( nrecv != n ) {
		cout << "\nerror: serial port read time-out";
		return 1; // error
	}

	return 0; // OK
}


int serial_send(char *buffer, int n, HANDLE h)
{
    DWORD nsent;

    if( !WriteFile(h, (LPVOID)buffer, (DWORD)n, &nsent, NULL) ) {
		cout << "\nserial port write error";
		return 1; // error
	}

	return 0; // OK
}


int serial_available(HANDLE h)
{
	COMSTAT status;
	DWORD errors;
	int nq;

    ClearCommError(h, &errors, &status);

	nq = status.cbInQue;

	return nq;
}
