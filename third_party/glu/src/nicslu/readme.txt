NICSLU, Copyright (c) 2011-2013 Tsinghua University. All Rights Reserved.

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.

_____________________________________________________________________________________

Read the following contents to quickly get how to compile and test NICSLU. The process is quite simple.

1. System requirements
	CPU: x86 or x86-64
	OS: Windows (XP SP3 or higher) or GNU Linux
	Compiler: Microsoft Visual Studio (2005 or higher) or gcc

2. How to compile
	Linux user: just type "make" at the top directory.
	Windows user: if you are using VS2012, open "nicslu.sln" in "win_vs2012" folder, and press F7 to compile the whole project. If you are not using VS2012, follow the steps in user guide 6.3.1.

3. How to test
	Linux user: change work directory to "demo", type "./demos" or "./demop <#threads>" (e.g. "./demop 4") to run the sequential or parallel demo program.
	Windows user: run "cmd", change work directory to "demo", type "demos" or "demop <#threads>" (e.g. "demop 4") to run the sequential or parallel demo program.

4. How to use NICSLU in your programs
	Include "nicslu.h" in your codes. When linking, 
	Linux user: add "-L. nicslu.a -lrt -lpthread -lm", assuming nicslu.a is in the current folder.
	Windows user: add "nicslu.lib" to "Additional Dependencies", or add the code "#pragma comment(lib, "nicslu.lib")" to any position of your codes, assuming nicslu.a is in the current folder.

5. Macros (see make.inc)
	SSE2: whether to use hand-optimized SSE2 code.
	NICS_INT64: whether int__t and uint__t are 64-bit or 32-bit. Only used in 64-bit compilation.
	NO_EXTENSION: control the feature of thread binding. If you cannot compile NICSLU successfully, define this macro and try again.
	NO_ATOMIC: please always define this macro, since the related functions are not used in NICSLU.

NOTE: when using gcc, the optimization flag can be -O2, but CANNOT be -O3.
If you want to get more details, refer to the user guide.