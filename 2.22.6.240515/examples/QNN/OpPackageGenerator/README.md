###### qnn-package-generator command line instructions

###### Setup Instructions
1. Set QNN_SDK_ROOT by running `source <QNN_SDK_ROOT>/bin/envsetup.sh`
2. Set HEXAGON_SDK_ROOT by running `source <HEXAGON_SDK_ROOT>/setup_sdk_env.source `
3. Ensure ANDROID_NDK_ROOT contains the path of the Android NDK

###### Execution Instructions
1. Run `qnn-op-package-generator -p <QNN_SDK_ROOT>/examples/QNN/OpPackageGenerator/ExampleOpPackageHtp.xml -o <output_dir>`
   to see a working example

###### Compiling the package
1. At the top level of the generated package, you can run:

    a.) `make all `to generate both hexagon and linux targets

    b.) `make htp_v68` to generate hexagon target only

    c.) `make htp_x86` to generate linux targets only

    d.) `make htp_aarch64` to generate android targets only

**Note: Ensure clang compiler is discoverable in your path, or set X86_CXX in the makefile to make x86 targets.**

