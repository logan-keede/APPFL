with import <nixpkgs> {};
mkShell {
  #nativeBuildInputs = with pkgs.buildPackages; [
    #cudaPackages_12.cudatoolkit
      #python311Packages.torch-bin
   # ];
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
  ];
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  sudo iptables -A INPUT -p tcp --dport 50051 -j ACCEPT
  '';
}
    
  #export LD_LIBRARY_PATH=${pkgs.cudaPackages_12_4.cudatoolkit}/lib:$LD_LIBRARY_PATH
  #CUDA_PATH=${pkgs.cudaPackages_12.cudatoolkit}
  