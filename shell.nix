{ pkgs ? import <nixpkgs> { }
}:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    (python3.withPackages (pyPkgs: [
     pyPkgs.torch 
    pyPkgs.numpy 
         pyPkgs.tqdm 
         pyPkgs.matplotlib 
 
     pyPkgs.gymnasium 
     ]))

  ];
}
