# to-do

## Zero Order
- [x] impl vec7 $(t,x,y,z,vx,vy,vz)$ + $\delta t$ for adaptive time step while in lockstep
- [ ] look into creating a python package using [maturin](https://github.com/PyO3/maturin)
- [ ] build out a test suite so that we can convince ourselves that it is doing what we expect
- [ ] look into developing the environment on the REDHAT whetu server for more compute and higher memory bus capacity (3x80G A100 + 20G Quadro GP100) Driver Version: 575.57.08 CUDA Version: 12.9
## First Order
- [ ] look into cubic interp between circular orbits for GMC plummer potential impl
- [ ] look into lbparticles impl that runs in gpu blocks so tha can wait for all positions in shared memories then processes after blocking step. 
- [ ] look into the TCP throughput scaling and packet loss algorithm for impl on the $\delta t$ variable for each I.C. integration
## Second Order
- [ ] look to impl a higher order rk model (8th or 15th?) (would this reduce the number of "packets" lost due to error restriction)
## Third Order
- [ ] look to scale the I.C. memory to fit into Block memory for possibly faster throughput
