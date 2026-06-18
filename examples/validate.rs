//! Print verified-enclosure validation: MC containment + tightness vs order.
//! (The same checks are enforced as a test in tests/containment.rs.)
use tmflow::integrators::TaylorModelIntegrator;
use tmflow::prelude::*;
use tmflow::system::System;

struct Lcg(u64);
impl Lcg{fn new(s:u64)->Self{Lcg(s.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))}
fn u(&mut self,lo:f64,hi:f64)->f64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);lo+(hi-lo)*((self.0>>11)as f64/(1u64<<53)as f64)}}

fn rk4(sys:&VanDerPol,mut x:[f64;2],h:f64,sub:usize)->[f64;2]{let dt=h/sub as f64;for _ in 0..sub{
    let k1=sys.eval(&x);let x2=[x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1]];let k2=sys.eval(&x2);
    let x3=[x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1]];let k3=sys.eval(&x3);
    let x4=[x[0]+dt*k3[0],x[1]+dt*k3[1]];let k4=sys.eval(&x4);
    x=[x[0]+dt/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0]),x[1]+dt/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])];}x}

fn main(){
    let sys=VanDerPol::new(1.0);
    let (center,half,h,n)=([1.4,0.0],[0.08,0.08],0.1,20usize);
    let n_mc=2000;
    println!("Van der Pol verified TM integrator (tmflow), {n} steps h={h}\n");
    for k in [2u32,3,4,5]{
        let tm=TaylorModelIntegrator::new(&sys,k);
        let traj=propagate(&tm,center,half,h,n);
        let mut rng=Lcg::new(0);
        let (mut mtm,mut mbb)=(vec![0usize;n+1],vec![0usize;n+1]);
        for _ in 0..n_mc{
            let s=[rng.u(-1.0,1.0),rng.u(-1.0,1.0)];
            let mut x=[center[0]+half[0]*s[0],center[1]+half[1]*s[1]];
            for st in 0..=n{ if st>0{x=rk4(&sys,x,h,10);}
                let bb=&traj.boxes[st];
                if bb[0].0<=x[0]&&x[0]<=bb[0].1&&bb[1].0<=x[1]&&x[1]<=bb[1].1{mbb[st]+=1;}
                let vx=traj.states[st][0].eval_at(&s);let vy=traj.states[st][1].eval_at(&s);
                if vx.lo<=x[0]&&x[0]<=vx.hi&&vy.lo<=x[1]&&x[1]<=vy.hi{mtm[st]+=1;}}
        }
        let mintm=mtm.iter().map(|&c|100.0*c as f64/n_mc as f64).fold(f64::INFINITY,f64::min);
        let minbb=mbb.iter().map(|&c|100.0*c as f64/n_mc as f64).fold(f64::INFINITY,f64::min);
        let area=*traj.measures.last().unwrap();
        let rx=traj.states[n][0].rem.width(); let ry=traj.states[n][1].rem.width();
        println!("k={k}:  min MC  TM={mintm:.2}%  bbox={minbb:.2}%   final area={area:.4e}  rem=({rx:.2e},{ry:.2e})");
    }
}
