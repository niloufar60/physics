/**************************************************************************************************
* PROGRAM  MULTIMODE PFC
* By: Nilou on July 2015
* Last updated: Tue Mar 22 09:26:52 EDT 2016
****************************************************************************************************
* Compilation on saw sharcnet:
  mpicc -I/opt/sharcnet/fftw/3.3.4/intel/include -L/opt/sharcnet/fftw/3.3.4/intel/lib -lfftw3_mpi -lfftw3 -I/opt/sharcnet/gsl/1.15/intel/include -L/opt/sharcnet/gsl/1.15/intel/lib -lgsl -lgslcblas -o file.out ./file.c
* Modules to load on saw sharcnet: module load gsl/intel/1.15 && module load fftw/intel/3.3.4
* Run: mpirun -np 2 ./file.out >& file.log
****************************************************************************************************/

#include </cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/MPI/intel2017.1/openmpi2.1/fftw-mpi/3.3.6/include/fftw3-mpi.h>
#include </cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/gcc4.8/gsl/2.2.1/include/gsl/gsl_math.h>
#include </cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/gcc4.8/gsl/2.2.1/include/gsl/gsl_rng.h>
#include <//cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/gcc4.8/gsl/2.2.1/include/gsl/gsl_randist.h>
#include <complex.h>
#include <time.h>
#include <string.h>


/* GLOBAL VARIABLES */
ptrdiff_t Lx,Ly;
int myid,np,idx;

/*************************************************************************************/
/* STARTING THE MAIN PROGRAM */

int main (int argc, char **argv)
{

/*************************************************************************************/
/* LOCAL VARIABLES */

  double facx,facy,k1f,k1f_e,dx,dy,dt,kx,ky;
  double k1f_m,k1f_m_e;
  double r,w,q0,q00,q1,q2,psi0,psiml,psim,mean_psi;
  double lambda,b0,b1,b2,alpha,alpha1,alpha2;
  double epsilon1,epsilon2,epsilon3;
  double Sf,tpi,enll,engg,gmx_sq;
  char run[100], filename[BUFSIZ];
  int ig,n,nend,nout,ix,l,pc;
    int c_new;
  double t1,t2;
    double theta,beta;
    

  double w0,r_c,gammaa,mean_mx,mean_my,mean_m;  /* magnetic corr. length, Curie T */

  double Bx_ext,By_ext,mu0;

  fftw_plan plan, planb; //, plang1, plang2;

  double *k1,*k1n,*ksq,*eng_arr;
  double *op;
 
  double *k1_m,*k1n_m;

  double *psi, *psin, *psiw, *in, *out;
  double *arr_t;
  double *psi_e,*mx_e,*my_e;
  double *nu1, *nu2;
    
  double *psib,*nb;

  double *opsi,*phi,*ophi;

  double *mx,*my,*msq,*m;
  double *mxn,*myn;
  
  double *gxmx,*gymx,*gxmy,*gymy;

  double *lap_az, *az, *Bx_ind, *By_ind;
  double *Bsq_ind,*B_ind;
  
  fftw_complex *psi_k, *psin_k, *in_k, *out_k;
  fftw_complex *psi_k_e, *mx_k_e, *my_k_e;
  fftw_complex *nu1_k, *nu2_k;
  fftw_complex *k_vec,*sfr_k,*k_vec1;

  fftw_complex *opsi_k,*phi_k,*ophi_k;

  fftw_complex *mx_k,*my_k;
  fftw_complex *mxn_k,*myn_k;

  fftw_complex *gxmx_k,*gymx_k,*gxmy_k,*gymy_k;

  fftw_complex *lap_az_k, *az_k, *Bx_k, *By_k;

  ptrdiff_t alloc_local, Lxl, local_0_start, i, j;

double *gpx,*gpy,*mgp,*div,*div3,*div5;

  const gsl_rng_type * T;
  gsl_rng * rnd;
  double sigma;
  unsigned long int s;

  double enl[10]; /* local energy terms */ 
  double eng[10]; 

  MPI_Init(&argc, &argv);
  fftw_mpi_init();

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);  



/*************************************************************************************/
/* READ THE INPUT FILE FOR PARAMS */

  FILE *inp;
  inp = fopen ("multimod-mag.in","r");
  fscanf(inp,"%s",run);
  fscanf(inp,"%td %td",&Lx,&Ly);
  fscanf(inp,"%lf %lf %lf",&dx,&dy,&dt);
  fscanf(inp,"%d %d",&nend,&nout);
  fscanf(inp,"%lf %lf %lf %lf %lf %lf",&r,&w,&psi0,&q0,&q1,&q2);
  fscanf(inp,"%lf %lf %lf %lf",&lambda,&b0,&b1,&b2);
  fscanf(inp,"%lf %lf %lf",&sigma,&theta,&beta);
  fscanf(inp,"%lf %lf %lf %lf %lf",&w0,&r_c,&gammaa,&alpha1,&alpha2);
  fscanf(inp,"%lf %lf %lf",&epsilon1,&epsilon2,&epsilon3);
  fscanf(inp,"%lf %lf %lf",&Bx_ext,&By_ext,&mu0);

/*************************************************************************************/
/* ALLOCATE MEMORY FOR ARRAYS */

    double psi_ij[Lx][Ly];
    double mx_ij[Lx][Ly];
    double my_ij[Lx][Ly];
    double gpx_ij[Lx][Ly];
    double gpy_ij[Lx][Ly];
    double mgp_ij[Lx][Ly];
    double mgp3_ij[Lx][Ly];
    double mgp5_ij[Lx][Ly];
    double mmgpx[Lx][Ly];
    double mmgp3x[Lx][Ly];
    double mmgp5x[Lx][Ly];
    double mmgpy[Lx][Ly];
    double mmgp3y[Lx][Ly];
    double mmgp5y[Lx][Ly];
    double gmmgpx[Lx][Ly];
    double gmmgp3x[Lx][Ly];
    double gmmgp5x[Lx][Ly];
    double gmmgpy[Lx][Ly];
    double gmmgp3y[Lx][Ly];
    double gmmgp5y[Lx][Ly];
    double div_ij[Lx][Ly];
    double div3_ij[Lx][Ly];
    double div5_ij[Lx][Ly];
    
    double psib_ij[Lx][Ly];
    double n_ij[Lx][Ly];
    
    
  alloc_local = fftw_mpi_local_size_2d (Lx, Ly/2+1, MPI_COMM_WORLD,
                                        &Lxl, &local_0_start);

  psi     = fftw_alloc_real(2 * alloc_local);
  psin    = fftw_alloc_real(2 * alloc_local); 
  psiw    = fftw_alloc_real(2 * alloc_local);

  arr_t   = fftw_alloc_real(2 * alloc_local);

  psib   =  fftw_alloc_real(2 * alloc_local);
    nb   =  fftw_alloc_real(2 * alloc_local);
    
    
  opsi    = fftw_alloc_real(2 * alloc_local);
  phi     = fftw_alloc_real(2 * alloc_local);
  ophi    = fftw_alloc_real(2 * alloc_local);

  k1      = fftw_alloc_real(alloc_local);
  k1n     = fftw_alloc_real(alloc_local);
  ksq     = fftw_alloc_real(alloc_local);
  op      = fftw_alloc_real(alloc_local);

  eng_arr = fftw_alloc_real(2 * alloc_local);
  psi_e   = fftw_alloc_real(2 * alloc_local);
  mx_e    = fftw_alloc_real(2 * alloc_local);
  my_e    = fftw_alloc_real(2 * alloc_local);

  in      = fftw_alloc_real(2 * alloc_local);
  out     = fftw_alloc_real(2 * alloc_local);
 
  nu1     = fftw_alloc_real(2 * alloc_local);
  nu2     = fftw_alloc_real(2 * alloc_local);

  mx      = fftw_alloc_real(2 * alloc_local);
  my      = fftw_alloc_real(2 * alloc_local);
  mxn     = fftw_alloc_real(2 * alloc_local);
  myn     = fftw_alloc_real(2 * alloc_local);  
  msq     = fftw_alloc_real(2 * alloc_local);
  m       = fftw_alloc_real(2 * alloc_local);

  gxmx    = fftw_alloc_real(2 * alloc_local);
  gymx    = fftw_alloc_real(2 * alloc_local);
  gxmy    = fftw_alloc_real(2 * alloc_local);
  gymy    = fftw_alloc_real(2 * alloc_local);

  k1_m   = fftw_alloc_real(2 * alloc_local);
  k1n_m  = fftw_alloc_real(2 * alloc_local);
    
  az     = fftw_alloc_real(2 * alloc_local);
  lap_az = fftw_alloc_real(2 * alloc_local);
    
  Bx_ind = fftw_alloc_real(2 * alloc_local);
  By_ind = fftw_alloc_real(2 * alloc_local);
  Bsq_ind = fftw_alloc_real(2 * alloc_local);
  B_ind   = fftw_alloc_real(2 * alloc_local);

  psi_k   = fftw_alloc_complex(alloc_local);
  psin_k  = fftw_alloc_complex(alloc_local);
  psi_k_e = fftw_alloc_complex(alloc_local);

  opsi_k  = fftw_alloc_complex(alloc_local);
  phi_k   = fftw_alloc_complex(alloc_local);
  ophi_k  = fftw_alloc_complex(alloc_local);

  in_k = fftw_alloc_complex(alloc_local);  
  out_k = fftw_alloc_complex(alloc_local);

  nu1_k  = fftw_alloc_complex(alloc_local);
  nu2_k  = fftw_alloc_complex(alloc_local);

  mx_k   = fftw_alloc_complex(alloc_local);
  my_k   = fftw_alloc_complex(alloc_local);
  mxn_k  = fftw_alloc_complex(alloc_local);
  myn_k  = fftw_alloc_complex(alloc_local);

  mx_k_e   = fftw_alloc_complex(alloc_local);
  my_k_e   = fftw_alloc_complex(alloc_local);

//  gxmx_k  = fftw_alloc_complex(alloc_local);
///  gymx_k  = fftw_alloc_complex(alloc_local);
//  gxmy_k  = fftw_alloc_complex(alloc_local);
//  gymy_k  = fftw_alloc_complex(alloc_local);

  k_vec  = fftw_alloc_complex(alloc_local);
  k_vec1 = fftw_alloc_complex(alloc_local);
  sfr_k  = fftw_alloc_complex(alloc_local);
    
  az_k     = fftw_alloc_complex(alloc_local);
  lap_az_k = fftw_alloc_complex(alloc_local);
   
  Bx_k    = fftw_alloc_complex(alloc_local);
  By_k    = fftw_alloc_complex(alloc_local);

  gpx    = fftw_alloc_real(2 * alloc_local);
  gpy    = fftw_alloc_real(2 * alloc_local);
  mgp    = fftw_alloc_real(2 * alloc_local);
  div    = fftw_alloc_real(2 * alloc_local);
  
    div3   = fftw_alloc_real(2 * alloc_local);
    div5   = fftw_alloc_real(2 * alloc_local);



  plan = fftw_mpi_plan_dft_r2c_2d(Lx, Ly, in, out_k, MPI_COMM_WORLD,
                                   FFTW_MEASURE);

  planb = fftw_mpi_plan_dft_c2r_2d(Lx, Ly, in_k, out, MPI_COMM_WORLD,
                                    FFTW_MEASURE);

//  plang2 = fftw_mpi_plan_dft_r2c_2d(Lx, Ly, nu2, nu2_k, MPI_COMM_WORLD,
//                                   FFTW_MEASURE);

/*************************************************************************************/
/* FUNCTION DECLARATIONS */

  void printout_arr (char* fname_arr, double* arr, int Lxl);
  void printout_vec_arr (char* fname_vec_arr, double* arr_x, double* arr_y, int Lxl);
  void printout_num (char* fname_n, int Lxl, double n1, double n2, double n3, double n4, double n5, double n6);
  void mean(double* arr, int Lxl, double* mean_func);
  void kx_ky(double Lxl, fftw_complex* k_vec);
  void printout_arr_k (char* fname_arr_k, fftw_complex* arr_k, int Lxl);
  void dump_out_ij(char* fname_arr, double arr[Lx][Ly]);
  void gradx(double p1[Lx][Ly], double gp[Lx][Ly]);
  void grady(double p1[Lx][Ly], double gp[Lx][Ly]);

/*************************************************************************************/
/* LINEAR TERM */

//   dy=sqrt(3.0)*Lx/Ly*dx;
   /* k-space loop */
   tpi=2.*acos(-1.); facx=tpi/(dx*Lx); facy=tpi/(dy*Ly); Sf=1./(Lx*Ly);
   kx_ky(Lxl, k_vec);
   for(i=0;i<Lxl;++i){
     for(j=0;j<Ly/2+1;++j){ix=i*(Ly/2+1)+j;
       kx=k_vec[ix][0]*facx;
       ky=k_vec[ix][1]*facy;    
       ksq[ix]=kx*kx+ky*ky;
//       k1f=-ksq[ix]*(r+lambda*(((q0*q0-ksq[ix])*(q0*q0-ksq[ix])+b0)*((q1*q1-ksq[ix])*(q1*q1-ksq[ix])+b1)
//           +((q2*q2-ksq[ix])*(q2*q2-ksq[ix])+b2)));
//       k1f=beta*(-ksq[ix]*(r+((q0*q0-ksq[ix])*(q0*q0-ksq[ix])+b0)*((q1*q1-ksq[ix])*(q1*q1-ksq[ix])+b1))); /* 2-mode */
       k1f=beta*(-ksq[ix]*(r+(q0*q0-ksq[ix])*(q0*q0-ksq[ix]))); /* hexa */
       k1[ix]=exp(k1f*dt)*Sf;
       k1n[ix]=((exp(k1f*dt)-1.0)/k1f)*Sf;
       if(k1f==0) {k1n[ix]=-dt*Sf;}
 
       k1f_m=(w0*w0*2.*ksq[ix]-2.*ksq[ix]*ksq[ix]-r_c);
       k1_m[ix]=exp(k1f_m*dt)*Sf;
       k1n_m[ix]=((exp(k1f_m*dt)-1.0)/k1f_m)*Sf;
       if(k1f_m==0) {k1n_m[ix]=-dt*Sf;}

       op[ix]=((q0*q0-ksq[ix])*(q0*q0-ksq[ix])+b0)*((q1*q1-ksq[ix])*(q1*q1-ksq[ix])+b1)*Sf; /* 2-mode */
//       op[ix]=(q0*q0-ksq[ix])*(q0*q0-ksq[ix])*Sf; /* hexa */

     }
   }

/*************************************************************************************/
/* INITIAL CONDITION */

  /* Gaussian noise stuff */ 
  srand(time(NULL)+myid);

  gsl_rng_env_setup();
  T = gsl_rng_default;
  rnd = gsl_rng_alloc (T);
  srand(time(NULL)+myid);
  s=rand();
  gsl_rng_set(rnd, s);


  FILE *fread;

/*
  fread=fopen("density0","r");
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
  MPI_Barrier(MPI_COMM_WORLD);
   if (myid==pc) {
   MPI_Barrier(MPI_COMM_WORLD);
    for (i=0;i<Lxl;++i) {
      for (j=0;j<Ly;++j) {
//        for (i=0;i<Lxl;++i) {
         ix=i*2*(Ly/2+1)+j;
//         fscanf(fread,"%lf",&arr_t[ix]);
//         if (arr_t[ix]!=0) {psi[ix]=arr_t[ix];}
//         if((i<42) && (j<72)) {psi[ix]=arr_t[ix];}
         fscanf(fread,"%lf",&psi[ix]);
//         if (j>Ly/2) {psi[ix]=0;}
//         printf("%d %lf \n",ix,psi[ix]);
        }
      }
    }
  }


  fread=fopen("densitys","r");
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
  MPI_Barrier(MPI_COMM_WORLD);
   if (myid==pc) {
   MPI_Barrier(MPI_COMM_WORLD);
    for (i=0;i<Lxl;++i) {
      for (j=0;j<Ly;++j) {
//        for (i=0;i<Lxl;++i) {
         ix=i*2*(Ly/2+1)+j;
         fscanf(fread,"%lf",&arr_t[ix]);
         if ((j<Ly/4)||(j>3*Ly/4)) {psi[ix]=arr_t[ix];}
        }
      }
    }
  }


  fread=fopen("mx0","r");
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
  MPI_Barrier(MPI_COMM_WORLD);
   if (myid==pc) {
   MPI_Barrier(MPI_COMM_WORLD);
    for (i=0;i<Lxl;++i) {
      for (j=0;j<Ly;++j) {
//        for (i=0;i<Lxl;++i) {
         ix=i*2*(Ly/2+1)+j;
//         fscanf(fread,"%lf",&arr_t[ix]);
//         if((i<42) && (j<72)) {mx[ix]=arr_t[ix];}
         fscanf(fread,"%lf",&mx[ix]);      
        }
      }
    }
  }


  fread=fopen("mxs","r");
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
  MPI_Barrier(MPI_COMM_WORLD);
   if (myid==pc) {
   MPI_Barrier(MPI_COMM_WORLD);
    for (i=0;i<Lxl;++i) {
      for (j=0;j<Ly;++j) {
//        for (i=0;i<Lxl;++i) {
         ix=i*2*(Ly/2+1)+j;
         fscanf(fread,"%lf",&arr_t[ix]);
         if ((j<Ly/4)||(j>3*Ly/4)) {mx[ix]=arr_t[ix];}
        }
      }
    }
  }



  fread=fopen("my0","r");
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
  MPI_Barrier(MPI_COMM_WORLD);
   if (myid==pc) {
   MPI_Barrier(MPI_COMM_WORLD);
    for (i=0;i<Lxl;++i) {
      for (j=0;j<Ly;++j) {
//        for (i=0;i<Lxl;++i) {
         ix=i*2*(Ly/2+1)+j;
//         fscanf(fread,"%lf",&arr_t[ix]);
//         if((i<42) && (j<72)) {my[ix]=arr_t[ix];}
         fscanf(fread,"%lf",&my[ix]);
        }
      }
    }
  } 


  fread=fopen("mys","r");
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
  MPI_Barrier(MPI_COMM_WORLD);
   if (myid==pc) {
   MPI_Barrier(MPI_COMM_WORLD);
    for (i=0;i<Lxl;++i) {
      for (j=0;j<Ly;++j) {
//        for (i=0;i<Lxl;++i) {
         ix=i*2*(Ly/2+1)+j;
         fscanf(fread,"%lf",&arr_t[ix]);
         if ((j<Ly/4)||(j>3*Ly/4)) {my[ix]=arr_t[ix];}
        }
      }
    }
  } 

*/
// q00=2.*tpi*3./(sqrt(3.0)*Lx*dx);

  /* real space loop */
  psiml=0.;
  for (i=0;i<Lxl;++i) {
    for (j=0;j<Ly;++j) {
       ix=i*2*(Ly/2+1)+j;
//       psi[ix] = psi0;
//       if ( (i-Lxl/2.)*(i-Lxl/2.)+(j-Ly/2.)*(j-Ly/2.) < 80.*80. ) {
//         psi[ix]=.1 * (.5-(rand() & 1000)/1000.)+ psi0;
         psi[ix]= 2.*0.1*(cos(i*dx)+cos(j*dy))+4.*0.1*(cos(i*dx)*cos(j*dy))+ psi0;
//         psi[ix]= 0.1*(cos(i*dx)*cos(j*dy/sqrt(3.))-(0.5)*cos(2.*j*dy/sqrt(3.)))+psi0;
//        psi[ix]= 2.*0.1*(cos((5.*tpi/(Lx*dx))*(i)*dx)+cos((5.*tpi/(Ly*dy))*(j)*dy)) + psi0;
//      psi[ix]=psi0+cos(-sqrt(3.)/2.*q00*i*dx-q00/2.*j*dy)+cos(q00*j*dy)+cos(sqrt(3.)/2.*q00*i*dx-q00/2.*j*dy);


        mx[ix]= .05 * (.5-(rand() & 1000)/1000.);//0.1;//.01 * cos(theta);  //0.03;//.05 * (.5-(rand() & 1000)/1000.);
        my[ix]= .05 * (.5-(rand() & 1000)/1000.);//0;//.01 * sin(theta);  //0.03;//.05 * (.5-(rand() & 1000)/1000.);


         Bx_ind[ix] = 0;
         By_ind[ix] = 0;
         B_ind[ix]  = 0;
        
         msq[ix] = mx[ix]*mx[ix]+my[ix]*my[ix];
         m[ix]   = sqrt(msq[ix]);
        
        
        psi_ij[i][j]=psi[ix];
        mx_ij[i][j]=mx[ix];
        my_ij[i][j]=my[ix];
        
//       }
        
       /* producing Gaussian random noise */
       nu1[ix]=gsl_ran_gaussian(rnd, sigma);
       nu2[ix]=gsl_ran_gaussian(rnd, sigma);

       psiml+=psi[ix]; 
    }
   }      

    dump_out_ij("density_ij",psi_ij);
    dump_out_ij("mx_ij",mx_ij);
    dump_out_ij("my_ij",my_ij);
    
    
   /* calculate and print the initial density to the console */
   MPI_Reduce(&psiml, &psim, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if (myid==0) {psim=psim/(Lx*Ly); printf("initial mean density=%lf %lf %lf %lf \n", psim,epsilon1,epsilon2,epsilon3);}

   /* initialize the output file index */
   idx=0; 
  

  alpha=alpha1;
 
   /* start the clock */
   t1 = MPI_Wtime();

/*************************************************************************************/
/* STARTING THE TIME LOOP */

   /* time loop */
   for (n=1;n<=nend;++n) {

     /* successive array output to file */
     if (n==1 || n%nout==0) {
       
       idx++;

       printout_arr("density",psi,Lxl);    

       mean(psi, Lxl, &mean_psi);
       if (myid==0) {printf("time, density = %i %lf \n", n, mean_psi); }

       mean(mx, Lxl, &mean_mx);
       mean(my, Lxl, &mean_my);
       mean(m, Lxl, &mean_m);
       printout_num("mean_m.txt", Lxl, mean_mx, mean_my, mean_m, 0, 0, 0);

       printout_arr("mx",mx,Lxl);
       printout_arr("my",my,Lxl);
       printout_arr("m",m,Lxl);
       printout_arr("nb",nb,Lxl);
       printout_arr("psib",psib,Lxl);
         
         
//       printout_arr("Bx_ind",Bx_ind,Lxl);
//       printout_arr("By_ind",By_ind,Lxl); 
//       printout_arr("B_ind",B_ind,Lxl);

       printout_vec_arr("m_vec",mx,my,Lxl);

    } /* if nout */


    if (n>nend/2) {alpha=alpha2;}

 
     for (i=0;i<Lxl;++i) {
       for (j=0;j<Ly;++j) {
         ix=i*2*(Ly/2+1)+j;
         msq[ix] = mx[ix]*mx[ix]+my[ix]*my[ix];
         m[ix]   = sqrt(msq[ix]);
         Bsq_ind[ix] = Bx_ind[ix]*Bx_ind[ix]+By_ind[ix]*By_ind[ix];
         B_ind[ix]=sqrt(Bsq_ind[ix]);
       }
     }


     memcpy(in,psi,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(psi_k,out_k,sizeof(fftw_complex)*alloc_local);
 
     memcpy(in,mx,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(mx_k,out_k,sizeof(fftw_complex)*alloc_local);
       
     memcpy(in,my,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(my_k,out_k,sizeof(fftw_complex)*alloc_local);

/****************************************************************************/
       c_new=10;
       for (i=0;i<Lxl;++i) {
           for (j=0;j<Ly;++j) {
               ix=i*2*(Ly/2+1)+j;
               
               if (psi[ix]<=0) {
                   psib[ix]=0;
               } else { psib[ix]=psi[ix];}
               
               nb[ix]=0;
               psib_ij[i][j]=psib[ix];
               n_ij[i][j]=nb[ix];
           }
       }
       
       for (i=0;i<Lxl;++i) {
           for (j=0;j<Ly;++j) {
//               ix=i*2*(Ly/2+1)+j;
              
               /* PBC */
 
               if (i>=Lxl-1) {psib_ij[i+1][j]=psib_ij[0][j];n_ij[i+1][j]=n_ij[0][j];}
               
               /* check all of the neighbors that have smaller n_ij */ 
               if (psib_ij[i][j]!=0 && n_ij[i][j] !=0 ) {
                   
                   if (psib_ij[i+1][j]!=0 && n_ij[i+1][j]<n_ij[i][j] && n_ij[i+1][j]!=0) {
                       n_ij[i][j]=n_ij[i+1][j];
                   }
                   if (psib_ij[i-1][j]!=0 && n_ij[i-1][j]<n_ij[i][j] && n_ij[i-1][j]!=0) {
                       n_ij[i][j]=n_ij[i-1][j];
                   }
                   if (psib_ij[i][j+1]!=0 && n_ij[i][j+1]<n_ij[i][j] && n_ij[i][j+1]!=0) {
                       n_ij[i][j]=n_ij[i][j+1];
                   }
                   if (psib_ij[i][j-1]!=0 && n_ij[i][j-1]<n_ij[i][j] && n_ij[i][j-1]!=0) {
                       n_ij[i][j]=n_ij[i][j-1];
                   }
                   if (psib_ij[i+1][j+1]!=0 && n_ij[i+1][j+1]<n_ij[i][j] && n_ij[i+1][j+1]!=0) {
                       n_ij[i][j]=n_ij[i+1][j+1];
                   }
                   if (psib_ij[i+1][j-1]!=0 && n_ij[i+1][j]<n_ij[i][j] && n_ij[i+1][j-1]!=0) {
                       n_ij[i][j]=n_ij[i+1][j-1];
                   }
                   if (psib_ij[i-1][j+1]!=0 && n_ij[i-1][j+1]<n_ij[i][j] && n_ij[i-1][j+1]!=0) {
                       n_ij[i][j]=n_ij[i-1][j+1];
                   }
                   if (psib_ij[i-1][j-1]!=0 && n_ij[i-1][j-1]<n_ij[i][j] && n_ij[i-1][j-1]!=0) {
                       n_ij[i][j]=n_ij[i-1][j-1];
                   }
                   
                   
                   /* give the value of n_ij to its neighbours */
                   if (psib_ij[i+1][j]!=0 && n_ij[i+1][j]>n_ij[i][j] || n_ij[i+1][j]==0) {
                       n_ij[i+1][j]=n_ij[i][j];
                   }
                   if (psib_ij[i-1][j]!=0 && n_ij[i-1][j]>n_ij[i][j] || n_ij[i-1][j]==0) {
                       n_ij[i-1][j]=n_ij[i][j];
                   }
                   if (psib_ij[i][j+1]!=0 && n_ij[i][j+1]>n_ij[i][j] || n_ij[i][j+1]==0) {
                       n_ij[i][j+1]=n_ij[i][j];
                   }
                   if (psib_ij[i][j-1]!=0 && n_ij[i][j-1]>n_ij[i][j] || n_ij[i][j-1]==0) {
                       n_ij[i][j-1]=n_ij[i][j];
                   }
                   if (psib_ij[i+1][j+1]!=0 && n_ij[i+1][j+1]>n_ij[i][j] || n_ij[i+1][j+1]==0) {
                       n_ij[i+1][j+1]=n_ij[i][j];
                   }
                   if (psib_ij[i+1][j-1]!=0 && n_ij[i+1][j-1]>n_ij[i][j] || n_ij[i+1][j-1]==0) {
                       n_ij[i+1][j-1]=n_ij[i][j];
                   }
                   if (psib_ij[i-1][j+1]!=0 && n_ij[i-1][j+1]>n_ij[i][j] || n_ij[i-1][j+1]==0) {
                       n_ij[i-1][j+1]=n_ij[i][j];
                   }
                   if (psib_ij[i-1][j-1]!=0 && n_ij[i-1][j-1]>n_ij[i][j] || n_ij[i-1][j-1]==0) {
                       n_ij[i-1][j-1]=n_ij[i][j];
                   }
                   
               }
               
               if (psib_ij[i][j]!=0 && n_ij[i][j] ==0 ) {
                   c_new=c_new+10;
                   n_ij[i][j]=c_new;
               
                   if (psib_ij[i+1][j]!=0 && n_ij[i+1][j]<n_ij[i][j] && n_ij[i+1][j]!=0) {
                       n_ij[i][j]=n_ij[i+1][j];
                   }
                   if (psib_ij[i-1][j]!=0 && n_ij[i-1][j]<n_ij[i][j] && n_ij[i-1][j]!=0) {
                       n_ij[i][j]=n_ij[i-1][j];
                   }
                   if (psib_ij[i][j+1]!=0 && n_ij[i][j+1]<n_ij[i][j] && n_ij[i][j+1]!=0) {
                       n_ij[i][j]=n_ij[i][j+1];
                   }
                   if (psib_ij[i][j-1]!=0 && n_ij[i][j-1]<n_ij[i][j] && n_ij[i][j-1]!=0) {
                       n_ij[i][j]=n_ij[i][j-1];
                   }
                   if (psib_ij[i+1][j+1]!=0 && n_ij[i+1][j+1]<n_ij[i][j] && n_ij[i+1][j+1]!=0) {
                       n_ij[i][j]=n_ij[i+1][j+1];
                   }
                   if (psib_ij[i+1][j-1]!=0 && n_ij[i+1][j]<n_ij[i][j] && n_ij[i+1][j-1]!=0) {
                       n_ij[i][j]=n_ij[i+1][j-1];
                   }
                   if (psib_ij[i-1][j+1]!=0 && n_ij[i-1][j+1]<n_ij[i][j] && n_ij[i-1][j+1]!=0) {
                       n_ij[i][j]=n_ij[i-1][j+1];
                   }
                   if (psib_ij[i-1][j-1]!=0 && n_ij[i-1][j-1]<n_ij[i][j] && n_ij[i-1][j-1]!=0) {
                       n_ij[i][j]=n_ij[i-1][j-1];
                   }
                   
                   
                   
                   if (psib_ij[i+1][j]!=0 && n_ij[i+1][j]>n_ij[i][j] || n_ij[i+1][j]==0) {
                       n_ij[i+1][j]=n_ij[i][j];
                   }
                   if (psib_ij[i-1][j]!=0 && n_ij[i-1][j]>n_ij[i][j] || n_ij[i-1][j]==0) {
                       n_ij[i-1][j]=n_ij[i][j];
                   }
                   if (psib_ij[i][j+1]!=0 && n_ij[i][j+1]>n_ij[i][j] || n_ij[i][j+1]==0) {
                       n_ij[i][j+1]=n_ij[i][j];
                   }
                   if (psib_ij[i][j-1]!=0 && n_ij[i][j-1]>n_ij[i][j] || n_ij[i][j-1]==0) {
                       n_ij[i][j-1]=n_ij[i][j];
                   }
                   if (psib_ij[i+1][j+1]!=0 && n_ij[i+1][j+1]>n_ij[i][j] || n_ij[i+1][j+1]==0) {
                       n_ij[i+1][j+1]=n_ij[i][j];
                   }
                   if (psib_ij[i+1][j-1]!=0 && n_ij[i+1][j-1]>n_ij[i][j] || n_ij[i+1][j-1]==0) {
                       n_ij[i+1][j-1]=n_ij[i][j];
                   }
                   if (psib_ij[i-1][j+1]!=0 && n_ij[i-1][j+1]>n_ij[i][j] || n_ij[i-1][j+1]==0) {
                       n_ij[i-1][j+1]=n_ij[i][j];
                   }
                   if (psib_ij[i-1][j-1]!=0 && n_ij[i-1][j-1]>n_ij[i][j] || n_ij[i-1][j-1]==0) {
                       n_ij[i-1][j-1]=n_ij[i][j];
                   }
                   
               }

                   
               }
       }


       /* scan from the top to check all connected clusters find each other */
       for (i=Lxl-1;i>=0;--i) {
           for (j=Ly-1;j>=0;--j) {
              
               if (n_ij[i][j] != 0) {
                     
                  if (psib_ij[i+1][j]!=0 && n_ij[i+1][j]<n_ij[i][j] && n_ij[i+1][j]!=0) {
                       n_ij[i][j]=n_ij[i+1][j];
                   }
                   if (psib_ij[i-1][j]!=0 && n_ij[i-1][j]<n_ij[i][j] && n_ij[i-1][j]!=0) {
                       n_ij[i][j]=n_ij[i-1][j];
                   }
                   if (psib_ij[i][j+1]!=0 && n_ij[i][j+1]<n_ij[i][j] && n_ij[i][j+1]!=0) {
                       n_ij[i][j]=n_ij[i][j+1];
                   }
                   if (psib_ij[i][j-1]!=0 && n_ij[i][j-1]<n_ij[i][j] && n_ij[i][j-1]!=0) {
                       n_ij[i][j]=n_ij[i][j-1];
                   }
                   if (psib_ij[i+1][j+1]!=0 && n_ij[i+1][j+1]<n_ij[i][j] && n_ij[i+1][j+1]!=0) {
                       n_ij[i][j]=n_ij[i+1][j+1];
                   }
                   if (psib_ij[i+1][j-1]!=0 && n_ij[i+1][j]<n_ij[i][j] && n_ij[i+1][j-1]!=0) {
                       n_ij[i][j]=n_ij[i+1][j-1];
                   }
                   if (psib_ij[i-1][j+1]!=0 && n_ij[i-1][j+1]<n_ij[i][j] && n_ij[i-1][j+1]!=0) {
                       n_ij[i][j]=n_ij[i-1][j+1];
                   }
                   if (psib_ij[i-1][j-1]!=0 && n_ij[i-1][j-1]<n_ij[i][j] && n_ij[i-1][j-1]!=0) {
                       n_ij[i][j]=n_ij[i-1][j-1];
                   }
               
               }
             }             
           }
            
       
       for (i=0;i<Lxl;++i) {
           for (j=0;j<Ly;++j) {
               ix=i*2*(Ly/2+1)+j;
               nb[ix]=n_ij[i][j];
           }
       }

       
/****************************************************************************/
/* THE SPIN-ORBIT COUPLING */
       
       gradx(psi_ij,gpx_ij);
       grady(psi_ij,gpy_ij);
       
       
       for (i=0;i<Lxl;++i) {
           for (j=0;j<Ly;++j) {
               mgp_ij[i][j]=mx_ij[i][j]*gpx_ij[i][j]/dx+my_ij[i][j]*gpy_ij[i][j]/dy;
               mgp3_ij[i][j]=mgp_ij[i][j]*mgp_ij[i][j]*mgp_ij[i][j];
               mgp5_ij[i][j]=pow(mgp_ij[i][j],5);
               
               
               mmgpx[i][j]=mx_ij[i][j]*mgp_ij[i][j];
               mmgpy[i][j]=my_ij[i][j]*mgp_ij[i][j];
               
               mmgp3x[i][j]=mx_ij[i][j]*mgp3_ij[i][j];
               mmgp3y[i][j]=my_ij[i][j]*mgp3_ij[i][j];
               
               
           }
       }
       
       gradx(mmgpx,gmmgpx);
       grady(mmgpy,gmmgpy);
       
       gradx(mmgp3x,gmmgp3x);
       grady(mmgp3y,gmmgp3y);
       
       gradx(mmgp5x,gmmgp5x);
       grady(mmgp5y,gmmgp5y);
       
       for (i=0;i<Lxl;++i) {
           for (j=0;j<Ly;++j) {
               
               div_ij[i][j]=gmmgpx[i][j]/dx+gmmgpy[i][j]/dy;
               div3_ij[i][j]=gmmgp3x[i][j]/dx+gmmgp3y[i][j]/dy;
               div5_ij[i][j]=gmmgp5x[i][j]/dx+gmmgp5y[i][j]/dy;
               
           }
       }
       
       for (i=0;i<Lxl;++i) {
           for (j=0;j<Ly;++j) {
               ix=i*2*(Ly/2+1)+j;
               gpx[ix]=gpx_ij[i][j];
               gpy[ix]=gpy_ij[i][j];
               mgp[ix]=mgp_ij[i][j];
               div[ix]=div_ij[i][j];
               div3[ix]=div3_ij[i][j];
               div5[ix]=div5_ij[i][j];
           }
       }
       
       
       
/***************************************************************************************/
/* POISSON SOLVER */
       
       
       for (i=0;i<Lxl;++i) {
           for (j=0;j<Ly/2+1;++j) {ix=i*(Ly/2+1)+j;
               lap_az_k[ix][0]=mu0*(k_vec[ix][0]*facx*my_k[ix][1]-k_vec[ix][1]*facy*mx_k[ix][1]);
               lap_az_k[ix][1]=mu0*(-k_vec[ix][0]*facx*my_k[ix][0]+k_vec[ix][1]*facy*mx_k[ix][0]);
       
               if(ksq[ix]==0) {az_k[ix][0]=0;az_k[ix][1]=0;}
               else{
                   az_k[ix][0]=-(lap_az_k[ix][0]/ksq[ix])*Sf;
                   az_k[ix][1]=-(lap_az_k[ix][1]/ksq[ix])*Sf;
               }
   
               
               Bx_k[ix][0]=-k_vec[ix][1]*facy*az_k[ix][1];
               Bx_k[ix][1]=k_vec[ix][1]*facy*az_k[ix][0];
               
               By_k[ix][0]=k_vec[ix][0]*facx*az_k[ix][1];
               By_k[ix][1]=-k_vec[ix][0]*facx*az_k[ix][0];
           }
       }
       
       memcpy(in_k,Bx_k,sizeof(fftw_complex)*alloc_local);
       fftw_execute(planb);
       memcpy(Bx_ind,out,2*sizeof(double)*alloc_local);
       
       memcpy(in_k,By_k,sizeof(fftw_complex)*alloc_local);
       fftw_execute(planb);
       memcpy(By_ind,out,2*sizeof(double)*alloc_local);
       
       /************************************************************************************/
       


     /* nonlinear term */
     for (i=0;i<Lxl;++i) {
       for (j=0;j<Ly;++j) {
         ix=i*2*(Ly/2+1)+j;

         psin[ix]=beta*(-w*psi[ix]*psi[ix]+psi[ix]*psi[ix]*psi[ix])
           - alpha*msq[ix] + epsilon1 * div[ix] +  epsilon2 * div3[ix]
           + epsilon3 * div5[ix];

         mxn[ix] = -gammaa*mx[ix]*msq[ix] + 2.*alpha*psi[ix]*mx[ix]
           + Bx_ind[ix] + Bx_ext + epsilon1*gpx[ix]*mgp[ix] + epsilon2*gpx[ix]*(mgp[ix]*mgp[ix]*mgp[ix])
           +epsilon3*gpx[ix]*pow(mgp[ix],5);
           
         myn[ix] = -gammaa*my[ix]*msq[ix] + 2.*alpha*psi[ix]*my[ix]
           + By_ind[ix] + By_ext + epsilon1*gpy[ix]*mgp[ix] + epsilon2*gpy[ix]*(mgp[ix]*mgp[ix]*mgp[ix])
           +epsilon3*gpy[ix]*pow(mgp[ix],5);

       }
     }

     memcpy(in,psin,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(psin_k,out_k,sizeof(fftw_complex)*alloc_local);

     memcpy(in,mxn,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(mxn_k,out_k,sizeof(fftw_complex)*alloc_local);

     memcpy(in,myn,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(myn_k,out_k,sizeof(fftw_complex)*alloc_local);
       

//     fftw_execute(plang1);
//     fftw_execute(plang2);
   
     /* eqs of motion in the k-space with conserved Gaussian noise */

//     for(i=0;i<Lxl;++i){
//       ig=i+myid*Lxl;
//       if(ig<=Lx/2){kx=ig*facx;} else {kx=(ig-Lx)*facx;}
//       for(j=0;j<Ly/2+1;++j){
//         ky=j*facy;
//         ix=i*(Ly/2+1)+j;
//         psi_k[ix][0]=k1[ix]*psi_k[ix][0]+k1n[ix]*((-ksq[ix])*psin_k[ix][0]-kx*nu1_k[ix][1]-ky*nu2_k[ix][1]);
//         psi_k[ix][1]=k1[ix]*psi_k[ix][1]+k1n[ix]*((-ksq[ix])*psin_k[ix][1]+kx*nu1_k[ix][0]+ky*nu2_k[ix][0]);
//         printf("ix, kx, ky, nu_k= %i %lf %lf %lf %lf %lf %lf \n", ix, kx, ky, nu1_k[ix][0], nu2_k[ix][0], nu1_k[ix][1], nu2_k[ix][1]);
//       }
//     }
 
     /* Eqs of motion in the k-space */
//     MPI_Barrier(MPI_COMM_WORLD);
     for (i=0;i<Lxl;++i) {
       for (j=0;j<Ly/2+1;++j) {
         ix=i*(Ly/2+1)+j;
         psi_k[ix][0]=k1[ix]*psi_k[ix][0]+k1n[ix]*(-ksq[ix])*psin_k[ix][0];
         psi_k[ix][1]=k1[ix]*psi_k[ix][1]+k1n[ix]*(-ksq[ix])*psin_k[ix][1];  
         sfr_k[ix][0]=psi_k[ix][0]*psi_k[ix][0]+psi_k[ix][1]*psi_k[ix][1];
         sfr_k[ix][1]=0.;

         mx_k[ix][0]=k1_m[ix]*mx_k[ix][0]+k1n_m[ix]*mxn_k[ix][0];
         mx_k[ix][1]=k1_m[ix]*mx_k[ix][1]+k1n_m[ix]*mxn_k[ix][1];

         my_k[ix][0]=k1_m[ix]*my_k[ix][0]+k1n_m[ix]*myn_k[ix][0];
         my_k[ix][1]=k1_m[ix]*my_k[ix][1]+k1n_m[ix]*myn_k[ix][1];

//         opsi_k[ix][0]=op[ix]*psi_k[ix][0];
//         opsi_k[ix][1]=op[ix]*psi_k[ix][1];

//         MPI_Barrier(MPI_COMM_WORLD);
//         printf("myid,psikRE,psikIM= %i %2.12lf %2.12lf \n",myid,psi_k[ix][0],psi_k[ix][1]);
//         MPI_Barrier(MPI_COMM_WORLD);
       }
     } 


     memcpy(in_k,psi_k,sizeof(fftw_complex)*alloc_local);
     fftw_execute(planb);
     memcpy(psi,out,2*sizeof(double)*alloc_local);

     memcpy(in_k,mx_k,sizeof(fftw_complex)*alloc_local);
     fftw_execute(planb);
     memcpy(mx,out,2*sizeof(double)*alloc_local);

     memcpy(in_k,my_k,sizeof(fftw_complex)*alloc_local);
     fftw_execute(planb);
     memcpy(my,out,2*sizeof(double)*alloc_local);


/*************************************************************************************/
/* ENERGY CALCULATION */

   if (n==1 || n%nout==0) {

     memcpy(in,psi,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(psi_k,out_k,sizeof(fftw_complex)*alloc_local);

     memcpy(in,mx,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(mx_k,out_k,sizeof(fftw_complex)*alloc_local);

     memcpy(in,my,2*sizeof(double)*alloc_local);
     fftw_execute(plan);
     memcpy(my_k,out_k,sizeof(fftw_complex)*alloc_local);
     
     for (l=0;l<10;++l) { enl[l]=0; eng[l]=0; }

     /* different terms of energy */

       /* k-space */
       for (i=0;i<Lxl;++i) {
         for (j=0;j<Ly/2+1;++j) {
           ix=i*(Ly/2+1)+j;

//             k1f_e=((q0*q0-ksq[ix])*(q0*q0-ksq[ix])+b0)*((q1*q1-ksq[ix])*(q1*q1-ksq[ix])+b1); /* 2-mode */             
             
             k1f_e=((q0*q0-ksq[ix])*(q0*q0-ksq[ix])); /* 1-mode */
             psi_k_e[ix][0]=k1f_e*psi_k[ix][0]*Sf;
             psi_k_e[ix][1]=k1f_e*psi_k[ix][1]*Sf;


          k1f_m_e=(-w0*w0*ksq[ix]+ksq[ix]*ksq[ix]);
          mx_k_e[ix][0]=k1f_m_e*mx_k[ix][0]*Sf;
          mx_k_e[ix][1]=k1f_m_e*mx_k[ix][1]*Sf;

          my_k_e[ix][0]=k1f_m_e*my_k[ix][0]*Sf;
          my_k_e[ix][1]=k1f_m_e*my_k[ix][1]*Sf;

//          gxmx_k[ix][1]= k_vec[ix][0]*facx*mx_k[ix][0]*Sf;

        }
      }

      memcpy(in_k,psi_k_e,sizeof(fftw_complex)*alloc_local);
      fftw_execute(planb);
      memcpy(psi_e,out,2*sizeof(double)*alloc_local);


      memcpy(in_k,mx_k_e,sizeof(fftw_complex)*alloc_local);
      fftw_execute(planb);
      memcpy(mx_e,out,2*sizeof(double)*alloc_local);


      memcpy(in_k,my_k_e,sizeof(fftw_complex)*alloc_local);
      fftw_execute(planb);
      memcpy(my_e,out,2*sizeof(double)*alloc_local);



     for (l=0;l<10;++l) {

      enl[l]=0;
      for (i=0;i<Lxl;++i) {
        for (j=0;j<Ly;++j) {
          ix=i*2*(Ly/2+1)+j;
          if (l==0) { enl[l]+=beta*(0.5*psi[ix]*psi_e[ix]+0.5*r*psi[ix]*psi[ix]+0.25*psi[ix]*psi[ix]*psi[ix]*psi[ix]); } /*pfc*/
            else if (l==1) { enl[l]+=-(epsilon1/2.)*(mgp[ix]*mgp[ix])-(epsilon2/4.)*(mgp[ix]*mgp[ix]*mgp[ix]*mgp[ix])
                                     -(epsilon3/6.)*pow(mgp[ix],6);  } /* coupling */
          else if (l==2) { enl[l]+=(mx[ix]*mx_e[ix]+my[ix]*my_e[ix])+0.5*r_c*msq[ix]+0.25*gammaa*msq[ix]*msq[ix]; } /* magnetic */
          else if (l==3) { enl[l]+=-alpha*psi[ix]*msq[ix]; } /* symmetry-breaking term */ 
          else if (l==4) { enl[l]+=-mx[ix]*Bx_ext-my[ix]*By_ext; } /* -m.b */
          else if (l==5) { enl[l]=enl[0]+enl[1]+enl[2]+enl[3]+enl[4]; }
        }
      }
  
      MPI_Reduce(&enl[l], &eng[l], 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD);
        
      eng[l]=eng[l]*Sf;

    } /* l loop for energy terms */

    printout_num ("mean_energy.txt", Lxl, eng[0], eng[1], eng[2], eng[3], eng[4], eng[5]);

  } /* n% nout - end of energy calculation */
 
/*********************************************************************************************/
/* TIME LOOP END */
  
  } /* n */

/*********************************************************************************************/
/* STRUCTURE FACTOR */

//  kx_ky(Lxl, k_vec);
  printout_arr_k ("kx_ky.txt", k_vec, Lxl);
  printout_arr_k ("sfr.txt", sfr_k, Lxl); 


/*********************************************************************************************/

   /* clock stops */
   t2 = MPI_Wtime();
   printf("Elapsed time is %f\n", t2 - t1);

   fftw_destroy_plan(plan);
   fftw_destroy_plan(planb);
   MPI_Finalize();  

/*********************************************************************************************/
/* END OF MAIN */

} /* main */


/*********************************************************************************************/
/* FUNCTIONS */
/* *******************************************************************************************/
/* PRINT OUT THE REAL-SPACE ARRAY "ARR" TO SUCCESSIVE FILES "FNAME_ARR%D.TXT" */

void printout_arr (char* fname_arr, double* arr, int Lxl)

{
  FILE *fout_arr;
  char fname_arr1[BUFSIZ];
  int ix,pc,i,j;
  sprintf(fname_arr1,"%s%d.txt",fname_arr,idx);
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid==pc) {
      MPI_Barrier(MPI_COMM_WORLD);
      fout_arr=fopen(fname_arr1,"a");
      for (i=0; i<Lxl; ++i) {
        for (j=0; j<Ly; ++j) {
//         for (i=0; i<Lxl; ++i) {
          ix = i*2*(Ly/2+1) + j;
//          if (arr[ix]!=0) {fprintf(fout_arr,"%lf \n",arr[ix]);}
         fprintf(fout_arr,"%lf \n",arr[ix]);
        }
      }
      fclose(fout_arr);
    }
  }
}



/****************************************************************************************************/

void dump_out_ij(char* fname_arr, double arr[Lx][Ly])

{
  FILE *fout_arr;
  char fname_arr1[BUFSIZ];
  int i,j;
  sprintf(fname_arr1,"%s%d.txt",fname_arr,idx);
  fout_arr=fopen(fname_arr1,"a");
  for (i=0; i<Lx; ++i) {
    for (j=0; j<Ly; ++j) {
          fprintf(fout_arr,"%lf \n",arr[i][j]);
        }
      }
      fclose(fout_arr);
}
  




/* *******************************************************************************************/
/* PRINT OUT THE REAL-SPACE ARRAY COMPONENTS "ARR_X" AND "ARR_Y"
 *  TO SUCCESSIVE FILES "FNAME_VEC_ARR%D.TXT" */

void printout_vec_arr (char* fname_vec_arr, double* arr_x, double* arr_y, int Lxl)

{
  FILE *fout_vec_arr;
  char fname_arr1[BUFSIZ];
  int ix,pc,i,j,ii;
  sprintf(fname_arr1,"%s%d.txt",fname_vec_arr,idx);
  MPI_Barrier(MPI_COMM_WORLD);
  for (pc=0; pc<np; pc++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid==pc) {
      MPI_Barrier(MPI_COMM_WORLD);
      fout_vec_arr=fopen(fname_arr1,"a");
      for (i=0; i<Lxl; ++i) {
        ii=pc*Lxl+i;
        for (j=0; j<Ly; ++j) {
          ix = i*2*(Ly/2+1) + j;
 //        if ((arr_x[ix]!=0) && (arr_y[ix]!=0))  fprintf(fout_vec_arr,"%i %i %lf %lf \n",ii,j,arr_x[ix],arr_y[ix]);
         fprintf(fout_vec_arr,"%i %i %lf %lf \n",ii,j,arr_x[ix],arr_y[ix]);
        }
      }
      fclose(fout_vec_arr);
    }
  }
}


/* *******************************************************************************************/
/* PRINT OUT NUMBERS "N1", .., "N5" TO A SINGLE FILE "FNAME_N" */
 
void printout_num (char* fname_n, int Lxl, double n1, double n2, double n3, double n4, double n5, double n6)

{
  FILE *fout_n;
  if (myid==0) {
    fout_n=fopen(fname_n,"a");
    fprintf(fout_n,"%2.15lf %2.15lf %2.15lf %2.15lf %2.15lf %2.20lf \n", n1, n2, n3, n4, n5, n6);
    fclose(fout_n);
  }
}

/* *******************************************************************************************/
/* CALCULATE THE MEAN VALUE OF THE ARRAY "ARR" AND SAVE IT IN "MEAN_FUNC" */

void mean(double* arr, int Lxl, double* mean_func)

{
  double arrml, arrm;
  int i,j,ix;
  for (i=0;i<Lxl;++i) {
    for (j=0;j<Ly;++j) {
      ix=i*2*(Ly/2+1)+j;
      arrml+=arr[ix];
    }
  }
  MPI_Reduce( &arrml, &arrm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
  if (myid==0) {arrm = arrm/(Lx*Ly); *mean_func=arrm;}
}

/* *******************************************************************************************/
/* PRODUCES THE K-SPACE "KX" AND "KY" AND SAVE THEM AS THE REAL AND IMAGINARY PARTS OF "K_VEC_FUNC"*/

void kx_ky(double Lxl, fftw_complex* k_vec_func)
{
  int i,j,ig,ix;
  double kx,ky;
  for(i=0;i<Lxl;++i){
    ig=i+myid*Lxl;
    if(ig<=Lx/2){kx=ig;} else {kx=(ig-Lx);}
      for(j=0;j<Ly/2+1;++j){ix=i*(Ly/2+1)+j;
        ky=j;
        k_vec_func[ix][0]=kx;
        k_vec_func[ix][1]=ky;
      }
  }  
}

/* *******************************************************************************************/
/* PRINT OUT THE K-SPACE ARRAY "ARR_K TO THE FILE "FNAME_ARR_K" */

void printout_arr_k (char* fname_arr_k, fftw_complex* arr_k, int Lxl)

{
 
  FILE *fout_arr_k;
  int i,j,ix,pc;
  MPI_Barrier(MPI_COMM_WORLD);
    for (pc=0;pc<np;pc++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (myid==pc) {
        MPI_Barrier(MPI_COMM_WORLD);
        fout_arr_k=fopen(fname_arr_k,"a");
        for(i=0;i<Lxl;++i){
          for(j=0;j<Ly/2+1;++j){ix=i*(Ly/2+1)+j;
             fprintf(fout_arr_k," %2.12lf %2.12lf \n",arr_k[ix][0],arr_k[ix][1]);
          }
        }
        fclose(fout_arr_k);
     } /* for myid */
   } /* for pc */

}

/* *******************************************************************************************/

void gradx(double p1[Lx][Ly], double gp[Lx][Ly])

{

int i,j;

/* center */

for (i=1;i<(Lx-1);++i) {
  for (j=1;j<(Ly-1);++j) {
    gp[i][j]=(p1[i+1][j]-p1[i-1][j])/2.0;
  }
}

/* edges */

for (j=1;j<Ly-1;++j) {
  gp[0][j]=(p1[1][j]-p1[Lx-1][j])/2.0;
  gp[Lx-1][j]=(p1[0][j]-p1[Lx-2][j])/2.0;

  }

for (i=1;i<Lx-1;++i) {
  gp[i][0]=(p1[i+1][0]-p1[i-1][0])/2.0;
  gp[i][Ly-1]=(p1[i+1][Ly-1]-p1[i-1][Ly-1])/2.0;

}

/* corners */

gp[0][0]=(p1[1][0]-p1[Lx-1][0])/2.0;
gp[Lx-1][0]=(p1[0][0]-p1[Lx-2][0])/2.0;
gp[0][Ly-1]=(p1[1][Ly-1]-p1[Lx-1][Ly-1])/2.0;
gp[Lx-1][Ly-1]=(p1[0][Ly-1]-p1[Lx-2][Ly-1])/2.0;

}

/* *******************************************************************************************/

void grady(double p1[Lx][Ly], double gp[Lx][Ly])

{

int i,j;

/* center */

for (i=1;i<(Lx-1);++i) {
  for (j=1;j<(Ly-1);++j) {
    gp[i][j]=(p1[i][j+1]-p1[i][j-1])/2.0;
  }
}

/* edges */

for (j=1;j<Ly-1;++j) {
  gp[0][j]=(p1[0][j+1]-p1[0][j-1])/2.0;
  gp[Lx-1][j]=(p1[Lx-1][j+1]-p1[Lx-1][j-1])/2.0;

  }

for (i=1;i<Lx-1;++i) {
  gp[i][0]=(p1[i][1]-p1[i][Ly-1])/2.0;
  gp[i][Ly-1]=(p1[i][0]-p1[i][Ly-2])/2.0;

}

/* corners */

gp[0][0]=(p1[0][1]-p1[0][Ly-1])/2.0;
gp[Lx-1][0]=(p1[Lx-1][1]-p1[Lx-1][Ly-1])/2.0;
gp[0][Ly-1]=(p1[0][0]-p1[0][Ly-2])/2.0;
gp[Lx-1][Ly-1]=(p1[Lx-1][0]-p1[Lx-1][Ly-2])/2.0;

}





