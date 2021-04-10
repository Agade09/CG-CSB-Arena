#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <list>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
#include <ext/stdio_filebuf.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <omp.h>
#include <algorithm>
using namespace std;
using namespace std::chrono;

constexpr bool tests{true};
constexpr bool Debug_AI{false},Timeout{false};
constexpr double FirstTurnTime{1*(Timeout?1:10)},TimeLimit{0.075*(Timeout?1:10)};
constexpr int PIPE_READ{0},PIPE_WRITE{1};
constexpr int N{2};
constexpr int N_L{3},Pod_Radius{400},CP_Radius{600};
constexpr double DegToRad{M_PI/180.0},Max_Speed{0.85*200/0.15};

bool stop{false};//Global flag to stop all arena threads when SIGTERM is received

struct vec{
    double x,y;
    inline double norm2()const noexcept{
        return x*x+y*y;
    }
    inline double norm()const noexcept{
        return sqrt(norm2());
    }
    inline void operator/=(const double a)noexcept{
        x/=a;
        y/=a;
    }
    inline void normalize()noexcept{
        *this/=norm();
    }
    inline bool operator!=(const vec &a)const noexcept{
        return x!=a.x || y!=a.y;
    }
    inline bool operator==(const int a)const noexcept{
        return x==a && y==a;
    }
    inline vec operator+(const vec &a)const noexcept{
        return vec{x+a.x,y+a.y};
    }
    inline vec operator-(const vec &a)const noexcept{
        return vec{x-a.x,y-a.y};
    }
    inline double operator*(const vec &a)const noexcept{
        return x*a.x+y*a.y;
    }
    inline vec operator*(const double a)const noexcept{
        return vec{x*a,y*a};
    }
    inline vec operator/(const double a)const noexcept{
        return vec{x/a,y/a};
    }
    inline void operator+=(const vec &a)noexcept{
        x+=a.x;
        y+=a.y;
    }
    inline void operator-=(const vec &a)noexcept{
        x-=a.x;
        y-=a.y;
    }
    inline void operator*=(const double a)noexcept{
        x*=a;
        y*=a;
    }
    inline void round_vec(){
        x=round(x);
        y=round(y);
    }
};

inline ostream& operator<<(ostream &os,const vec &a){
    os << a.x << " " << a.y;
    return os;
}

inline istream& operator>>(istream &is,vec &a){
    is >> a.x >> a.y;
    return is;
}

typedef vector<vec> Map;
const vector<Map> Maps{{{12460,1350},{10540,5980},{3580,5180},{13580,7600}},{{3600,5280},{13840,5080},{10680,2280},{8700,7460},{7200,2160}},{{4560,2180},{7350,4940},{3320,7230},{14580,7700},{10560,5060},{13100,2320}},{{5010,5260},{11480,6080},{9100,1840}},{{14660,1410},{3450,7220},{9420,7240},{5970,4240}},{{3640,4420},{8000,7900},{13300,5540},{9560,1400}},{{4100,7420},{13500,2340},{12940,7220},{5640,2580}},{{14520,7780},{6320,4290},{7800,860},{7660,5970},{3140,7540},{9520,4380}},{{10040,5970},{13920,1940},{8020,3260},{2670,7020}},{{7500,6940},{6000,5360},{11300,2820}},{{4060,4660},{13040,1900},{6560,7840},{7480,1360},{12700,7100}},{{3020,5190},{6280,7760},{14100,7760},{13880,1220},{10240,4920},{6100,2200}},{{10323,3366},{11203,5425},{7259,6656},{5425,2838}}};
default_random_engine generator(system_clock::now().time_since_epoch().count());

struct pod{
    vec r,v;
    double angle;
    int next,lap,shield_cd;
};

struct action{
    int thrust;
    double delta_angle;
};

struct state{
    array<pod,N*2> p;//runner and then blocker
    array<int,N> timeout;
    Map C;
};

struct collision{
    double t;
    int a,b;
};

struct AI{
    int id,pid,outPipe,errPipe,inPipe,turnOfDeath;
    string name;
    inline void stop(const int turn=-1){
        if(alive()){
            kill(pid,SIGTERM);
            int status;
            waitpid(pid,&status,0);//It is necessary to read the exit code for the process to stop
            if(!WIFEXITED(status)){//If not exited normally try to "kill -9" the process
                kill(pid,SIGKILL);
            }
            turnOfDeath=turn;
        }
    }
    inline bool alive()const{
        return kill(pid,0)!=-1;//Check if process is still running
    }
    inline void Feed_Inputs(const string &inputs){
        if(write(inPipe,&inputs[0],inputs.size())!=inputs.size()){
            throw(5);
        }
    }
    inline ~AI(){
        close(errPipe);
        close(outPipe);
        close(inPipe);
        stop();
    }
};

void StartProcess(AI &Bot){
    int StdinPipe[2];
    int StdoutPipe[2];
    int StderrPipe[2];
    if(pipe(StdinPipe)<0){
        perror("allocating pipe for child input redirect");
    }
    if(pipe(StdoutPipe)<0){
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        perror("allocating pipe for child output redirect");
    }
    if(pipe(StderrPipe)<0){
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        perror("allocating pipe for child stderr redirect failed");
    }
    int nchild{fork()};
    if(nchild==0){//Child process
        if(dup2(StdinPipe[PIPE_READ],STDIN_FILENO)==-1){// redirect stdin
            perror("redirecting stdin");
            return;
        }
        if(dup2(StdoutPipe[PIPE_WRITE],STDOUT_FILENO)==-1){// redirect stdout
            perror("redirecting stdout");
            return;
        }
        if(dup2(StderrPipe[PIPE_WRITE],STDERR_FILENO)==-1){// redirect stderr
            perror("redirecting stderr");
            return;
        }
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        execl(Bot.name.c_str(),Bot.name.c_str(),(char*)NULL);//(char*)Null is really important
        //If you get past the previous line its an error
        perror("exec of the child process");
    }
    else if(nchild>0){//Parent process
        close(StdinPipe[PIPE_READ]);//Parent does not read from stdin of child
        close(StdoutPipe[PIPE_WRITE]);//Parent does not write to stdout of child
        close(StderrPipe[PIPE_WRITE]);//Parent does not write to stderr of child
        Bot.inPipe=StdinPipe[PIPE_WRITE];
        Bot.outPipe=StdoutPipe[PIPE_READ];
        Bot.errPipe=StderrPipe[PIPE_READ];
        Bot.pid=nchild;
    }
    else{//failed to create child
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        perror("Failed to create child process");
    }
}

inline double Angular_Distance(const double a,const double b)noexcept{
    const double diff{abs(a-b)};
    return min(diff,360-diff);
}

inline double Angle_Sum(const double a,const double b)noexcept{
    const double sum{a+b};
    return sum+(sum>360?-360:sum<0?360:0);
}

inline double Angle(const vec &d){
    if(d.x==0 && d.y==0){
        return 0;//Avoid domain error
    }
    const double rad{atan2(d.y,d.x)};// atan2() returns radians [-Pi,Pi]
    return (rad>0?rad:2*M_PI+rad)/DegToRad;//[0,360]
}

inline double Closest_Angle(const double a,const double t){
    if(Angular_Distance(a,t)<=18){
        return t;
    }
    else{
        const double a1{Angle_Sum(a,18.0)},a2{Angle_Sum(a,-18.0)};
        return Angular_Distance(a1,t)<Angular_Distance(a2,t)?a1:a2;
    }
}

inline void Passes_Checkpoint(const pod &p,const int idx_p,const vec &CP,list<collision> &C,const double T)noexcept{
    const vec R{p.r-CP};
    const double R2{R.norm2()},RV{R*p.v},V2{p.v.norm2()},det{RV*RV-V2*(R2-pow(CP_Radius,2))};
    if(RV<0.0 && det>=0.0){
        const double t{T-(RV+sqrt(det))/V2};
        if(t<1){
            C.push_back(collision{t,idx_p,-1});
        }
    }
}

void is_colliding(const pod &a,const pod &b,const int idx_a,const int idx_b,const double T,list<collision> &C){
    const vec R{a.r-b.r},V{a.v-b.v};
    const double R2{R.norm2()},RV{R*V},V2{V.norm2()},det{RV*RV-V2*(R2-pow(2*Pod_Radius,2))};
    if(RV<0 && det>=0){
        const double t{T-(RV+sqrt(det))/V2};
        if(t<1){//Margin from pb4 to handle pod overlap
            if(tests && t<-0.1){
                cerr << "Negative time collision at time " << t << " between pods " << idx_a << " and " << idx_b << endl;
                cerr << "Pos: " << a.r << " " << b.r << " Pods are " << (a.r-b.r).norm() << " apart." << endl;
                cerr << "Vel: " << a.v << " " << b.v << endl;
                throw(0);
            }
            /*
            cerr << "Collisiong between " << idx_a << " and " << idx_b << " planned at time " << t << " pods are currently " << (a.r-b.r).norm() << " apart" << endl;
            cerr << "Pos: " << a.r << " " << b.r << " Pods are " << (a.r-b.r).norm() << " apart. R: " << R << endl;
            cerr << "Vel: " << a.v << " " << b.v << " V: " << V << endl; 
            cerr << "T: " << T << " " << -(RV+sqrt(det))/V2 << endl;
            */
            C.push_back(collision{t,idx_a,idx_b});
        }
    }
}

inline void ScanForAllCollisions(list<collision> &C,const state &S)noexcept{
    for(int i=0;i<S.p.size();++i){
        for(int j=i+1;j<S.p.size();++j){
            is_colliding(S.p[i],S.p[j],i,j,0,C);
        }
        Passes_Checkpoint(S.p[i],i,S.C[S.p[i].next],C,0);
    }
}

inline void ScanForNewCollisions(list<collision> &C,const state &S,const double T,const int a,const int exclude)noexcept{
    for(int i=0;i<S.p.size();++i){
        if(i!=a && i!=exclude){
            is_colliding(S.p[a],S.p[i],a,i,T,C);
        }
    }
    Passes_Checkpoint(S.p[a],a,S.C[S.p[a].next],C,T);
}

template <bool is_coll> void Simulation_Step(const collision &c,state &S,list<collision> &C,double &T){
    for(pod &p:S.p){
        p.r+=p.v*(c.t-T);
    }
    T=c.t;
    if(is_coll){
        if(c.b!=-1){//Pod collision
            //cerr << "Collision at time " << T << " between " << c.a << " and " << c.b << endl;
            C.remove_if([&](const collision &col){return col.a==c.a || col.b==c.a || col.a==c.b || col.b==c.b;});
            pod &a=S.p[c.a],&b=S.p[c.b];
            vec n=b.r-a.r,v_rel=b.v-a.v;
            double Inv_n_norm2{1.0f/n.norm2()},projection{n*v_rel},m_a{a.shield_cd==4?10.0f:1.0f},m_b{b.shield_cd==4?10.0f:1.0f},mu{m_a*m_b/(m_a+m_b)};
            vec f=n*projection*Inv_n_norm2*mu;
            b.v-=f/m_b;
            a.v+=f/m_a;
            double impulse{f.norm()};
            if(impulse<120){
                if(tests && impulse==0){
                    cerr << "Warning: Impulse is 0, division would crash" << endl;
                    cerr << "n: " << n << " v_rel: " << v_rel << " projection: " << projection << " Inv_n_norm2: " << Inv_n_norm2 << " mu: " << mu << endl;
                    f=n*sqrt(Inv_n_norm2);
                    impulse=1;
                }
                f*=120/impulse;
            }
            b.v-=f/m_b;
            a.v+=f/m_a;
            ScanForNewCollisions(C,S,T,c.a,c.b);
            ScanForNewCollisions(C,S,T,c.b,c.a);
        }
        else{//CP pass
            C.remove_if([&](const collision &col){return col.a==c.a && col.b==-1;});
            pod &p=S.p[c.a];
             ++p.next;
            if(p.next==S.C.size()){
                p.next=0;
            }
            else if(p.next==1){
                ++p.lap;
            }
            S.timeout[c.a<S.p.size()/2?0:1]=100;
        }
    }
}

void Simulate_Pod_Move(pod &p,const action &m){
    p.angle=Angle_Sum(p.angle,m.delta_angle);
    if(m.thrust!=-1){
        if(p.shield_cd==0){
            vec dir;
            sincos(p.angle*DegToRad,&dir.y,&dir.x);
            p.v+=dir*m.thrust;
        }
    }   
    else{//Shield
        p.shield_cd=4;
    }
}

void Simulate_Moves(state &S,const array<action,2> &MyMove,const array<action,2> &EnemyMove){
    for(int i=0;i<S.p.size()/2;++i){
        Simulate_Pod_Move(S.p[i],MyMove[i]);
        Simulate_Pod_Move(S.p[S.p.size()/2+i],EnemyMove[i]);
    }
}

template <bool verbose> int Simulate(state &S,const array<action,2> &MyMove,const array<action,2> &EnemyMove){
    Simulate_Moves(S,MyMove,EnemyMove);
    double T{0};
    list<collision> C;
    ScanForAllCollisions(C,S);
    if(verbose){
        cerr << C.size() << " collisions" << endl;
    }
    int collisions{0};
    while(!C.empty()){
        collision first=*min_element(C.begin(),C.end(),[](const collision &a,const collision &b){return a.t<b.t;});
        Simulation_Step<true>(first,S,C,T);
        if(first.b!=-1){
            ++collisions;
        }
    }
    Simulation_Step<false>(collision{1,-1,-1},S,C,T);
    for(pod &p:S.p){
        p.r.round_vec();
        p.v=vec{trunc(0.85f*p.v.x),trunc(0.85f*p.v.y)};
        p.shield_cd=max(0,p.shield_cd-1);
    }
    for(int &t:S.timeout){
        --t;
    }
    if(tests){
        for(int i=0;i<S.p.size();++i){
            for(int j=i+1;j<S.p.size();++j){
                const double dist2{(S.p[i].r-S.p[j].r).norm2()};
                if(dist2<pow(2*Pod_Radius-25,2)){
                    cerr << "Warning: Pods " << i << " and " << j << " are " << sqrt(dist2) << " apart, in " << S.p[i].r << " and " << S.p[j].r << endl;
                }
            }
        }
    }
    return collisions;
}

inline Map Generate_Map(default_random_engine &generator)noexcept{
    uniform_int_distribution<int> Map_Distrib(0,Maps.size()-1);
    Map C{Maps[Map_Distrib(generator)]};
    uniform_int_distribution<int> Rotate_Distrib(0,C.size()-1),Delta_Distrib(-30,30);
    rotate(C.begin(),C.begin()+Rotate_Distrib(generator),C.end());
    for(vec &cp:C){
        cp+=vec{static_cast<double>(Delta_Distrib(generator)),static_cast<double>(Delta_Distrib(generator))};
    }
    return C;
}

inline string EmptyPipe(const int fd){
    int nbytes;
    if(ioctl(fd,FIONREAD,&nbytes)<0){
        throw(4);
    }
    string out;
    out.resize(nbytes);
    if(read(fd,&out[0],nbytes)<0){
        throw(4);
    }
    return out;
}

bool ValidStoiArgument(const string &s){
    try{
        stoi(s);
        return true;
    }
    catch(...){
        return false;
    }
}

bool IsValidMove(const state &S,const AI &Bot,const string &Move){
    stringstream ss(Move);
    for(int i=0;i<2;++i){
        string line;
        getline(ss,line);
        stringstream ss2(line);
        vec target;
        string thrust;
        if(!(ss2 >> target >> thrust)){
            return false;
        }
        if(thrust!="SHIELD" && thrust!="BOOST" && !ValidStoiArgument(thrust)){
            return false;
        }
    }
    return true;
}

string GetMove(const state &S,AI &Bot,const int turn){
    pollfd outpoll{Bot.outPipe,POLLIN};
    time_point<system_clock> Start_Time{system_clock::now()};
    string out;
    while(static_cast<duration<double>>(system_clock::now()-Start_Time).count()<(turn==1?FirstTurnTime:TimeLimit) && !IsValidMove(S,Bot,out)){
        double TimeLeft{(turn==1?FirstTurnTime:TimeLimit)-static_cast<duration<double>>(system_clock::now()-Start_Time).count()};
        if(poll(&outpoll,1,TimeLeft)){
            out+=EmptyPipe(Bot.outPipe);
        }
    }
    if(!IsValidMove(S,Bot,out)){
    	throw(1);
    }
    return out;
}

action StringToAction(const string &mv_str,const pod &p,const int turn){
    stringstream ss(mv_str);
    action mv;
    string thrust_str;
    vec target;
    ss >> target >> thrust_str;
    mv.thrust=thrust_str=="BOOST"?650:thrust_str=="SHIELD"?-1:stoi(thrust_str);
    double desired_angle{Angle(target-p.r)};
    const double angle_given{turn==1?desired_angle:Closest_Angle(p.angle,desired_angle)};
    const double angle_dist{Angular_Distance(angle_given,p.angle)};
    mv.delta_angle=Angular_Distance(desired_angle,Angle_Sum(p.angle,angle_dist))<Angular_Distance(desired_angle,Angle_Sum(p.angle,-angle_dist))?angle_dist:-angle_dist;
    if(tests && Angular_Distance(desired_angle,Angle_Sum(p.angle,mv.delta_angle))>1){
        cerr << "Target to delta_angle conversion is dodgy " << setprecision(3) << p.angle  << " " << angle_given << " " << mv.delta_angle << endl;
    }
    if(tests && turn>1 && abs(mv.delta_angle)>18+1e-3){
        cerr << "Error: Delta angle : " << mv.delta_angle << endl;
        cerr << "Desired: " << desired_angle << " Current: " << p.angle << " Given: " << angle_given << endl;
    }
    return mv;
}

array<action,2> StringToAction(const string &mv_str,const state &S,const int player_id,const int turn){
    array<action,2> Actions;
    stringstream ss(mv_str);
    for(int i=0;i<N;++i){
        string mv_pod_str;
        getline(ss,mv_pod_str);
        Actions[i]=StringToAction(mv_pod_str,S.p[player_id*2+i],turn);
        //cerr << mv_pod_str << endl;
        //cerr << Actions[i].thrust << " " << Actions[i].delta_angle << endl;
    }
    return Actions;
}

inline bool Has_Won(const array<AI,N> &Bot,const int idx)noexcept{
    if(!Bot[idx].alive()){
        return false;
    }
    for(int i=0;i<N;++i){
        if(i!=idx && Bot[i].alive()){
            return false;
        }
    }
    return true;
}

inline bool All_Dead(const array<AI,N> &Bot)noexcept{
    for(const AI &b:Bot){
        if(b.alive()){
            return false;
        }
    }
    return true;
}

inline int CPs_Left(const pod &p,const Map &C){
    const int CPs_Left_This_Lap{p.next==0?1:1+(static_cast<int>(C.size())-p.next)};
    return (3-p.lap-1)*C.size()+CPs_Left_This_Lap;
}

int Play_Game(const array<string,N> &Bot_Names,const Map &C,const array<array<vec,2>,N> &Spawns){
    array<AI,N> Bot;
    state S;
    S.C=C;
    for(int i=0;i<N;++i){
        Bot[i].id=i;
        Bot[i].name=Bot_Names[i];
        S.timeout[i]=100;
        StartProcess(Bot[i]);
        for(int j=0;j<2;++j){
            pod &p{S.p[i*2+j]};
            p.r=Spawns[i][j];
            p.v=vec{0,0};
            p.lap=0;
            p.next=1;
            p.shield_cd=0;
            p.angle=Angle(S.C[1]-p.r);
        }
    }
    int turn{0};
    //Feed first turn inputs
    for(AI &b:Bot){
        stringstream ss;
        ss << N_L << endl
        ss << C.size() << endl;
        for(const vec &cp:C){
            ss << cp << endl;
        }
        b.Feed_Inputs(ss.str().c_str());
    }
    while(turn++<500){
        //cerr << "Turn " << turn << endl;
        array<array<action,2>,N> Actions_Played;
        for(int id=0;id<N;++id){
            if(Bot[id].alive()){
                //Feed turn inputs
                stringstream ss;
                for(int j=0;j<2*N;++j){
                    const int pod_idx{static_cast<int>((id*2+j)%S.p.size())};
                    ss << S.p[pod_idx].r << " " << S.p[pod_idx].v << " " << round(S.p[pod_idx].angle) << " " << S.p[pod_idx].next << endl;
                }
                try{
                    Bot[id].Feed_Inputs(ss.str().c_str());
                    string out{GetMove(S,Bot[id],turn)};
                    Actions_Played[id]=StringToAction(out,S,id,turn);
                    string err_str{EmptyPipe(Bot[id].errPipe)};
                    if(Debug_AI){
                        ofstream err_out("log.txt",ios::app);
                        err_out << err_str << endl;
                    }
                }
                catch(int ex){
                    if(ex==1){//Timeout
                        cerr << "Loss by Timeout of AI " << Bot[id].id << " name: " << Bot[id].name << endl;
                    }
                    else if(ex==3){
                        cerr << "Invalid move from AI " << Bot[id].id << " name: " << Bot[id].name << endl;
                    }
                    else if(ex==4){
                        cerr << "Error emptying pipe of AI " << Bot[id].name << endl;
                    }
                    else if(ex==5){
                        cerr << "AI " << Bot[id].name << " died before being able to give it inputs" << endl;
                    }
                    else{
                        cerr << "AI " << Bot[id].name << " stopped after throw int " << ex << endl;
                    }
                    Bot[id].stop(turn);
                }
            }
        }
        for(int id=0;id<N;++id){
            if(S.timeout[id]<=0){
                Bot[id].stop(turn);
            }
        }
        if(All_Dead(Bot)){
            return -1;//Draw
        }
        else{
            for(int i=0;i<2;++i){
                if(Has_Won(Bot,i)){
                    return i;
                }
            }
        }
        Simulate<false>(S,Actions_Played[0],Actions_Played[1]);
        array<bool,2> Finished_Race{false,false};
        for(int i=0;i<N;++i){
            for(int j=0;j<2;++j){
                if(S.p[i*N+j].lap==3){
                    Finished_Race[i]=true;
                }
            }
        }
        if(Finished_Race[0] || Finished_Race[1]){
            return Finished_Race[0]==Finished_Race[1]?-1:Finished_Race[0]?0:1;
        }
    }
    array<int,2> Distance{numeric_limits<int>::max(),numeric_limits<int>::max()};
    for(int i=0;i<2;++i){
        for(int j=i*2;j<(i+1)*2;++j){
            Distance[i]=min(Distance[i],CPs_Left(S.p[j],S.C));
        }
    }
    if(Distance[0]!=Distance[1]){
        return Distance[0]<Distance[1]?0:1;
    }
    return -1;
}

double Play_Round(const array<string,N> &Bot_Names){
    Map C=Generate_Map(generator);
    array<array<vec,2>,N> Spawns;
    //Spawn generation
    vec CP0_CP1{C[1]-C[0]};
    CP0_CP1.normalize();
    vec orth{CP0_CP1.y,-CP0_CP1.x};
    for(int i=0;i<N;++i){
        const int displacement{((i==0?1:3)*(Pod_Radius+100))};
        Spawns[i][0]=C[0]+orth*displacement;
        Spawns[i][1]=C[0]-orth*displacement;
        Spawns[i][0].round_vec();
        Spawns[i][1].round_vec();
    }
    array<int,N> winner;
    for(int i=0;i<N;++i){
        winner[i]=Play_Game(Bot_Names,C,Spawns);
        rotate(Spawns.begin(),Spawns.begin()+1,Spawns.end());
    }
    double points{0};
    for(const int &win:winner){
        if(win==-1){
            points+=0.5;
        }
        else if(win==0){
            points+=1.0;
        }
    }
    return points;
}

void StopArena(const int signum){
    stop=true;
}

int main(int argc,char **argv){
    if(argc<3){
        cerr << "Program takes 2 inputs, the names of the AIs fighting each other" << endl;
        return 0;
    }
    int N_Threads{1};
    if(argc>=4){//Optional N_Threads parameter
        N_Threads=min(2*omp_get_num_procs(),max(1,atoi(argv[3])));
        cerr << "Running " << N_Threads << " arena threads" << endl;
    }
    array<string,N> Bot_Names;
    for(int i=0;i<2;++i){
        Bot_Names[i]=argv[i+1];
    }
    cout << "Testing AI " << Bot_Names[0];
    for(int i=1;i<N;++i){
        cerr << " vs " << Bot_Names[i];
    }
    cerr << endl;
    for(int i=0;i<N;++i){//Check that AI binaries are present
        ifstream Test{Bot_Names[i].c_str()};
        if(!Test){
            cerr << Bot_Names[i] << " couldn't be found" << endl;
            return 0;
        }
        Test.close();
    }
    signal(SIGTERM,StopArena);//Register SIGTERM signal handler so the arena can cleanup when you kill it
    signal(SIGPIPE,SIG_IGN);//Ignore SIGPIPE to avoid the arena crashing when an AI crashes
    int games{0},draws{0};
    array<double,2> points{0,0};
    #pragma omp parallel num_threads(N_Threads) shared(games,points,Bot_Names)
    while(!stop){
        const double player_0_points{Play_Round(Bot_Names)};
        #pragma omp atomic
        points[0]+=player_0_points;
        #pragma omp atomic
        points[1]+=(2-player_0_points);
        #pragma omp atomic
        games+=2;
        const double p{static_cast<double>(points[0])/games};
        const double sigma{sqrt(p*(1-p)/games)};
        const double better{0.5+0.5*erf((p-0.5)/(sqrt(2)*sigma))};
        #pragma omp critical
        cout << "Wins:" << setprecision(4) << 100*p << "+-" << 100*sigma << "% Rounds " << games/2 << " " << better*100 << "% chance that " << Bot_Names[0] << " is better" << endl;
    }
}
