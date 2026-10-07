(()=>{/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */var cs={LEFT:0,MIDDLE:1,RIGHT:2,ROTATE:0,DOLLY:1,PAN:2},hs={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},Tf=0,pu=1,Af=2;var Us=1,Rf=2,Ir=3,Zi=0,ti=1,Mi=2,zt=0,Ps=1,mu=2,gu=3,_u=4,Zl=5;var Ci=100,Cf=101,Pf=102,If=103,Df=104,Ns=200,Lf=201,Uf=202,Nf=203,al=204,ll=205,Yo=206,Ff=207,$o=208,Of=209,Bf=210,zf=211,kf=212,Hf=213,Vf=214,cl=0,hl=1,ul=2,Is=3,dl=4,fl=5,pl=6,ml=7,Jl=0,Gf=1,Wf=2,en=0,Zo=1,Jo=2,jo=3,us=4,Ko=5,Qo=6,Fs=7;var xu=300,ds=301,Os=302,jl=303,Kl=304,ea=306,zi=1e3,un=1001,gl=1002,Ot=1003,Xf=1004;var ta=1005;var ei=1006,Ql=1007;var fs=1008;var ci=1009,vu=1010,yu=1011,Dr=1012,ec=1013,tn=1014,Hi=1015,ii=1016,tc=1017,ic=1018,ps=1020,Mu=35902,Su=35899,bu=1021,Eu=1022,Si=1023,fn=1026,_n=1027,nc=1028,sc=1029,ms=1030,rc=1031;var oc=1033,ia=33776,na=33777,sa=33778,ra=33779,ac=35840,lc=35841,cc=35842,hc=35843,uc=36196,dc=37492,fc=37496,pc=37488,mc=37489,oa=37490,gc=37491,_c=37808,xc=37809,vc=37810,yc=37811,Mc=37812,Sc=37813,bc=37814,Ec=37815,wc=37816,Tc=37817,Ac=37818,Rc=37819,Cc=37820,Pc=37821,Ic=36492,Dc=36494,Lc=36495,Uc=36283,Nc=36284,aa=36285,Fc=36286;var oo=2300,_l=2301,ol=2302,Qh=2303,eu=2400,tu=2401,iu=2402;var qf=3200;var Lr=0,Yf=1,On="",Lt="srgb",ao="srgb-linear",lo="linear",pt="srgb";var As=7680;var nu=519,$f=512,Zf=513,Jf=514,Oc=515,jf=516,Kf=517,Bc=518,Qf=519,xl=35044,xn=35048;var wu="300 es",$i=2e3,gr=2001;function Km(n){for(let e=n.length-1;e>=0;--e)if(n[e]>=65535)return!0;return!1}function Qm(n){return ArrayBuffer.isView(n)&&!(n instanceof DataView)}function co(n){return document.createElementNS("http://www.w3.org/1999/xhtml",n)}function ep(){let n=co("canvas");return n.style.display="block",n}var Bd={},_r=null;function ho(...n){let e="THREE."+n.shift();_r?_r("log",e,...n):console.log(e,...n)}function tp(n){let e=n[0];if(typeof e=="string"&&e.startsWith("TSL:")){let t=n[1];t&&t.isStackTrace?n[0]+=" "+t.getLocation():n[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return n}function $e(...n){n=tp(n);let e="THREE."+n.shift();if(_r)_r("warn",e,...n);else{let t=n[0];t&&t.isStackTrace?console.warn(t.getError(e)):console.warn(e,...n)}}function Ze(...n){n=tp(n);let e="THREE."+n.shift();if(_r)_r("error",e,...n);else{let t=n[0];t&&t.isStackTrace?console.error(t.getError(e)):console.error(e,...n)}}function Cs(...n){let e=n.join(" ");e in Bd||(Bd[e]=!0,$e(...n))}function ip(n,e,t){return new Promise(function(i,s){function r(){switch(n.clientWaitSync(e,n.SYNC_FLUSH_COMMANDS_BIT,0)){case n.WAIT_FAILED:s();break;case n.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:i()}}setTimeout(r,t)})}var np={[cl]:hl,[ul]:pl,[dl]:ml,[Is]:fl,[hl]:cl,[pl]:ul,[ml]:dl,[fl]:Is},Ji=class{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});let i=this._listeners;i[e]===void 0&&(i[e]=[]),i[e].indexOf(t)===-1&&i[e].push(t)}hasEventListener(e,t){let i=this._listeners;return i===void 0?!1:i[e]!==void 0&&i[e].indexOf(t)!==-1}removeEventListener(e,t){let i=this._listeners;if(i===void 0)return;let s=i[e];if(s!==void 0){let r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){let t=this._listeners;if(t===void 0)return;let i=t[e.type];if(i!==void 0){e.target=this;let s=i.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}},ai=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],zd=1234567,pr=Math.PI/180,xr=180/Math.PI;function dn(){let n=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(ai[n&255]+ai[n>>8&255]+ai[n>>16&255]+ai[n>>24&255]+"-"+ai[e&255]+ai[e>>8&255]+"-"+ai[e>>16&15|64]+ai[e>>24&255]+"-"+ai[t&63|128]+ai[t>>8&255]+"-"+ai[t>>16&255]+ai[t>>24&255]+ai[i&255]+ai[i>>8&255]+ai[i>>16&255]+ai[i>>24&255]).toLowerCase()}function je(n,e,t){return Math.max(e,Math.min(t,n))}function Tu(n,e){return(n%e+e)%e}function eg(n,e,t,i,s){return i+(n-e)*(s-i)/(t-e)}function tg(n,e,t){return n!==e?(t-n)/(e-n):0}function no(n,e,t){return(1-t)*n+t*e}function ig(n,e,t,i){return no(n,e,1-Math.exp(-t*i))}function ng(n,e=1){return e-Math.abs(Tu(n,e*2)-e)}function sg(n,e,t){return n<=e?0:n>=t?1:(n=(n-e)/(t-e),n*n*(3-2*n))}function rg(n,e,t){return n<=e?0:n>=t?1:(n=(n-e)/(t-e),n*n*n*(n*(n*6-15)+10))}function og(n,e){return n+Math.floor(Math.random()*(e-n+1))}function ag(n,e){return n+Math.random()*(e-n)}function lg(n){return n*(.5-Math.random())}function cg(n){n!==void 0&&(zd=n);let e=zd+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function hg(n){return n*pr}function ug(n){return n*xr}function dg(n){return(n&n-1)===0&&n!==0}function fg(n){return Math.pow(2,Math.ceil(Math.log(n)/Math.LN2))}function pg(n){return Math.pow(2,Math.floor(Math.log(n)/Math.LN2))}function mg(n,e,t,i,s){let r=Math.cos,o=Math.sin,a=r(t/2),c=o(t/2),l=r((e+i)/2),h=o((e+i)/2),u=r((e-i)/2),d=o((e-i)/2),f=r((i-e)/2),g=o((i-e)/2);switch(s){case"XYX":n.set(a*h,c*u,c*d,a*l);break;case"YZY":n.set(c*d,a*h,c*u,a*l);break;case"ZXZ":n.set(c*u,c*d,a*h,a*l);break;case"XZX":n.set(a*h,c*g,c*f,a*l);break;case"YXY":n.set(c*f,a*h,c*g,a*l);break;case"ZYZ":n.set(c*g,c*f,a*h,a*l);break;default:$e("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function Yi(n,e){switch(e.constructor){case Float32Array:return n;case Uint32Array:return n/4294967295;case Uint16Array:return n/65535;case Uint8Array:return n/255;case Int32Array:return Math.max(n/2147483647,-1);case Int16Array:return Math.max(n/32767,-1);case Int8Array:return Math.max(n/127,-1);default:throw new Error("THREE.MathUtils: Invalid component type.")}}function gt(n,e){switch(e.constructor){case Float32Array:return n;case Uint32Array:return Math.round(n*4294967295);case Uint16Array:return Math.round(n*65535);case Uint8Array:return Math.round(n*255);case Int32Array:return Math.round(n*2147483647);case Int16Array:return Math.round(n*32767);case Int8Array:return Math.round(n*127);default:throw new Error("THREE.MathUtils: Invalid component type.")}}var Vt={DEG2RAD:pr,RAD2DEG:xr,generateUUID:dn,clamp:je,euclideanModulo:Tu,mapLinear:eg,inverseLerp:tg,lerp:no,damp:ig,pingpong:ng,smoothstep:sg,smootherstep:rg,randInt:og,randFloat:ag,randFloatSpread:lg,seededRandom:cg,degToRad:hg,radToDeg:ug,isPowerOfTwo:dg,ceilPowerOfTwo:fg,floorPowerOfTwo:pg,setQuaternionFromProperEuler:mg,normalize:gt,denormalize:Yi},Du=class Du{constructor(e=0,t=0){this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("THREE.Vector2: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("THREE.Vector2: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){let t=this.x,i=this.y,s=e.elements;return this.x=s[0]*t+s[3]*i+s[6],this.y=s[1]*t+s[4]*i+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=je(this.x,e.x,t.x),this.y=je(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=je(this.x,e,t),this.y=je(this.y,e,t),this}clampLength(e,t){let i=this.length();return this.divideScalar(i||1).multiplyScalar(je(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let i=this.dot(e)/t;return Math.acos(je(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,i=this.y-e.y;return t*t+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){let i=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*i-o*s+e.x,this.y=r*s+o*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}};Du.prototype.isVector2=!0;var $=Du,Pi=class{constructor(e=0,t=0,i=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=i,this._w=s}static slerpFlat(e,t,i,s,r,o,a){let c=i[s+0],l=i[s+1],h=i[s+2],u=i[s+3],d=r[o+0],f=r[o+1],g=r[o+2],x=r[o+3];if(u!==x||c!==d||l!==f||h!==g){let p=c*d+l*f+h*g+u*x;p<0&&(d=-d,f=-f,g=-g,x=-x,p=-p);let m=1-a;if(p<.9995){let M=Math.acos(p),b=Math.sin(M);m=Math.sin(m*M)/b,a=Math.sin(a*M)/b,c=c*m+d*a,l=l*m+f*a,h=h*m+g*a,u=u*m+x*a}else{c=c*m+d*a,l=l*m+f*a,h=h*m+g*a,u=u*m+x*a;let M=1/Math.sqrt(c*c+l*l+h*h+u*u);c*=M,l*=M,h*=M,u*=M}}e[t]=c,e[t+1]=l,e[t+2]=h,e[t+3]=u}static multiplyQuaternionsFlat(e,t,i,s,r,o){let a=i[s],c=i[s+1],l=i[s+2],h=i[s+3],u=r[o],d=r[o+1],f=r[o+2],g=r[o+3];return e[t]=a*g+h*u+c*f-l*d,e[t+1]=c*g+h*d+l*u-a*f,e[t+2]=l*g+h*f+a*d-c*u,e[t+3]=h*g-a*u-c*d-l*f,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,i,s){return this._x=e,this._y=t,this._z=i,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){let i=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,c=Math.sin,l=a(i/2),h=a(s/2),u=a(r/2),d=c(i/2),f=c(s/2),g=c(r/2);switch(o){case"XYZ":this._x=d*h*u+l*f*g,this._y=l*f*u-d*h*g,this._z=l*h*g+d*f*u,this._w=l*h*u-d*f*g;break;case"YXZ":this._x=d*h*u+l*f*g,this._y=l*f*u-d*h*g,this._z=l*h*g-d*f*u,this._w=l*h*u+d*f*g;break;case"ZXY":this._x=d*h*u-l*f*g,this._y=l*f*u+d*h*g,this._z=l*h*g+d*f*u,this._w=l*h*u-d*f*g;break;case"ZYX":this._x=d*h*u-l*f*g,this._y=l*f*u+d*h*g,this._z=l*h*g-d*f*u,this._w=l*h*u+d*f*g;break;case"YZX":this._x=d*h*u+l*f*g,this._y=l*f*u+d*h*g,this._z=l*h*g-d*f*u,this._w=l*h*u-d*f*g;break;case"XZY":this._x=d*h*u-l*f*g,this._y=l*f*u-d*h*g,this._z=l*h*g+d*f*u,this._w=l*h*u+d*f*g;break;default:$e("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){let i=t/2,s=Math.sin(i);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(e){let t=e.elements,i=t[0],s=t[4],r=t[8],o=t[1],a=t[5],c=t[9],l=t[2],h=t[6],u=t[10],d=i+a+u;if(d>0){let f=.5/Math.sqrt(d+1);this._w=.25/f,this._x=(h-c)*f,this._y=(r-l)*f,this._z=(o-s)*f}else if(i>a&&i>u){let f=2*Math.sqrt(1+i-a-u);this._w=(h-c)/f,this._x=.25*f,this._y=(s+o)/f,this._z=(r+l)/f}else if(a>u){let f=2*Math.sqrt(1+a-i-u);this._w=(r-l)/f,this._x=(s+o)/f,this._y=.25*f,this._z=(c+h)/f}else{let f=2*Math.sqrt(1+u-i-a);this._w=(o-s)/f,this._x=(r+l)/f,this._y=(c+h)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let i=e.dot(t)+1;return i<1e-8?(i=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=i):(this._x=0,this._y=-e.z,this._z=e.y,this._w=i)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=i),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(je(this.dot(e),-1,1)))}rotateTowards(e,t){let i=this.angleTo(e);if(i===0)return this;let s=Math.min(1,t/i);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){let i=e._x,s=e._y,r=e._z,o=e._w,a=t._x,c=t._y,l=t._z,h=t._w;return this._x=i*h+o*a+s*l-r*c,this._y=s*h+o*c+r*a-i*l,this._z=r*h+o*l+i*c-s*a,this._w=o*h-i*a-s*c-r*l,this._onChangeCallback(),this}slerp(e,t){let i=e._x,s=e._y,r=e._z,o=e._w,a=this.dot(e);a<0&&(i=-i,s=-s,r=-r,o=-o,a=-a);let c=1-t;if(a<.9995){let l=Math.acos(a),h=Math.sin(l);c=Math.sin(c*l)/h,t=Math.sin(t*l)/h,this._x=this._x*c+i*t,this._y=this._y*c+s*t,this._z=this._z*c+r*t,this._w=this._w*c+o*t,this._onChangeCallback()}else this._x=this._x*c+i*t,this._y=this._y*c+s*t,this._z=this._z*c+r*t,this._w=this._w*c+o*t,this.normalize();return this}slerpQuaternions(e,t,i){return this.copy(e).slerp(t,i)}random(){let e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),i=Math.random(),s=Math.sqrt(1-i),r=Math.sqrt(i);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},Lu=class Lu{constructor(e=0,t=0,i=0){this.x=e,this.y=t,this.z=i}set(e,t,i){return i===void 0&&(i=this.z),this.x=e,this.y=t,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("THREE.Vector3: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("THREE.Vector3: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(kd.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(kd.setFromAxisAngle(e,t))}applyMatrix3(e){let t=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*i+r[6]*s,this.y=r[1]*t+r[4]*i+r[7]*s,this.z=r[2]*t+r[5]*i+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){let t=this.x,i=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*i+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*i+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*i+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*i+r[10]*s+r[14])*o,this}applyQuaternion(e){let t=this.x,i=this.y,s=this.z,r=e.x,o=e.y,a=e.z,c=e.w,l=2*(o*s-a*i),h=2*(a*t-r*s),u=2*(r*i-o*t);return this.x=t+c*l+o*u-a*h,this.y=i+c*h+a*l-r*u,this.z=s+c*u+r*h-o*l,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){let t=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*i+r[8]*s,this.y=r[1]*t+r[5]*i+r[9]*s,this.z=r[2]*t+r[6]*i+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=je(this.x,e.x,t.x),this.y=je(this.y,e.y,t.y),this.z=je(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=je(this.x,e,t),this.y=je(this.y,e,t),this.z=je(this.z,e,t),this}clampLength(e,t){let i=this.length();return this.divideScalar(i||1).multiplyScalar(je(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this.z=e.z+(t.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){let i=e.x,s=e.y,r=e.z,o=t.x,a=t.y,c=t.z;return this.x=s*c-r*a,this.y=r*o-i*c,this.z=i*a-s*o,this}projectOnVector(e){let t=e.lengthSq();if(t===0)return this.set(0,0,0);let i=e.dot(this)/t;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return Eh.copy(this).projectOnVector(e),this.sub(Eh)}reflect(e){return this.sub(Eh.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let i=this.dot(e)/t;return Math.acos(je(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,i=this.y-e.y,s=this.z-e.z;return t*t+i*i+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,i){let s=Math.sin(t)*e;return this.x=s*Math.sin(i),this.y=Math.cos(t)*e,this.z=s*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,i){return this.x=e*Math.sin(t),this.y=i,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){let t=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=i,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let e=Math.random()*Math.PI*2,t=Math.random()*2-1,i=Math.sqrt(1-t*t);return this.x=i*Math.cos(e),this.y=t,this.z=i*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}};Lu.prototype.isVector3=!0;var P=Lu,Eh=new P,kd=new Pi,Uu=class Uu{constructor(e,t,i,s,r,o,a,c,l){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,i,s,r,o,a,c,l)}set(e,t,i,s,r,o,a,c,l){let h=this.elements;return h[0]=e,h[1]=s,h[2]=a,h[3]=t,h[4]=r,h[5]=c,h[6]=i,h[7]=o,h[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){let t=this.elements,i=e.elements;return t[0]=i[0],t[1]=i[1],t[2]=i[2],t[3]=i[3],t[4]=i[4],t[5]=i[5],t[6]=i[6],t[7]=i[7],t[8]=i[8],this}extractBasis(e,t,i){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){let t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let i=e.elements,s=t.elements,r=this.elements,o=i[0],a=i[3],c=i[6],l=i[1],h=i[4],u=i[7],d=i[2],f=i[5],g=i[8],x=s[0],p=s[3],m=s[6],M=s[1],b=s[4],y=s[7],T=s[2],S=s[5],A=s[8];return r[0]=o*x+a*M+c*T,r[3]=o*p+a*b+c*S,r[6]=o*m+a*y+c*A,r[1]=l*x+h*M+u*T,r[4]=l*p+h*b+u*S,r[7]=l*m+h*y+u*A,r[2]=d*x+f*M+g*T,r[5]=d*p+f*b+g*S,r[8]=d*m+f*y+g*A,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){let e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],o=e[4],a=e[5],c=e[6],l=e[7],h=e[8];return t*o*h-t*a*l-i*r*h+i*a*c+s*r*l-s*o*c}invert(){let e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],o=e[4],a=e[5],c=e[6],l=e[7],h=e[8],u=h*o-a*l,d=a*c-h*r,f=l*r-o*c,g=t*u+i*d+s*f;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);let x=1/g;return e[0]=u*x,e[1]=(s*l-h*i)*x,e[2]=(a*i-s*o)*x,e[3]=d*x,e[4]=(h*t-s*c)*x,e[5]=(s*r-a*t)*x,e[6]=f*x,e[7]=(i*c-l*t)*x,e[8]=(o*t-i*r)*x,this}transpose(){let e,t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){let t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,i,s,r,o,a){let c=Math.cos(r),l=Math.sin(r);return this.set(i*c,i*l,-i*(c*o+l*a)+o+e,-s*l,s*c,-s*(-l*o+c*a)+a+t,0,0,1),this}scale(e,t){return Cs("Matrix3: .scale() is deprecated. Use .makeScale() instead."),this.premultiply(wh.makeScale(e,t)),this}rotate(e){return Cs("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."),this.premultiply(wh.makeRotation(-e)),this}translate(e,t){return Cs("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."),this.premultiply(wh.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){let t=Math.cos(e),i=Math.sin(e);return this.set(t,-i,0,i,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){let t=this.elements,i=e.elements;for(let s=0;s<9;s++)if(t[s]!==i[s])return!1;return!0}fromArray(e,t=0){for(let i=0;i<9;i++)this.elements[i]=e[i+t];return this}toArray(e=[],t=0){let i=this.elements;return e[t]=i[0],e[t+1]=i[1],e[t+2]=i[2],e[t+3]=i[3],e[t+4]=i[4],e[t+5]=i[5],e[t+6]=i[6],e[t+7]=i[7],e[t+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}};Uu.prototype.isMatrix3=!0;var tt=Uu,wh=new tt,Hd=new tt().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Vd=new tt().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function gg(){let n={enabled:!0,workingColorSpace:ao,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===pt&&(s.r=In(s.r),s.g=In(s.g),s.b=In(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===pt&&(s.r=mr(s.r),s.g=mr(s.g),s.b=mr(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===On?lo:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return Cs("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),n.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return Cs("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),n.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],i=[.3127,.329];return n.define({[ao]:{primaries:e,whitePoint:i,transfer:lo,toXYZ:Hd,fromXYZ:Vd,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Lt},outputColorSpaceConfig:{drawingBufferColorSpace:Lt}},[Lt]:{primaries:e,whitePoint:i,transfer:pt,toXYZ:Hd,fromXYZ:Vd,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Lt}}}),n}var ht=gg();function In(n){return n<.04045?n*.0773993808:Math.pow(n*.9478672986+.0521327014,2.4)}function mr(n){return n<.0031308?n*12.92:1.055*Math.pow(n,.41666)-.055}var Zs,vl=class{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let i;if(e instanceof HTMLCanvasElement)i=e;else{Zs===void 0&&(Zs=co("canvas")),Zs.width=e.width,Zs.height=e.height;let s=Zs.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),i=Zs}return i.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){let t=co("canvas");t.width=e.width,t.height=e.height;let i=t.getContext("2d");i.drawImage(e,0,0,e.width,e.height);let s=i.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=In(r[o]/255)*255;return i.putImageData(s,0,0),t}else if(e.data){let t=e.data.slice(0);for(let i=0;i<t.length;i++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[i]=Math.floor(In(t[i]/255)*255):t[i]=In(t[i]);return{data:t,width:e.width,height:e.height}}else return $e("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}},_g=0,vr=class{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:_g++}),this.uuid=dn(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){let t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):typeof VideoFrame<"u"&&t instanceof VideoFrame?e.set(t.displayWidth,t.displayHeight,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){let t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];let i={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(Th(s[o].image)):r.push(Th(s[o]))}else r=Th(s);i.url=r}return t||(e.images[this.uuid]=i),i}};function Th(n){return typeof HTMLImageElement<"u"&&n instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&n instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&n instanceof ImageBitmap?vl.getDataURL(n):n.data?{data:Array.from(n.data),width:n.width,height:n.height,type:n.data.constructor.name}:($e("Texture: Unable to serialize Texture."),{})}var xg=0,Ah=new P,fi=class n extends Ji{constructor(e=n.DEFAULT_IMAGE,t=n.DEFAULT_MAPPING,i=un,s=un,r=ei,o=fs,a=Si,c=ci,l=n.DEFAULT_ANISOTROPY,h=On){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:xg++}),this.uuid=dn(),this.name="",this.source=new vr(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=i,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=l,this.format=a,this.internalFormat=null,this.type=c,this.offset=new $(0,0),this.repeat=new $(1,1),this.center=new $(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new tt,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0,this.normalized=!1}get width(){return this.source.getSize(Ah).x}get height(){return this.source.getSize(Ah).y}get depth(){return this.source.getSize(Ah).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(let t in e){let i=e[t];if(i===void 0){$e(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}let s=this[t];if(s===void 0){$e(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&i&&s.isVector2&&i.isVector2||s&&i&&s.isVector3&&i.isVector3||s&&i&&s.isMatrix3&&i.isMatrix3?s.copy(i):this[t]=i}}toJSON(e){let t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];let i={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),t||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==xu)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case zi:e.x=e.x-Math.floor(e.x);break;case un:e.x=e.x<0?0:1;break;case gl:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case zi:e.y=e.y-Math.floor(e.y);break;case un:e.y=e.y<0?0:1;break;case gl:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}};fi.DEFAULT_IMAGE=null;fi.DEFAULT_MAPPING=xu;fi.DEFAULT_ANISOTROPY=1;var Nu=class Nu{constructor(e=0,t=0,i=0,s=1){this.x=e,this.y=t,this.z=i,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,i,s){return this.x=e,this.y=t,this.z=i,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("THREE.Vector4: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("THREE.Vector4: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){let t=this.x,i=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*i+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*i+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*i+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*i+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);let t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,i,s,r,c=e.elements,l=c[0],h=c[4],u=c[8],d=c[1],f=c[5],g=c[9],x=c[2],p=c[6],m=c[10];if(Math.abs(h-d)<.01&&Math.abs(u-x)<.01&&Math.abs(g-p)<.01){if(Math.abs(h+d)<.1&&Math.abs(u+x)<.1&&Math.abs(g+p)<.1&&Math.abs(l+f+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;let b=(l+1)/2,y=(f+1)/2,T=(m+1)/2,S=(h+d)/4,A=(u+x)/4,_=(g+p)/4;return b>y&&b>T?b<.01?(i=0,s=.707106781,r=.707106781):(i=Math.sqrt(b),s=S/i,r=A/i):y>T?y<.01?(i=.707106781,s=0,r=.707106781):(s=Math.sqrt(y),i=S/s,r=_/s):T<.01?(i=.707106781,s=.707106781,r=0):(r=Math.sqrt(T),i=A/r,s=_/r),this.set(i,s,r,t),this}let M=Math.sqrt((p-g)*(p-g)+(u-x)*(u-x)+(d-h)*(d-h));return Math.abs(M)<.001&&(M=1),this.x=(p-g)/M,this.y=(u-x)/M,this.z=(d-h)/M,this.w=Math.acos((l+f+m-1)/2),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=je(this.x,e.x,t.x),this.y=je(this.y,e.y,t.y),this.z=je(this.z,e.z,t.z),this.w=je(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=je(this.x,e,t),this.y=je(this.y,e,t),this.z=je(this.z,e,t),this.w=je(this.w,e,t),this}clampLength(e,t){let i=this.length();return this.divideScalar(i||1).multiplyScalar(je(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this.z=e.z+(t.z-e.z)*i,this.w=e.w+(t.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}};Nu.prototype.isVector4=!0;var mt=Nu,yl=class extends Ji{constructor(e=1,t=1,i={}){super(),i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:ei,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},i),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=i.depth,this.scissor=new mt(0,0,e,t),this.scissorTest=!1,this.viewport=new mt(0,0,e,t),this.textures=[];let s={width:e,height:t,depth:i.depth},r=new fi(s),o=i.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(i),this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=i.depthTexture,this.samples=i.samples,this.multiview=i.multiview,this.useArrayDepthTexture=i.useArrayDepthTexture}_setTextureOptions(e={}){let t={minFilter:ei,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let i=0;i<this.textures.length;i++)this.textures[i].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,i=1){if(this.width!==e||this.height!==t||this.depth!==i){this.width=e,this.height=t,this.depth=i;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=i,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,i=e.textures.length;t<i;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;let s=Object.assign({},e.textures[t].image);this.textures[t].source=new vr(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this.multiview=e.multiview,this.useArrayDepthTexture=e.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}},Ht=class extends yl{constructor(e=1,t=1,i={}){super(e,t,i),this.isWebGLRenderTarget=!0}},uo=class extends fi{constructor(e=null,t=1,i=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:i,depth:s},this.magFilter=Ot,this.minFilter=Ot,this.wrapR=un,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}};var Ml=class extends fi{constructor(e=null,t=1,i=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:i,depth:s},this.magFilter=Ot,this.minFilter=Ot,this.wrapR=un,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var $l=class $l{constructor(e,t,i,s,r,o,a,c,l,h,u,d,f,g,x,p){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,i,s,r,o,a,c,l,h,u,d,f,g,x,p)}set(e,t,i,s,r,o,a,c,l,h,u,d,f,g,x,p){let m=this.elements;return m[0]=e,m[4]=t,m[8]=i,m[12]=s,m[1]=r,m[5]=o,m[9]=a,m[13]=c,m[2]=l,m[6]=h,m[10]=u,m[14]=d,m[3]=f,m[7]=g,m[11]=x,m[15]=p,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new $l().fromArray(this.elements)}copy(e){let t=this.elements,i=e.elements;return t[0]=i[0],t[1]=i[1],t[2]=i[2],t[3]=i[3],t[4]=i[4],t[5]=i[5],t[6]=i[6],t[7]=i[7],t[8]=i[8],t[9]=i[9],t[10]=i[10],t[11]=i[11],t[12]=i[12],t[13]=i[13],t[14]=i[14],t[15]=i[15],this}copyPosition(e){let t=this.elements,i=e.elements;return t[12]=i[12],t[13]=i[13],t[14]=i[14],this}setFromMatrix3(e){let t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,i){return this.determinantAffine()===0?(e.set(1,0,0),t.set(0,1,0),i.set(0,0,1),this):(e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this)}makeBasis(e,t,i){return this.set(e.x,t.x,i.x,0,e.y,t.y,i.y,0,e.z,t.z,i.z,0,0,0,0,1),this}extractRotation(e){if(e.determinantAffine()===0)return this.identity();let t=this.elements,i=e.elements,s=1/Js.setFromMatrixColumn(e,0).length(),r=1/Js.setFromMatrixColumn(e,1).length(),o=1/Js.setFromMatrixColumn(e,2).length();return t[0]=i[0]*s,t[1]=i[1]*s,t[2]=i[2]*s,t[3]=0,t[4]=i[4]*r,t[5]=i[5]*r,t[6]=i[6]*r,t[7]=0,t[8]=i[8]*o,t[9]=i[9]*o,t[10]=i[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){let t=this.elements,i=e.x,s=e.y,r=e.z,o=Math.cos(i),a=Math.sin(i),c=Math.cos(s),l=Math.sin(s),h=Math.cos(r),u=Math.sin(r);if(e.order==="XYZ"){let d=o*h,f=o*u,g=a*h,x=a*u;t[0]=c*h,t[4]=-c*u,t[8]=l,t[1]=f+g*l,t[5]=d-x*l,t[9]=-a*c,t[2]=x-d*l,t[6]=g+f*l,t[10]=o*c}else if(e.order==="YXZ"){let d=c*h,f=c*u,g=l*h,x=l*u;t[0]=d+x*a,t[4]=g*a-f,t[8]=o*l,t[1]=o*u,t[5]=o*h,t[9]=-a,t[2]=f*a-g,t[6]=x+d*a,t[10]=o*c}else if(e.order==="ZXY"){let d=c*h,f=c*u,g=l*h,x=l*u;t[0]=d-x*a,t[4]=-o*u,t[8]=g+f*a,t[1]=f+g*a,t[5]=o*h,t[9]=x-d*a,t[2]=-o*l,t[6]=a,t[10]=o*c}else if(e.order==="ZYX"){let d=o*h,f=o*u,g=a*h,x=a*u;t[0]=c*h,t[4]=g*l-f,t[8]=d*l+x,t[1]=c*u,t[5]=x*l+d,t[9]=f*l-g,t[2]=-l,t[6]=a*c,t[10]=o*c}else if(e.order==="YZX"){let d=o*c,f=o*l,g=a*c,x=a*l;t[0]=c*h,t[4]=x-d*u,t[8]=g*u+f,t[1]=u,t[5]=o*h,t[9]=-a*h,t[2]=-l*h,t[6]=f*u+g,t[10]=d-x*u}else if(e.order==="XZY"){let d=o*c,f=o*l,g=a*c,x=a*l;t[0]=c*h,t[4]=-u,t[8]=l*h,t[1]=d*u+x,t[5]=o*h,t[9]=f*u-g,t[2]=g*u-f,t[6]=a*h,t[10]=x*u+d}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(vg,e,yg)}lookAt(e,t,i){let s=this.elements;return Ai.subVectors(e,t),Ai.lengthSq()===0&&(Ai.z=1),Ai.normalize(),jn.crossVectors(i,Ai),jn.lengthSq()===0&&(Math.abs(i.z)===1?Ai.x+=1e-4:Ai.z+=1e-4,Ai.normalize(),jn.crossVectors(i,Ai)),jn.normalize(),Ta.crossVectors(Ai,jn),s[0]=jn.x,s[4]=Ta.x,s[8]=Ai.x,s[1]=jn.y,s[5]=Ta.y,s[9]=Ai.y,s[2]=jn.z,s[6]=Ta.z,s[10]=Ai.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let i=e.elements,s=t.elements,r=this.elements,o=i[0],a=i[4],c=i[8],l=i[12],h=i[1],u=i[5],d=i[9],f=i[13],g=i[2],x=i[6],p=i[10],m=i[14],M=i[3],b=i[7],y=i[11],T=i[15],S=s[0],A=s[4],_=s[8],E=s[12],C=s[1],I=s[5],L=s[9],V=s[13],q=s[2],N=s[6],Y=s[10],X=s[14],ne=s[3],ie=s[7],ge=s[11],ue=s[15];return r[0]=o*S+a*C+c*q+l*ne,r[4]=o*A+a*I+c*N+l*ie,r[8]=o*_+a*L+c*Y+l*ge,r[12]=o*E+a*V+c*X+l*ue,r[1]=h*S+u*C+d*q+f*ne,r[5]=h*A+u*I+d*N+f*ie,r[9]=h*_+u*L+d*Y+f*ge,r[13]=h*E+u*V+d*X+f*ue,r[2]=g*S+x*C+p*q+m*ne,r[6]=g*A+x*I+p*N+m*ie,r[10]=g*_+x*L+p*Y+m*ge,r[14]=g*E+x*V+p*X+m*ue,r[3]=M*S+b*C+y*q+T*ne,r[7]=M*A+b*I+y*N+T*ie,r[11]=M*_+b*L+y*Y+T*ge,r[15]=M*E+b*V+y*X+T*ue,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){let e=this.elements,t=e[0],i=e[4],s=e[8],r=e[12],o=e[1],a=e[5],c=e[9],l=e[13],h=e[2],u=e[6],d=e[10],f=e[14],g=e[3],x=e[7],p=e[11],m=e[15],M=c*f-l*d,b=a*f-l*u,y=a*d-c*u,T=o*f-l*h,S=o*d-c*h,A=o*u-a*h;return t*(x*M-p*b+m*y)-i*(g*M-p*T+m*S)+s*(g*b-x*T+m*A)-r*(g*y-x*S+p*A)}determinantAffine(){let e=this.elements,t=e[0],i=e[4],s=e[8],r=e[1],o=e[5],a=e[9],c=e[2],l=e[6],h=e[10];return t*(o*h-a*l)-i*(r*h-a*c)+s*(r*l-o*c)}transpose(){let e=this.elements,t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,i){let s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=i),this}invert(){let e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],o=e[4],a=e[5],c=e[6],l=e[7],h=e[8],u=e[9],d=e[10],f=e[11],g=e[12],x=e[13],p=e[14],m=e[15],M=t*a-i*o,b=t*c-s*o,y=t*l-r*o,T=i*c-s*a,S=i*l-r*a,A=s*l-r*c,_=h*x-u*g,E=h*p-d*g,C=h*m-f*g,I=u*p-d*x,L=u*m-f*x,V=d*m-f*p,q=M*V-b*L+y*I+T*C-S*E+A*_;if(q===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let N=1/q;return e[0]=(a*V-c*L+l*I)*N,e[1]=(s*L-i*V-r*I)*N,e[2]=(x*A-p*S+m*T)*N,e[3]=(d*S-u*A-f*T)*N,e[4]=(c*C-o*V-l*E)*N,e[5]=(t*V-s*C+r*E)*N,e[6]=(p*y-g*A-m*b)*N,e[7]=(h*A-d*y+f*b)*N,e[8]=(o*L-a*C+l*_)*N,e[9]=(i*C-t*L-r*_)*N,e[10]=(g*S-x*y+m*M)*N,e[11]=(u*y-h*S-f*M)*N,e[12]=(a*E-o*I-c*_)*N,e[13]=(t*I-i*E+s*_)*N,e[14]=(x*b-g*T-p*M)*N,e[15]=(h*T-u*b+d*M)*N,this}scale(e){let t=this.elements,i=e.x,s=e.y,r=e.z;return t[0]*=i,t[4]*=s,t[8]*=r,t[1]*=i,t[5]*=s,t[9]*=r,t[2]*=i,t[6]*=s,t[10]*=r,t[3]*=i,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){let e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,i,s))}makeTranslation(e,t,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,i,0,0,0,1),this}makeRotationX(e){let t=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,t,-i,0,0,i,t,0,0,0,0,1),this}makeRotationY(e){let t=Math.cos(e),i=Math.sin(e);return this.set(t,0,i,0,0,1,0,0,-i,0,t,0,0,0,0,1),this}makeRotationZ(e){let t=Math.cos(e),i=Math.sin(e);return this.set(t,-i,0,0,i,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){let i=Math.cos(t),s=Math.sin(t),r=1-i,o=e.x,a=e.y,c=e.z,l=r*o,h=r*a;return this.set(l*o+i,l*a-s*c,l*c+s*a,0,l*a+s*c,h*a+i,h*c-s*o,0,l*c-s*a,h*c+s*o,r*c*c+i,0,0,0,0,1),this}makeScale(e,t,i){return this.set(e,0,0,0,0,t,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,t,i,s,r,o){return this.set(1,i,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,i){let s=this.elements,r=t._x,o=t._y,a=t._z,c=t._w,l=r+r,h=o+o,u=a+a,d=r*l,f=r*h,g=r*u,x=o*h,p=o*u,m=a*u,M=c*l,b=c*h,y=c*u,T=i.x,S=i.y,A=i.z;return s[0]=(1-(x+m))*T,s[1]=(f+y)*T,s[2]=(g-b)*T,s[3]=0,s[4]=(f-y)*S,s[5]=(1-(d+m))*S,s[6]=(p+M)*S,s[7]=0,s[8]=(g+b)*A,s[9]=(p-M)*A,s[10]=(1-(d+x))*A,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,i){let s=this.elements;e.x=s[12],e.y=s[13],e.z=s[14];let r=this.determinantAffine();if(r===0)return i.set(1,1,1),t.identity(),this;let o=Js.set(s[0],s[1],s[2]).length(),a=Js.set(s[4],s[5],s[6]).length(),c=Js.set(s[8],s[9],s[10]).length();r<0&&(o=-o),Wi.copy(this);let l=1/o,h=1/a,u=1/c;return Wi.elements[0]*=l,Wi.elements[1]*=l,Wi.elements[2]*=l,Wi.elements[4]*=h,Wi.elements[5]*=h,Wi.elements[6]*=h,Wi.elements[8]*=u,Wi.elements[9]*=u,Wi.elements[10]*=u,t.setFromRotationMatrix(Wi),i.x=o,i.y=a,i.z=c,this}makePerspective(e,t,i,s,r,o,a=$i,c=!1){let l=this.elements,h=2*r/(t-e),u=2*r/(i-s),d=(t+e)/(t-e),f=(i+s)/(i-s),g,x;if(c)g=r/(o-r),x=o*r/(o-r);else if(a===$i)g=-(o+r)/(o-r),x=-2*o*r/(o-r);else if(a===gr)g=-o/(o-r),x=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return l[0]=h,l[4]=0,l[8]=d,l[12]=0,l[1]=0,l[5]=u,l[9]=f,l[13]=0,l[2]=0,l[6]=0,l[10]=g,l[14]=x,l[3]=0,l[7]=0,l[11]=-1,l[15]=0,this}makeOrthographic(e,t,i,s,r,o,a=$i,c=!1){let l=this.elements,h=2/(t-e),u=2/(i-s),d=-(t+e)/(t-e),f=-(i+s)/(i-s),g,x;if(c)g=1/(o-r),x=o/(o-r);else if(a===$i)g=-2/(o-r),x=-(o+r)/(o-r);else if(a===gr)g=-1/(o-r),x=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return l[0]=h,l[4]=0,l[8]=0,l[12]=d,l[1]=0,l[5]=u,l[9]=0,l[13]=f,l[2]=0,l[6]=0,l[10]=g,l[14]=x,l[3]=0,l[7]=0,l[11]=0,l[15]=1,this}equals(e){let t=this.elements,i=e.elements;for(let s=0;s<16;s++)if(t[s]!==i[s])return!1;return!0}fromArray(e,t=0){for(let i=0;i<16;i++)this.elements[i]=e[i+t];return this}toArray(e=[],t=0){let i=this.elements;return e[t]=i[0],e[t+1]=i[1],e[t+2]=i[2],e[t+3]=i[3],e[t+4]=i[4],e[t+5]=i[5],e[t+6]=i[6],e[t+7]=i[7],e[t+8]=i[8],e[t+9]=i[9],e[t+10]=i[10],e[t+11]=i[11],e[t+12]=i[12],e[t+13]=i[13],e[t+14]=i[14],e[t+15]=i[15],e}};$l.prototype.isMatrix4=!0;var rt=$l,Js=new P,Wi=new rt,vg=new P(0,0,0),yg=new P(1,1,1),jn=new P,Ta=new P,Ai=new P,Gd=new rt,Wd=new Pi,Ii=class n{constructor(e=0,t=0,i=0,s=n.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=i,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,i,s=this._order){return this._x=e,this._y=t,this._z=i,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,i=!0){let s=e.elements,r=s[0],o=s[4],a=s[8],c=s[1],l=s[5],h=s[9],u=s[2],d=s[6],f=s[10];switch(t){case"XYZ":this._y=Math.asin(je(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-h,f),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,l),this._z=0);break;case"YXZ":this._x=Math.asin(-je(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(a,f),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-u,r),this._z=0);break;case"ZXY":this._x=Math.asin(je(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-u,f),this._z=Math.atan2(-o,l)):(this._y=0,this._z=Math.atan2(c,r));break;case"ZYX":this._y=Math.asin(-je(u,-1,1)),Math.abs(u)<.9999999?(this._x=Math.atan2(d,f),this._z=Math.atan2(c,r)):(this._x=0,this._z=Math.atan2(-o,l));break;case"YZX":this._z=Math.asin(je(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-h,l),this._y=Math.atan2(-u,r)):(this._x=0,this._y=Math.atan2(a,f));break;case"XZY":this._z=Math.asin(-je(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,l),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-h,f),this._y=0);break;default:$e("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,i){return Gd.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Gd,t,i)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Wd.setFromEuler(this),this.setFromQuaternion(Wd,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};Ii.DEFAULT_ORDER="XYZ";var yr=class{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}},Mg=0,Xd=new P,js=new Pi,Tn=new rt,Aa=new P,qr=new P,Sg=new P,bg=new Pi,qd=new P(1,0,0),Yd=new P(0,1,0),$d=new P(0,0,1),Zd={type:"added"},Eg={type:"removed"},Ks={type:"childadded",child:null},Rh={type:"childremoved",child:null},ft=class n extends Ji{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:Mg++}),this.uuid=dn(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=n.DEFAULT_UP.clone();let e=new P,t=new Ii,i=new Pi,s=new P(1,1,1);function r(){i.setFromEuler(t,!1)}function o(){t.setFromQuaternion(i,void 0,!1)}t._onChange(r),i._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new rt},normalMatrix:{value:new tt}}),this.matrix=new rt,this.matrixWorld=new rt,this.matrixAutoUpdate=n.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=n.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new yr,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return js.setFromAxisAngle(e,t),this.quaternion.multiply(js),this}rotateOnWorldAxis(e,t){return js.setFromAxisAngle(e,t),this.quaternion.premultiply(js),this}rotateX(e){return this.rotateOnAxis(qd,e)}rotateY(e){return this.rotateOnAxis(Yd,e)}rotateZ(e){return this.rotateOnAxis($d,e)}translateOnAxis(e,t){return Xd.copy(e).applyQuaternion(this.quaternion),this.position.add(Xd.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(qd,e)}translateY(e){return this.translateOnAxis(Yd,e)}translateZ(e){return this.translateOnAxis($d,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Tn.copy(this.matrixWorld).invert())}lookAt(e,t,i){e.isVector3?Aa.copy(e):Aa.set(e,t,i);let s=this.parent;this.updateWorldMatrix(!0,!1),qr.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Tn.lookAt(qr,Aa,this.up):Tn.lookAt(Aa,qr,this.up),this.quaternion.setFromRotationMatrix(Tn),s&&(Tn.extractRotation(s.matrixWorld),js.setFromRotationMatrix(Tn),this.quaternion.premultiply(js.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(Ze("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Zd),Ks.child=e,this.dispatchEvent(Ks),Ks.child=null):Ze("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}let t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(Eg),Rh.child=e,this.dispatchEvent(Rh),Rh.child=null),this}removeFromParent(){let e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Tn.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Tn.multiply(e.parent.matrixWorld)),e.applyMatrix4(Tn),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Zd),Ks.child=e,this.dispatchEvent(Ks),Ks.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let i=0,s=this.children.length;i<s;i++){let o=this.children[i].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,i=[]){this[e]===t&&i.push(this);let s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(qr,e,Sg),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(qr,bg,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);let t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);let t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);let t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].traverseVisible(e)}traverseAncestors(e){let t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let e=this.pivot;if(e!==null){let t=e.x,i=e.y,s=e.z,r=this.matrix.elements;r[12]+=t-r[0]*t-r[4]*i-r[8]*s,r[13]+=i-r[1]*t-r[5]*i-r[9]*s,r[14]+=s-r[2]*t-r[6]*i-r[10]*s}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);let t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].updateMatrixWorld(e)}updateWorldMatrix(e,t,i=!1){let s=this.parent;if(e===!0&&s!==null&&s.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||i)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,i=!0),t===!0){let r=this.children;for(let o=0,a=r.length;o<a;o++)r[o].updateWorldMatrix(!1,!0,i)}}toJSON(e){let t=e===void 0||typeof e=="string",i={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),this.static!==!1&&(s.static=this.static),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.pivot!==null&&(s.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(s.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(s.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,c){return a[c.uuid]===void 0&&(a[c.uuid]=c.toJSON(e)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);let a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){let c=a.shapes;if(Array.isArray(c))for(let l=0,h=c.length;l<h;l++){let u=c[l];r(e.shapes,u)}else r(e.shapes,c)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let a=[];for(let c=0,l=this.material.length;c<l;c++)a.push(r(e.materials,this.material[c]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){let c=this.animations[a];s.animations.push(r(e.animations,c))}}if(t){let a=o(e.geometries),c=o(e.materials),l=o(e.textures),h=o(e.images),u=o(e.shapes),d=o(e.skeletons),f=o(e.animations),g=o(e.nodes);a.length>0&&(i.geometries=a),c.length>0&&(i.materials=c),l.length>0&&(i.textures=l),h.length>0&&(i.images=h),u.length>0&&(i.shapes=u),d.length>0&&(i.skeletons=d),f.length>0&&(i.animations=f),g.length>0&&(i.nodes=g)}return i.object=s,i;function o(a){let c=[];for(let l in a){let h=a[l];delete h.metadata,c.push(h)}return c}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot!==null?e.pivot.clone():null,this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let i=0;i<e.children.length;i++){let s=e.children[i];this.add(s.clone())}return this}};ft.DEFAULT_UP=new P(0,1,0);ft.DEFAULT_MATRIX_AUTO_UPDATE=!0;ft.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var et=class extends ft{constructor(){super(),this.isGroup=!0,this.type="Group"}},wg={type:"move"},Mr=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new et,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new et,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new P,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new P),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new et,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new P,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new P,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){let t=this._hand;if(t)for(let i of e.hand.values())this._getHandJoint(t,i)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,i){let s=null,r=null,o=null,a=this._targetRay,c=this._grip,l=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(l&&e.hand){o=!0;for(let x of e.hand.values()){let p=t.getJointPose(x,i),m=this._getHandJoint(l,x);p!==null&&(m.matrix.fromArray(p.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=p.radius),m.visible=p!==null}let h=l.joints["index-finger-tip"],u=l.joints["thumb-tip"],d=h.position.distanceTo(u.position),f=.02,g=.005;l.inputState.pinching&&d>f+g?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!l.inputState.pinching&&d<=f-g&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else c!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,i),r!==null&&(c.matrix.fromArray(r.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),c.matrixWorldNeedsUpdate=!0,r.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(r.linearVelocity)):c.hasLinearVelocity=!1,r.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(r.angularVelocity)):c.hasAngularVelocity=!1,c.eventsEnabled&&c.dispatchEvent({type:"gripUpdated",data:e,target:this})));a!==null&&(s=t.getPose(e.targetRaySpace,i),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(wg)))}return a!==null&&(a.visible=s!==null),c!==null&&(c.visible=r!==null),l!==null&&(l.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){let i=new et;i.matrixAutoUpdate=!1,i.visible=!1,e.joints[t.jointName]=i,e.add(i)}return e.joints[t.jointName]}},sp={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Kn={h:0,s:0,l:0},Ra={h:0,s:0,l:0};function Ch(n,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?n+(e-n)*6*t:t<1/2?e:t<2/3?n+(e-n)*6*(2/3-t):n}var Te=class{constructor(e,t,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,i)}set(e,t,i){if(t===void 0&&i===void 0){let s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,i);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Lt){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,ht.colorSpaceToWorking(this,t),this}setRGB(e,t,i,s=ht.workingColorSpace){return this.r=e,this.g=t,this.b=i,ht.colorSpaceToWorking(this,s),this}setHSL(e,t,i,s=ht.workingColorSpace){if(e=Tu(e,1),t=je(t,0,1),i=je(i,0,1),t===0)this.r=this.g=this.b=i;else{let r=i<=.5?i*(1+t):i+t-i*t,o=2*i-r;this.r=Ch(o,r,e+1/3),this.g=Ch(o,r,e),this.b=Ch(o,r,e-1/3)}return ht.colorSpaceToWorking(this,s),this}setStyle(e,t=Lt){function i(r){r!==void 0&&parseFloat(r)<1&&$e("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r,o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:$e("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){let r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);$e("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Lt){let i=sp[e.toLowerCase()];return i!==void 0?this.setHex(i,t):$e("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=In(e.r),this.g=In(e.g),this.b=In(e.b),this}copyLinearToSRGB(e){return this.r=mr(e.r),this.g=mr(e.g),this.b=mr(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Lt){return ht.workingToColorSpace(li.copy(this),e),Math.round(je(li.r*255,0,255))*65536+Math.round(je(li.g*255,0,255))*256+Math.round(je(li.b*255,0,255))}getHexString(e=Lt){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=ht.workingColorSpace){ht.workingToColorSpace(li.copy(this),t);let i=li.r,s=li.g,r=li.b,o=Math.max(i,s,r),a=Math.min(i,s,r),c,l,h=(a+o)/2;if(a===o)c=0,l=0;else{let u=o-a;switch(l=h<=.5?u/(o+a):u/(2-o-a),o){case i:c=(s-r)/u+(s<r?6:0);break;case s:c=(r-i)/u+2;break;case r:c=(i-s)/u+4;break}c/=6}return e.h=c,e.s=l,e.l=h,e}getRGB(e,t=ht.workingColorSpace){return ht.workingToColorSpace(li.copy(this),t),e.r=li.r,e.g=li.g,e.b=li.b,e}getStyle(e=Lt){ht.workingToColorSpace(li.copy(this),e);let t=li.r,i=li.g,s=li.b;return e!==Lt?`color(${e} ${t.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(i*255)},${Math.round(s*255)})`}offsetHSL(e,t,i){return this.getHSL(Kn),this.setHSL(Kn.h+e,Kn.s+t,Kn.l+i)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,i){return this.r=e.r+(t.r-e.r)*i,this.g=e.g+(t.g-e.g)*i,this.b=e.b+(t.b-e.b)*i,this}lerpHSL(e,t){this.getHSL(Kn),e.getHSL(Ra);let i=no(Kn.h,Ra.h,t),s=no(Kn.s,Ra.s,t),r=no(Kn.l,Ra.l,t);return this.setHSL(i,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){let t=this.r,i=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*i+r[6]*s,this.g=r[1]*t+r[4]*i+r[7]*s,this.b=r[2]*t+r[5]*i+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},li=new Te;Te.NAMES=sp;var fo=class n{constructor(e,t=1,i=1e3){this.isFog=!0,this.name="",this.color=new Te(e),this.near=t,this.far=i}clone(){return new n(this.color,this.near,this.far)}toJSON(){return{type:"Fog",name:this.name,color:this.color.getHex(),near:this.near,far:this.far}}},Ds=class extends ft{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Ii,this.environmentIntensity=1,this.environmentRotation=new Ii,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){let t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}},Xi=new P,An=new P,Ph=new P,Rn=new P,Qs=new P,er=new P,Jd=new P,Ih=new P,Dh=new P,Lh=new P,Uh=new mt,Nh=new mt,Fh=new mt,hn=class n{constructor(e=new P,t=new P,i=new P){this.a=e,this.b=t,this.c=i}static getNormal(e,t,i,s){s.subVectors(i,t),Xi.subVectors(e,t),s.cross(Xi);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,i,s,r){Xi.subVectors(s,t),An.subVectors(i,t),Ph.subVectors(e,t);let o=Xi.dot(Xi),a=Xi.dot(An),c=Xi.dot(Ph),l=An.dot(An),h=An.dot(Ph),u=o*l-a*a;if(u===0)return r.set(0,0,0),null;let d=1/u,f=(l*c-a*h)*d,g=(o*h-a*c)*d;return r.set(1-f-g,g,f)}static containsPoint(e,t,i,s){return this.getBarycoord(e,t,i,s,Rn)===null?!1:Rn.x>=0&&Rn.y>=0&&Rn.x+Rn.y<=1}static getInterpolation(e,t,i,s,r,o,a,c){return this.getBarycoord(e,t,i,s,Rn)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(r,Rn.x),c.addScaledVector(o,Rn.y),c.addScaledVector(a,Rn.z),c)}static getInterpolatedAttribute(e,t,i,s,r,o){return Uh.setScalar(0),Nh.setScalar(0),Fh.setScalar(0),Uh.fromBufferAttribute(e,t),Nh.fromBufferAttribute(e,i),Fh.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(Uh,r.x),o.addScaledVector(Nh,r.y),o.addScaledVector(Fh,r.z),o}static isFrontFacing(e,t,i,s){return Xi.subVectors(i,t),An.subVectors(e,t),Xi.cross(An).dot(s)<0}set(e,t,i){return this.a.copy(e),this.b.copy(t),this.c.copy(i),this}setFromPointsAndIndices(e,t,i,s){return this.a.copy(e[t]),this.b.copy(e[i]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,i,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Xi.subVectors(this.c,this.b),An.subVectors(this.a,this.b),Xi.cross(An).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return n.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return n.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,i,s,r){return n.getInterpolation(e,this.a,this.b,this.c,t,i,s,r)}containsPoint(e){return n.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return n.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){let i=this.a,s=this.b,r=this.c,o,a;Qs.subVectors(s,i),er.subVectors(r,i),Ih.subVectors(e,i);let c=Qs.dot(Ih),l=er.dot(Ih);if(c<=0&&l<=0)return t.copy(i);Dh.subVectors(e,s);let h=Qs.dot(Dh),u=er.dot(Dh);if(h>=0&&u<=h)return t.copy(s);let d=c*u-h*l;if(d<=0&&c>=0&&h<=0)return o=c/(c-h),t.copy(i).addScaledVector(Qs,o);Lh.subVectors(e,r);let f=Qs.dot(Lh),g=er.dot(Lh);if(g>=0&&f<=g)return t.copy(r);let x=f*l-c*g;if(x<=0&&l>=0&&g<=0)return a=l/(l-g),t.copy(i).addScaledVector(er,a);let p=h*g-f*u;if(p<=0&&u-h>=0&&f-g>=0)return Jd.subVectors(r,s),a=(u-h)/(u-h+(f-g)),t.copy(s).addScaledVector(Jd,a);let m=1/(p+x+d);return o=x*m,a=d*m,t.copy(i).addScaledVector(Qs,o).addScaledVector(er,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},pi=class{constructor(e=new P(1/0,1/0,1/0),t=new P(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,i=e.length;t<i;t+=3)this.expandByPoint(qi.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,i=e.count;t<i;t++)this.expandByPoint(qi.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,i=e.length;t<i;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){let i=qi.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(i),this.max.copy(e).add(i),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);let i=e.geometry;if(i!==void 0){let r=i.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,qi):qi.fromBufferAttribute(r,o),qi.applyMatrix4(e.matrixWorld),this.expandByPoint(qi);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Ca.copy(e.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),Ca.copy(i.boundingBox)),Ca.applyMatrix4(e.matrixWorld),this.union(Ca)}let s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,qi),qi.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,i;return e.normal.x>0?(t=e.normal.x*this.min.x,i=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,i=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,i+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,i+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,i+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,i+=e.normal.z*this.min.z),t<=-e.constant&&i>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Yr),Pa.subVectors(this.max,Yr),tr.subVectors(e.a,Yr),ir.subVectors(e.b,Yr),nr.subVectors(e.c,Yr),Qn.subVectors(ir,tr),es.subVectors(nr,ir),bs.subVectors(tr,nr);let t=[0,-Qn.z,Qn.y,0,-es.z,es.y,0,-bs.z,bs.y,Qn.z,0,-Qn.x,es.z,0,-es.x,bs.z,0,-bs.x,-Qn.y,Qn.x,0,-es.y,es.x,0,-bs.y,bs.x,0];return!Oh(t,tr,ir,nr,Pa)||(t=[1,0,0,0,1,0,0,0,1],!Oh(t,tr,ir,nr,Pa))?!1:(Ia.crossVectors(Qn,es),t=[Ia.x,Ia.y,Ia.z],Oh(t,tr,ir,nr,Pa))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,qi).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(qi).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Cn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Cn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Cn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Cn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Cn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Cn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Cn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Cn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Cn),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}},Cn=[new P,new P,new P,new P,new P,new P,new P,new P],qi=new P,Ca=new pi,tr=new P,ir=new P,nr=new P,Qn=new P,es=new P,bs=new P,Yr=new P,Pa=new P,Ia=new P,Es=new P;function Oh(n,e,t,i,s){for(let r=0,o=n.length-3;r<=o;r+=3){Es.fromArray(n,r);let a=s.x*Math.abs(Es.x)+s.y*Math.abs(Es.y)+s.z*Math.abs(Es.z),c=e.dot(Es),l=t.dot(Es),h=i.dot(Es);if(Math.max(-Math.max(c,l,h),Math.min(c,l,h))>a)return!1}return!0}var kt=new P,Da=new $,Tg=0,Ut=class extends Ji{constructor(e,t,i=!1){if(super(),Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:Tg++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=i,this.usage=xl,this.updateRanges=[],this.gpuType=Hi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,i){e*=this.itemSize,i*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[i+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,i=this.count;t<i;t++)Da.fromBufferAttribute(this,t),Da.applyMatrix3(e),this.setXY(t,Da.x,Da.y);else if(this.itemSize===3)for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix3(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyMatrix4(e){for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix4(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyNormalMatrix(e){for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.applyNormalMatrix(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}transformDirection(e){for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.transformDirection(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let i=this.array[e*this.itemSize+t];return this.normalized&&(i=Yi(i,this.array)),i}setComponent(e,t,i){return this.normalized&&(i=gt(i,this.array)),this.array[e*this.itemSize+t]=i,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=Yi(t,this.array)),t}setX(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=Yi(t,this.array)),t}setY(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=Yi(t,this.array)),t}setZ(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=Yi(t,this.array)),t}setW(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,i){return e*=this.itemSize,this.normalized&&(t=gt(t,this.array),i=gt(i,this.array)),this.array[e+0]=t,this.array[e+1]=i,this}setXYZ(e,t,i,s){return e*=this.itemSize,this.normalized&&(t=gt(t,this.array),i=gt(i,this.array),s=gt(s,this.array)),this.array[e+0]=t,this.array[e+1]=i,this.array[e+2]=s,this}setXYZW(e,t,i,s,r){return e*=this.itemSize,this.normalized&&(t=gt(t,this.array),i=gt(i,this.array),s=gt(s,this.array),r=gt(r,this.array)),this.array[e+0]=t,this.array[e+1]=i,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==xl&&(e.usage=this.usage),e}dispose(){this.dispatchEvent({type:"dispose"})}};var po=class extends Ut{constructor(e,t,i){super(new Uint16Array(e),t,i)}};var mo=class extends Ut{constructor(e,t,i){super(new Uint32Array(e),t,i)}};var it=class extends Ut{constructor(e,t,i){super(new Float32Array(e),t,i)}},Ag=new pi,$r=new P,Bh=new P,vi=class{constructor(e=new P,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){let i=this.center;t!==void 0?i.copy(t):Ag.setFromPoints(e).getCenter(i);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,i.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){let t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){let i=this.center.distanceToSquared(e);return t.copy(e),i>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;$r.subVectors(e,this.center);let t=$r.lengthSq();if(t>this.radius*this.radius){let i=Math.sqrt(t),s=(i-this.radius)*.5;this.center.addScaledVector($r,s/i),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Bh.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint($r.copy(e.center).add(Bh)),this.expandByPoint($r.copy(e.center).sub(Bh))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}},Rg=0,Oi=new rt,zh=new ft,sr=new P,Ri=new pi,Zr=new pi,Jt=new P,ut=class n extends Ji{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:Rg++}),this.uuid=dn(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(Km(e)?mo:po)(e,1):this.index=e,this}setIndirect(e,t=0){return this.indirect=e,this.indirectOffset=t,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,i=0){this.groups.push({start:e,count:t,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){let t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);let i=this.attributes.normal;if(i!==void 0){let r=new tt().getNormalMatrix(e);i.applyNormalMatrix(r),i.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(e){return Oi.makeRotationFromQuaternion(e),this.applyMatrix4(Oi),this}rotateX(e){return Oi.makeRotationX(e),this.applyMatrix4(Oi),this}rotateY(e){return Oi.makeRotationY(e),this.applyMatrix4(Oi),this}rotateZ(e){return Oi.makeRotationZ(e),this.applyMatrix4(Oi),this}translate(e,t,i){return Oi.makeTranslation(e,t,i),this.applyMatrix4(Oi),this}scale(e,t,i){return Oi.makeScale(e,t,i),this.applyMatrix4(Oi),this}lookAt(e){return zh.lookAt(e),zh.updateMatrix(),this.applyMatrix4(zh.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(sr).negate(),this.translate(sr.x,sr.y,sr.z),this}setFromPoints(e){let t=this.getAttribute("position");if(t===void 0){let i=[];for(let s=0,r=e.length;s<r;s++){let o=e[s];i.push(o.x,o.y,o.z||0)}this.setAttribute("position",new it(i,3))}else{let i=Math.min(e.length,t.count);for(let s=0;s<i;s++){let r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&$e("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new pi);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Ze("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new P(-1/0,-1/0,-1/0),new P(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let i=0,s=t.length;i<s;i++){let r=t[i];Ri.setFromBufferAttribute(r),this.morphTargetsRelative?(Jt.addVectors(this.boundingBox.min,Ri.min),this.boundingBox.expandByPoint(Jt),Jt.addVectors(this.boundingBox.max,Ri.max),this.boundingBox.expandByPoint(Jt)):(this.boundingBox.expandByPoint(Ri.min),this.boundingBox.expandByPoint(Ri.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Ze('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new vi);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Ze("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new P,1/0);return}if(e){let i=this.boundingSphere.center;if(Ri.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){let a=t[r];Zr.setFromBufferAttribute(a),this.morphTargetsRelative?(Jt.addVectors(Ri.min,Zr.min),Ri.expandByPoint(Jt),Jt.addVectors(Ri.max,Zr.max),Ri.expandByPoint(Jt)):(Ri.expandByPoint(Zr.min),Ri.expandByPoint(Zr.max))}Ri.getCenter(i);let s=0;for(let r=0,o=e.count;r<o;r++)Jt.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared(Jt));if(t)for(let r=0,o=t.length;r<o;r++){let a=t[r],c=this.morphTargetsRelative;for(let l=0,h=a.count;l<h;l++)Jt.fromBufferAttribute(a,l),c&&(sr.fromBufferAttribute(e,l),Jt.add(sr)),s=Math.max(s,i.distanceToSquared(Jt))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&Ze('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){Ze("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let i=t.position,s=t.normal,r=t.uv,o=this.getAttribute("tangent");(o===void 0||o.count!==i.count)&&(o=new Ut(new Float32Array(4*i.count),4),this.setAttribute("tangent",o));let a=[],c=[];for(let _=0;_<i.count;_++)a[_]=new P,c[_]=new P;let l=new P,h=new P,u=new P,d=new $,f=new $,g=new $,x=new P,p=new P;function m(_,E,C){l.fromBufferAttribute(i,_),h.fromBufferAttribute(i,E),u.fromBufferAttribute(i,C),d.fromBufferAttribute(r,_),f.fromBufferAttribute(r,E),g.fromBufferAttribute(r,C),h.sub(l),u.sub(l),f.sub(d),g.sub(d);let I=1/(f.x*g.y-g.x*f.y);isFinite(I)&&(x.copy(h).multiplyScalar(g.y).addScaledVector(u,-f.y).multiplyScalar(I),p.copy(u).multiplyScalar(f.x).addScaledVector(h,-g.x).multiplyScalar(I),a[_].add(x),a[E].add(x),a[C].add(x),c[_].add(p),c[E].add(p),c[C].add(p))}let M=this.groups;M.length===0&&(M=[{start:0,count:e.count}]);for(let _=0,E=M.length;_<E;++_){let C=M[_],I=C.start,L=C.count;for(let V=I,q=I+L;V<q;V+=3)m(e.getX(V+0),e.getX(V+1),e.getX(V+2))}let b=new P,y=new P,T=new P,S=new P;function A(_){T.fromBufferAttribute(s,_),S.copy(T);let E=a[_];b.copy(E),b.sub(T.multiplyScalar(T.dot(E))).normalize(),y.crossVectors(S,E);let I=y.dot(c[_])<0?-1:1;o.setXYZW(_,b.x,b.y,b.z,I)}for(let _=0,E=M.length;_<E;++_){let C=M[_],I=C.start,L=C.count;for(let V=I,q=I+L;V<q;V+=3)A(e.getX(V+0)),A(e.getX(V+1)),A(e.getX(V+2))}this._transformed=!0}computeVertexNormals(){let e=this.index,t=this.getAttribute("position");if(t!==void 0){let i=this.getAttribute("normal");if(i===void 0||i.count!==t.count)i=new Ut(new Float32Array(t.count*3),3),this.setAttribute("normal",i);else for(let d=0,f=i.count;d<f;d++)i.setXYZ(d,0,0,0);let s=new P,r=new P,o=new P,a=new P,c=new P,l=new P,h=new P,u=new P;if(e)for(let d=0,f=e.count;d<f;d+=3){let g=e.getX(d+0),x=e.getX(d+1),p=e.getX(d+2);s.fromBufferAttribute(t,g),r.fromBufferAttribute(t,x),o.fromBufferAttribute(t,p),h.subVectors(o,r),u.subVectors(s,r),h.cross(u),a.fromBufferAttribute(i,g),c.fromBufferAttribute(i,x),l.fromBufferAttribute(i,p),a.add(h),c.add(h),l.add(h),i.setXYZ(g,a.x,a.y,a.z),i.setXYZ(x,c.x,c.y,c.z),i.setXYZ(p,l.x,l.y,l.z)}else for(let d=0,f=t.count;d<f;d+=3)s.fromBufferAttribute(t,d+0),r.fromBufferAttribute(t,d+1),o.fromBufferAttribute(t,d+2),h.subVectors(o,r),u.subVectors(s,r),h.cross(u),i.setXYZ(d+0,h.x,h.y,h.z),i.setXYZ(d+1,h.x,h.y,h.z),i.setXYZ(d+2,h.x,h.y,h.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){let e=this.attributes.normal;for(let t=0,i=e.count;t<i;t++)Jt.fromBufferAttribute(e,t),Jt.normalize(),e.setXYZ(t,Jt.x,Jt.y,Jt.z)}toNonIndexed(){function e(a,c){let l=a.array,h=a.itemSize,u=a.normalized,d=new l.constructor(c.length*h),f=0,g=0;for(let x=0,p=c.length;x<p;x++){a.isInterleavedBufferAttribute?f=c[x]*a.data.stride+a.offset:f=c[x]*h;for(let m=0;m<h;m++)d[g++]=l[f++]}return new Ut(d,h,u)}if(this.index===null)return $e("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let t=new n,i=this.index.array,s=this.attributes;for(let a in s){let c=s[a],l=e(c,i);t.setAttribute(a,l)}let r=this.morphAttributes;for(let a in r){let c=[],l=r[a];for(let h=0,u=l.length;h<u;h++){let d=l[h],f=e(d,i);c.push(f)}t.morphAttributes[a]=c}t.morphTargetsRelative=this.morphTargetsRelative;let o=this.groups;for(let a=0,c=o.length;a<c;a++){let l=o[a];t.addGroup(l.start,l.count,l.materialIndex)}return t}toJSON(){let e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.parameters!==void 0&&this._transformed===!0?"BufferGeometry":this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let c=this.parameters;for(let l in c)c[l]!==void 0&&(e[l]=c[l]);return e}e.data={attributes:{}};let t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});let i=this.attributes;for(let c in i){let l=i[c];e.data.attributes[c]=l.toJSON(e.data)}let s={},r=!1;for(let c in this.morphAttributes){let l=this.morphAttributes[c],h=[];for(let u=0,d=l.length;u<d;u++){let f=l[u];h.push(f.toJSON(e.data))}h.length>0&&(s[c]=h,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);let o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));let a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let t={};this.name=e.name;let i=e.index;i!==null&&this.setIndex(i.clone());let s=e.attributes;for(let l in s){let h=s[l];this.setAttribute(l,h.clone(t))}let r=e.morphAttributes;for(let l in r){let h=[],u=r[l];for(let d=0,f=u.length;d<f;d++)h.push(u[d].clone(t));this.morphAttributes[l]=h}this.morphTargetsRelative=e.morphTargetsRelative;let o=e.groups;for(let l=0,h=o.length;l<h;l++){let u=o[l];this.addGroup(u.start,u.count,u.materialIndex)}let a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());let c=e.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this._transformed=e._transformed,this}dispose(){this.dispatchEvent({type:"dispose"})}},go=class{constructor(e,t){this.isInterleavedBuffer=!0,this.array=e,this.stride=t,this.count=e!==void 0?e.length/t:0,this.usage=xl,this.updateRanges=[],this.version=0,this.uuid=dn()}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.array=new e.array.constructor(e.array),this.count=e.count,this.stride=e.stride,this.usage=e.usage,this}copyAt(e,t,i){e*=this.stride,i*=t.stride;for(let s=0,r=this.stride;s<r;s++)this.array[e+s]=t.array[i+s];return this}set(e,t=0){return this.array.set(e,t),this}clone(e){e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=dn()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);let t=new this.array.constructor(e.arrayBuffers[this.array.buffer._uuid]),i=new this.constructor(t,this.stride);return i.setUsage(this.usage),i}onUpload(e){return this.onUploadCallback=e,this}toJSON(e){return e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=dn()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=Array.from(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}},di=new P,Di=class n{constructor(e,t,i,s=!1){this.isInterleavedBufferAttribute=!0,this.name="",this.data=e,this.itemSize=t,this.offset=i,this.normalized=s}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(e){this.data.needsUpdate=e}applyMatrix4(e){for(let t=0,i=this.data.count;t<i;t++)di.fromBufferAttribute(this,t),di.applyMatrix4(e),this.setXYZ(t,di.x,di.y,di.z);return this}applyNormalMatrix(e){for(let t=0,i=this.count;t<i;t++)di.fromBufferAttribute(this,t),di.applyNormalMatrix(e),this.setXYZ(t,di.x,di.y,di.z);return this}transformDirection(e){for(let t=0,i=this.count;t<i;t++)di.fromBufferAttribute(this,t),di.transformDirection(e),this.setXYZ(t,di.x,di.y,di.z);return this}getComponent(e,t){let i=this.array[e*this.data.stride+this.offset+t];return this.normalized&&(i=Yi(i,this.array)),i}setComponent(e,t,i){return this.normalized&&(i=gt(i,this.array)),this.data.array[e*this.data.stride+this.offset+t]=i,this}setX(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset]=t,this}setY(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset+1]=t,this}setZ(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset+2]=t,this}setW(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset+3]=t,this}getX(e){let t=this.data.array[e*this.data.stride+this.offset];return this.normalized&&(t=Yi(t,this.array)),t}getY(e){let t=this.data.array[e*this.data.stride+this.offset+1];return this.normalized&&(t=Yi(t,this.array)),t}getZ(e){let t=this.data.array[e*this.data.stride+this.offset+2];return this.normalized&&(t=Yi(t,this.array)),t}getW(e){let t=this.data.array[e*this.data.stride+this.offset+3];return this.normalized&&(t=Yi(t,this.array)),t}setXY(e,t,i){return e=e*this.data.stride+this.offset,this.normalized&&(t=gt(t,this.array),i=gt(i,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this}setXYZ(e,t,i,s){return e=e*this.data.stride+this.offset,this.normalized&&(t=gt(t,this.array),i=gt(i,this.array),s=gt(s,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this.data.array[e+2]=s,this}setXYZW(e,t,i,s,r){return e=e*this.data.stride+this.offset,this.normalized&&(t=gt(t,this.array),i=gt(i,this.array),s=gt(s,this.array),r=gt(r,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this.data.array[e+2]=s,this.data.array[e+3]=r,this}clone(e){if(e===void 0){ho("InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.");let t=[];for(let i=0;i<this.count;i++){let s=i*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return new Ut(new this.array.constructor(t),this.itemSize,this.normalized)}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.clone(e)),new n(e.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(e){if(e===void 0){ho("InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.");let t=[];for(let i=0;i<this.count;i++){let s=i*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:t,normalized:this.normalized}}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.toJSON(e)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}},Cg=0,yi=class extends Ji{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Cg++}),this.uuid=dn(),this.name="",this.type="Material",this.blending=Ps,this.side=Zi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=al,this.blendDst=ll,this.blendEquation=Ci,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Te(0,0,0),this.blendAlpha=0,this.depthFunc=Is,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=nu,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=As,this.stencilZFail=As,this.stencilZPass=As,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(let t in e){let i=e[t];if(i===void 0){$e(`Material: parameter '${t}' has value of undefined.`);continue}let s=this[t];if(s===void 0){$e(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(i):s&&s.isVector2&&i&&i.isVector2||s&&s.isEuler&&i&&i.isEuler||s&&s.isVector3&&i&&i.isVector3?s.copy(i):this[t]=i}}toJSON(e){let t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});let i={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,this.name!==""&&(i.name=this.name),this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(i.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(i.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(i.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(e).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(e).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(e).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(e).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(e).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.shadowSide!==null&&(i.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),this.blending!==Ps&&(i.blending=this.blending),this.side!==Zi&&(i.side=this.side),this.vertexColors===!0&&(i.vertexColors=!0),this.opacity<1&&(i.opacity=this.opacity),this.transparent===!0&&(i.transparent=!0),this.blendSrc!==al&&(i.blendSrc=this.blendSrc),this.blendDst!==ll&&(i.blendDst=this.blendDst),this.blendEquation!==Ci&&(i.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(i.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(i.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(i.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(i.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(i.blendAlpha=this.blendAlpha),this.depthFunc!==Is&&(i.depthFunc=this.depthFunc),this.depthTest===!1&&(i.depthTest=this.depthTest),this.depthWrite===!1&&(i.depthWrite=this.depthWrite),this.colorWrite===!1&&(i.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(i.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==nu&&(i.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(i.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(i.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==As&&(i.stencilFail=this.stencilFail),this.stencilZFail!==As&&(i.stencilZFail=this.stencilZFail),this.stencilZPass!==As&&(i.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(i.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(i.rotation=this.rotation),this.polygonOffset===!0&&(i.polygonOffset=!0),this.polygonOffsetFactor!==0&&(i.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(i.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(i.linewidth=this.linewidth),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.dithering===!0&&(i.dithering=!0),this.alphaTest>0&&(i.alphaTest=this.alphaTest),this.alphaHash===!0&&(i.alphaHash=!0),this.alphaToCoverage===!0&&(i.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(i.premultipliedAlpha=!0),this.forceSinglePass===!0&&(i.forceSinglePass=!0),this.allowOverride===!1&&(i.allowOverride=!1),this.wireframe===!0&&(i.wireframe=!0),this.wireframeLinewidth>1&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(i.flatShading=!0),this.visible===!1&&(i.visible=!1),this.toneMapped===!1&&(i.toneMapped=!1),this.fog===!1&&(i.fog=!1),Object.keys(this.userData).length>0&&(i.userData=this.userData);function s(r){let o=[];for(let a in r){let c=r[a];delete c.metadata,o.push(c)}return o}if(t){let r=s(e.textures),o=s(e.images);r.length>0&&(i.textures=r),o.length>0&&(i.images=o)}return i}fromJSON(e,t){if(e.uuid!==void 0&&(this.uuid=e.uuid),e.name!==void 0&&(this.name=e.name),e.color!==void 0&&this.color!==void 0&&this.color.setHex(e.color),e.roughness!==void 0&&(this.roughness=e.roughness),e.metalness!==void 0&&(this.metalness=e.metalness),e.sheen!==void 0&&(this.sheen=e.sheen),e.sheenColor!==void 0&&(this.sheenColor=new Te().setHex(e.sheenColor)),e.sheenRoughness!==void 0&&(this.sheenRoughness=e.sheenRoughness),e.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(e.emissive),e.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(e.specular),e.specularIntensity!==void 0&&(this.specularIntensity=e.specularIntensity),e.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(e.specularColor),e.shininess!==void 0&&(this.shininess=e.shininess),e.clearcoat!==void 0&&(this.clearcoat=e.clearcoat),e.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=e.clearcoatRoughness),e.dispersion!==void 0&&(this.dispersion=e.dispersion),e.iridescence!==void 0&&(this.iridescence=e.iridescence),e.iridescenceIOR!==void 0&&(this.iridescenceIOR=e.iridescenceIOR),e.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessRange=e.iridescenceThicknessRange),e.transmission!==void 0&&(this.transmission=e.transmission),e.thickness!==void 0&&(this.thickness=e.thickness),e.attenuationDistance!==void 0&&(this.attenuationDistance=e.attenuationDistance),e.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(e.attenuationColor),e.anisotropy!==void 0&&(this.anisotropy=e.anisotropy),e.anisotropyRotation!==void 0&&(this.anisotropyRotation=e.anisotropyRotation),e.fog!==void 0&&(this.fog=e.fog),e.flatShading!==void 0&&(this.flatShading=e.flatShading),e.blending!==void 0&&(this.blending=e.blending),e.combine!==void 0&&(this.combine=e.combine),e.side!==void 0&&(this.side=e.side),e.shadowSide!==void 0&&(this.shadowSide=e.shadowSide),e.opacity!==void 0&&(this.opacity=e.opacity),e.transparent!==void 0&&(this.transparent=e.transparent),e.alphaTest!==void 0&&(this.alphaTest=e.alphaTest),e.alphaHash!==void 0&&(this.alphaHash=e.alphaHash),e.depthFunc!==void 0&&(this.depthFunc=e.depthFunc),e.depthTest!==void 0&&(this.depthTest=e.depthTest),e.depthWrite!==void 0&&(this.depthWrite=e.depthWrite),e.colorWrite!==void 0&&(this.colorWrite=e.colorWrite),e.blendSrc!==void 0&&(this.blendSrc=e.blendSrc),e.blendDst!==void 0&&(this.blendDst=e.blendDst),e.blendEquation!==void 0&&(this.blendEquation=e.blendEquation),e.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=e.blendSrcAlpha),e.blendDstAlpha!==void 0&&(this.blendDstAlpha=e.blendDstAlpha),e.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=e.blendEquationAlpha),e.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(e.blendColor),e.blendAlpha!==void 0&&(this.blendAlpha=e.blendAlpha),e.stencilWriteMask!==void 0&&(this.stencilWriteMask=e.stencilWriteMask),e.stencilFunc!==void 0&&(this.stencilFunc=e.stencilFunc),e.stencilRef!==void 0&&(this.stencilRef=e.stencilRef),e.stencilFuncMask!==void 0&&(this.stencilFuncMask=e.stencilFuncMask),e.stencilFail!==void 0&&(this.stencilFail=e.stencilFail),e.stencilZFail!==void 0&&(this.stencilZFail=e.stencilZFail),e.stencilZPass!==void 0&&(this.stencilZPass=e.stencilZPass),e.stencilWrite!==void 0&&(this.stencilWrite=e.stencilWrite),e.wireframe!==void 0&&(this.wireframe=e.wireframe),e.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=e.wireframeLinewidth),e.wireframeLinecap!==void 0&&(this.wireframeLinecap=e.wireframeLinecap),e.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=e.wireframeLinejoin),e.rotation!==void 0&&(this.rotation=e.rotation),e.linewidth!==void 0&&(this.linewidth=e.linewidth),e.dashSize!==void 0&&(this.dashSize=e.dashSize),e.gapSize!==void 0&&(this.gapSize=e.gapSize),e.scale!==void 0&&(this.scale=e.scale),e.polygonOffset!==void 0&&(this.polygonOffset=e.polygonOffset),e.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=e.polygonOffsetFactor),e.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=e.polygonOffsetUnits),e.dithering!==void 0&&(this.dithering=e.dithering),e.alphaToCoverage!==void 0&&(this.alphaToCoverage=e.alphaToCoverage),e.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=e.premultipliedAlpha),e.forceSinglePass!==void 0&&(this.forceSinglePass=e.forceSinglePass),e.allowOverride!==void 0&&(this.allowOverride=e.allowOverride),e.visible!==void 0&&(this.visible=e.visible),e.toneMapped!==void 0&&(this.toneMapped=e.toneMapped),e.userData!==void 0&&(this.userData=e.userData),e.vertexColors!==void 0&&(typeof e.vertexColors=="number"?this.vertexColors=e.vertexColors>0:this.vertexColors=e.vertexColors),e.size!==void 0&&(this.size=e.size),e.sizeAttenuation!==void 0&&(this.sizeAttenuation=e.sizeAttenuation),e.map!==void 0&&(this.map=t[e.map]||null),e.matcap!==void 0&&(this.matcap=t[e.matcap]||null),e.alphaMap!==void 0&&(this.alphaMap=t[e.alphaMap]||null),e.bumpMap!==void 0&&(this.bumpMap=t[e.bumpMap]||null),e.bumpScale!==void 0&&(this.bumpScale=e.bumpScale),e.normalMap!==void 0&&(this.normalMap=t[e.normalMap]||null),e.normalMapType!==void 0&&(this.normalMapType=e.normalMapType),e.normalScale!==void 0){let i=e.normalScale;Array.isArray(i)===!1&&(i=[i,i]),this.normalScale=new $().fromArray(i)}return e.displacementMap!==void 0&&(this.displacementMap=t[e.displacementMap]||null),e.displacementScale!==void 0&&(this.displacementScale=e.displacementScale),e.displacementBias!==void 0&&(this.displacementBias=e.displacementBias),e.roughnessMap!==void 0&&(this.roughnessMap=t[e.roughnessMap]||null),e.metalnessMap!==void 0&&(this.metalnessMap=t[e.metalnessMap]||null),e.emissiveMap!==void 0&&(this.emissiveMap=t[e.emissiveMap]||null),e.emissiveIntensity!==void 0&&(this.emissiveIntensity=e.emissiveIntensity),e.specularMap!==void 0&&(this.specularMap=t[e.specularMap]||null),e.specularIntensityMap!==void 0&&(this.specularIntensityMap=t[e.specularIntensityMap]||null),e.specularColorMap!==void 0&&(this.specularColorMap=t[e.specularColorMap]||null),e.envMap!==void 0&&(this.envMap=t[e.envMap]||null),e.envMapRotation!==void 0&&this.envMapRotation.fromArray(e.envMapRotation),e.envMapIntensity!==void 0&&(this.envMapIntensity=e.envMapIntensity),e.reflectivity!==void 0&&(this.reflectivity=e.reflectivity),e.refractionRatio!==void 0&&(this.refractionRatio=e.refractionRatio),e.lightMap!==void 0&&(this.lightMap=t[e.lightMap]||null),e.lightMapIntensity!==void 0&&(this.lightMapIntensity=e.lightMapIntensity),e.aoMap!==void 0&&(this.aoMap=t[e.aoMap]||null),e.aoMapIntensity!==void 0&&(this.aoMapIntensity=e.aoMapIntensity),e.gradientMap!==void 0&&(this.gradientMap=t[e.gradientMap]||null),e.clearcoatMap!==void 0&&(this.clearcoatMap=t[e.clearcoatMap]||null),e.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=t[e.clearcoatRoughnessMap]||null),e.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=t[e.clearcoatNormalMap]||null),e.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new $().fromArray(e.clearcoatNormalScale)),e.iridescenceMap!==void 0&&(this.iridescenceMap=t[e.iridescenceMap]||null),e.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=t[e.iridescenceThicknessMap]||null),e.transmissionMap!==void 0&&(this.transmissionMap=t[e.transmissionMap]||null),e.thicknessMap!==void 0&&(this.thicknessMap=t[e.thicknessMap]||null),e.anisotropyMap!==void 0&&(this.anisotropyMap=t[e.anisotropyMap]||null),e.sheenColorMap!==void 0&&(this.sheenColorMap=t[e.sheenColorMap]||null),e.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=t[e.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;let t=e.clippingPlanes,i=null;if(t!==null){let s=t.length;i=new Array(s);for(let r=0;r!==s;++r)i[r]=t[r].clone()}return this.clippingPlanes=i,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}},Sr=class extends yi{constructor(e){super(),this.isSpriteMaterial=!0,this.type="SpriteMaterial",this.color=new Te(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.rotation=e.rotation,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}},rr,Jr=new P,or=new P,ar=new P,lr=new $,jr=new $,rp=new rt,La=new P,Kr=new P,Ua=new P,jd=new $,kh=new $,Kd=new $,_o=class extends ft{constructor(e=new Sr){if(super(),this.isSprite=!0,this.type="Sprite",rr===void 0){rr=new ut;let t=new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),i=new go(t,5);rr.setIndex([0,1,2,0,2,3]),rr.setAttribute("position",new Di(i,3,0,!1)),rr.setAttribute("uv",new Di(i,2,3,!1))}this.geometry=rr,this.material=e,this.center=new $(.5,.5),this.count=1}raycast(e,t){e.camera===null&&Ze('Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.'),or.setFromMatrixScale(this.matrixWorld),rp.copy(e.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(e.camera.matrixWorldInverse,this.matrixWorld),ar.setFromMatrixPosition(this.modelViewMatrix),e.camera.isPerspectiveCamera&&this.material.sizeAttenuation===!1&&or.multiplyScalar(-ar.z);let i=this.material.rotation,s,r;i!==0&&(r=Math.cos(i),s=Math.sin(i));let o=this.center;Na(La.set(-.5,-.5,0),ar,o,or,s,r),Na(Kr.set(.5,-.5,0),ar,o,or,s,r),Na(Ua.set(.5,.5,0),ar,o,or,s,r),jd.set(0,0),kh.set(1,0),Kd.set(1,1);let a=e.ray.intersectTriangle(La,Kr,Ua,!1,Jr);if(a===null&&(Na(Kr.set(-.5,.5,0),ar,o,or,s,r),kh.set(0,1),a=e.ray.intersectTriangle(La,Ua,Kr,!1,Jr),a===null))return;let c=e.ray.origin.distanceTo(Jr);c<e.near||c>e.far||t.push({distance:c,point:Jr.clone(),uv:hn.getInterpolation(Jr,La,Kr,Ua,jd,kh,Kd,new $),face:null,object:this})}copy(e,t){return super.copy(e,t),e.center!==void 0&&this.center.copy(e.center),this.material=e.material,this}};function Na(n,e,t,i,s,r){lr.subVectors(n,t).addScalar(.5).multiply(i),s!==void 0?(jr.x=r*lr.x-s*lr.y,jr.y=s*lr.x+r*lr.y):jr.copy(lr),n.copy(e),n.x+=jr.x,n.y+=jr.y,n.applyMatrix4(rp)}var Pn=new P,Hh=new P,Fa=new P,ts=new P,Vh=new P,Oa=new P,Gh=new P,Dn=class{constructor(e=new P,t=new P(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Pn)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);let i=t.dot(this.direction);return i<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){let t=Pn.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(Pn.copy(this.origin).addScaledVector(this.direction,t),Pn.distanceToSquared(e))}distanceSqToSegment(e,t,i,s){Hh.copy(e).add(t).multiplyScalar(.5),Fa.copy(t).sub(e).normalize(),ts.copy(this.origin).sub(Hh);let r=e.distanceTo(t)*.5,o=-this.direction.dot(Fa),a=ts.dot(this.direction),c=-ts.dot(Fa),l=ts.lengthSq(),h=Math.abs(1-o*o),u,d,f,g;if(h>0)if(u=o*c-a,d=o*a-c,g=r*h,u>=0)if(d>=-g)if(d<=g){let x=1/h;u*=x,d*=x,f=u*(u+o*d+2*a)+d*(o*u+d+2*c)+l}else d=r,u=Math.max(0,-(o*d+a)),f=-u*u+d*(d+2*c)+l;else d=-r,u=Math.max(0,-(o*d+a)),f=-u*u+d*(d+2*c)+l;else d<=-g?(u=Math.max(0,-(-o*r+a)),d=u>0?-r:Math.min(Math.max(-r,-c),r),f=-u*u+d*(d+2*c)+l):d<=g?(u=0,d=Math.min(Math.max(-r,-c),r),f=d*(d+2*c)+l):(u=Math.max(0,-(o*r+a)),d=u>0?r:Math.min(Math.max(-r,-c),r),f=-u*u+d*(d+2*c)+l);else d=o>0?-r:r,u=Math.max(0,-(o*d+a)),f=-u*u+d*(d+2*c)+l;return i&&i.copy(this.origin).addScaledVector(this.direction,u),s&&s.copy(Hh).addScaledVector(Fa,d),f}intersectSphere(e,t){Pn.subVectors(e.center,this.origin);let i=Pn.dot(this.direction),s=Pn.dot(Pn)-i*i,r=e.radius*e.radius;if(s>r)return null;let o=Math.sqrt(r-s),a=i-o,c=i+o;return c<0?null:a<0?this.at(c,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){let t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;let i=-(this.origin.dot(e.normal)+e.constant)/t;return i>=0?i:null}intersectPlane(e,t){let i=this.distanceToPlane(e);return i===null?null:this.at(i,t)}intersectsPlane(e){let t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let i,s,r,o,a,c,l=1/this.direction.x,h=1/this.direction.y,u=1/this.direction.z,d=this.origin;return l>=0?(i=(e.min.x-d.x)*l,s=(e.max.x-d.x)*l):(i=(e.max.x-d.x)*l,s=(e.min.x-d.x)*l),h>=0?(r=(e.min.y-d.y)*h,o=(e.max.y-d.y)*h):(r=(e.max.y-d.y)*h,o=(e.min.y-d.y)*h),i>o||r>s||((r>i||isNaN(i))&&(i=r),(o<s||isNaN(s))&&(s=o),u>=0?(a=(e.min.z-d.z)*u,c=(e.max.z-d.z)*u):(a=(e.max.z-d.z)*u,c=(e.min.z-d.z)*u),i>c||a>s)||((a>i||i!==i)&&(i=a),(c<s||s!==s)&&(s=c),s<0)?null:this.at(i>=0?i:s,t)}intersectsBox(e){return this.intersectBox(e,Pn)!==null}intersectTriangle(e,t,i,s,r){Vh.subVectors(t,e),Oa.subVectors(i,e),Gh.crossVectors(Vh,Oa);let o=this.direction.dot(Gh),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;ts.subVectors(this.origin,e);let c=a*this.direction.dot(Oa.crossVectors(ts,Oa));if(c<0)return null;let l=a*this.direction.dot(Vh.cross(ts));if(l<0||c+l>o)return null;let h=-a*ts.dot(Gh);return h<0?null:this.at(h/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},Ln=class extends yi{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Te(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ii,this.combine=Jl,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}},Qd=new rt,ws=new Dn,Ba=new vi,ef=new P,za=new P,ka=new P,Ha=new P,Wh=new P,Va=new P,tf=new P,Ga=new P,Ke=class extends ft{constructor(e=new ut,t=new Ln){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){let t=this.geometry.morphAttributes,i=Object.keys(t);if(i.length>0){let s=t[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){let a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){let i=this.geometry,s=i.attributes.position,r=i.morphAttributes.position,o=i.morphTargetsRelative;t.fromBufferAttribute(s,e);let a=this.morphTargetInfluences;if(r&&a){Va.set(0,0,0);for(let c=0,l=r.length;c<l;c++){let h=a[c],u=r[c];h!==0&&(Wh.fromBufferAttribute(u,e),o?Va.addScaledVector(Wh,h):Va.addScaledVector(Wh.sub(t),h))}t.add(Va)}return t}raycast(e,t){let i=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),Ba.copy(i.boundingSphere),Ba.applyMatrix4(r),ws.copy(e.ray).recast(e.near),!(Ba.containsPoint(ws.origin)===!1&&(ws.intersectSphere(Ba,ef)===null||ws.origin.distanceToSquared(ef)>(e.far-e.near)**2))&&(Qd.copy(r).invert(),ws.copy(e.ray).applyMatrix4(Qd),!(i.boundingBox!==null&&ws.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(e,t,ws)))}_computeIntersections(e,t,i){let s,r=this.geometry,o=this.material,a=r.index,c=r.attributes.position,l=r.attributes.uv,h=r.attributes.uv1,u=r.attributes.normal,d=r.groups,f=r.drawRange;if(a!==null)if(Array.isArray(o))for(let g=0,x=d.length;g<x;g++){let p=d[g],m=o[p.materialIndex],M=Math.max(p.start,f.start),b=Math.min(a.count,Math.min(p.start+p.count,f.start+f.count));for(let y=M,T=b;y<T;y+=3){let S=a.getX(y),A=a.getX(y+1),_=a.getX(y+2);s=Wa(this,m,e,i,l,h,u,S,A,_),s&&(s.faceIndex=Math.floor(y/3),s.face.materialIndex=p.materialIndex,t.push(s))}}else{let g=Math.max(0,f.start),x=Math.min(a.count,f.start+f.count);for(let p=g,m=x;p<m;p+=3){let M=a.getX(p),b=a.getX(p+1),y=a.getX(p+2);s=Wa(this,o,e,i,l,h,u,M,b,y),s&&(s.faceIndex=Math.floor(p/3),t.push(s))}}else if(c!==void 0)if(Array.isArray(o))for(let g=0,x=d.length;g<x;g++){let p=d[g],m=o[p.materialIndex],M=Math.max(p.start,f.start),b=Math.min(c.count,Math.min(p.start+p.count,f.start+f.count));for(let y=M,T=b;y<T;y+=3){let S=y,A=y+1,_=y+2;s=Wa(this,m,e,i,l,h,u,S,A,_),s&&(s.faceIndex=Math.floor(y/3),s.face.materialIndex=p.materialIndex,t.push(s))}}else{let g=Math.max(0,f.start),x=Math.min(c.count,f.start+f.count);for(let p=g,m=x;p<m;p+=3){let M=p,b=p+1,y=p+2;s=Wa(this,o,e,i,l,h,u,M,b,y),s&&(s.faceIndex=Math.floor(p/3),t.push(s))}}}};function Pg(n,e,t,i,s,r,o,a){let c;if(e.side===ti?c=i.intersectTriangle(o,r,s,!0,a):c=i.intersectTriangle(s,r,o,e.side===Zi,a),c===null)return null;Ga.copy(a),Ga.applyMatrix4(n.matrixWorld);let l=t.ray.origin.distanceTo(Ga);return l<t.near||l>t.far?null:{distance:l,point:Ga.clone(),object:n}}function Wa(n,e,t,i,s,r,o,a,c,l){n.getVertexPosition(a,za),n.getVertexPosition(c,ka),n.getVertexPosition(l,Ha);let h=Pg(n,e,t,i,za,ka,Ha,tf);if(h){let u=new P;hn.getBarycoord(tf,za,ka,Ha,u),s&&(h.uv=hn.getInterpolatedAttribute(s,a,c,l,u,new $)),r&&(h.uv1=hn.getInterpolatedAttribute(r,a,c,l,u,new $)),o&&(h.normal=hn.getInterpolatedAttribute(o,a,c,l,u,new P),h.normal.dot(i.direction)>0&&h.normal.multiplyScalar(-1));let d={a,b:c,c:l,normal:new P,materialIndex:0};hn.getNormal(za,ka,Ha,d.normal),h.face=d,h.barycoord=u}return h}var Un=class extends fi{constructor(e=null,t=1,i=1,s,r,o,a,c,l=Ot,h=Ot,u,d){super(null,o,a,c,l,h,s,r,u,d),this.isDataTexture=!0,this.image={data:e,width:t,height:i},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var xo=class extends Ut{constructor(e,t,i,s=1){super(e,t,i),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){let e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}},cr=new rt,nf=new rt,Xa=[],sf=new pi,Ig=new rt,Qr=new Ke,eo=new vi,jt=class extends Ke{constructor(e,t,i){super(e,t),this.isInstancedMesh=!0,this.instanceMatrix=new xo(new Float32Array(i*16),16),this.instanceColor=null,this.morphTexture=null,this.count=i,this.boundingBox=null,this.boundingSphere=null;for(let s=0;s<i;s++)this.setMatrixAt(s,Ig)}computeBoundingBox(){let e=this.geometry,t=this.count;this.boundingBox===null&&(this.boundingBox=new pi),e.boundingBox===null&&e.computeBoundingBox(),this.boundingBox.makeEmpty();for(let i=0;i<t;i++)this.getMatrixAt(i,cr),sf.copy(e.boundingBox).applyMatrix4(cr),this.boundingBox.union(sf)}computeBoundingSphere(){let e=this.geometry,t=this.count;this.boundingSphere===null&&(this.boundingSphere=new vi),e.boundingSphere===null&&e.computeBoundingSphere(),this.boundingSphere.makeEmpty();for(let i=0;i<t;i++)this.getMatrixAt(i,cr),eo.copy(e.boundingSphere).applyMatrix4(cr),this.boundingSphere.union(eo)}copy(e,t){return super.copy(e,t),this.instanceMatrix.copy(e.instanceMatrix),e.morphTexture!==null&&(this.morphTexture=e.morphTexture.clone()),e.instanceColor!==null&&(this.instanceColor=e.instanceColor.clone()),this.count=e.count,e.boundingBox!==null&&(this.boundingBox=e.boundingBox.clone()),e.boundingSphere!==null&&(this.boundingSphere=e.boundingSphere.clone()),this}getColorAt(e,t){return this.instanceColor===null?t.setRGB(1,1,1):t.fromArray(this.instanceColor.array,e*3)}getMatrixAt(e,t){return t.fromArray(this.instanceMatrix.array,e*16)}getMorphAt(e,t){let i=t.morphTargetInfluences,s=this.morphTexture.source.data.data,r=i.length+1,o=e*r+1;for(let a=0;a<i.length;a++)i[a]=s[o+a]}raycast(e,t){let i=this.matrixWorld,s=this.count;if(Qr.geometry=this.geometry,Qr.material=this.material,Qr.material!==void 0&&(this.boundingSphere===null&&this.computeBoundingSphere(),eo.copy(this.boundingSphere),eo.applyMatrix4(i),e.ray.intersectsSphere(eo)!==!1))for(let r=0;r<s;r++){this.getMatrixAt(r,cr),nf.multiplyMatrices(i,cr),Qr.matrixWorld=nf,Qr.raycast(e,Xa);for(let o=0,a=Xa.length;o<a;o++){let c=Xa[o];c.instanceId=r,c.object=this,t.push(c)}Xa.length=0}}setColorAt(e,t){return this.instanceColor===null&&(this.instanceColor=new xo(new Float32Array(this.instanceMatrix.count*3).fill(1),3)),t.toArray(this.instanceColor.array,e*3),this}setMatrixAt(e,t){return t.toArray(this.instanceMatrix.array,e*16),this}setMorphAt(e,t){let i=t.morphTargetInfluences,s=i.length+1;this.morphTexture===null&&(this.morphTexture=new Un(new Float32Array(s*this.count),s,this.count,nc,Hi));let r=this.morphTexture.source.data.data,o=0;for(let l=0;l<i.length;l++)o+=i[l];let a=this.geometry.morphTargetsRelative?1:1-o,c=s*e;return r[c]=a,r.set(i,c+1),this}updateMorphTargets(){}dispose(){this.dispatchEvent({type:"dispose"}),this.morphTexture!==null&&(this.morphTexture.dispose(),this.morphTexture=null)}},Xh=new P,Dg=new P,Lg=new tt,Bi=class{constructor(e=new P(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,i,s){return this.normal.set(e,t,i),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,i){let s=Xh.subVectors(i,t).cross(Dg.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){let e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t,i=!0){let s=e.delta(Xh),r=this.normal.dot(s);if(r===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;let o=-(e.start.dot(this.normal)+this.constant)/r;return i===!0&&(o<0||o>1)?null:t.copy(e.start).addScaledVector(s,o)}intersectsLine(e){let t=this.distanceToPoint(e.start),i=this.distanceToPoint(e.end);return t<0&&i>0||i<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){let i=t||Lg.getNormalMatrix(e),s=this.coplanarPoint(Xh).applyMatrix4(e),r=this.normal.applyMatrix3(i).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}},Ts=new vi,Ug=new $(.5,.5),qa=new P,br=class{constructor(e=new Bi,t=new Bi,i=new Bi,s=new Bi,r=new Bi,o=new Bi){this.planes=[e,t,i,s,r,o]}set(e,t,i,s,r,o){let a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(i),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){let t=this.planes;for(let i=0;i<6;i++)t[i].copy(e.planes[i]);return this}setFromProjectionMatrix(e,t=$i,i=!1){let s=this.planes,r=e.elements,o=r[0],a=r[1],c=r[2],l=r[3],h=r[4],u=r[5],d=r[6],f=r[7],g=r[8],x=r[9],p=r[10],m=r[11],M=r[12],b=r[13],y=r[14],T=r[15];if(s[0].setComponents(l-o,f-h,m-g,T-M).normalize(),s[1].setComponents(l+o,f+h,m+g,T+M).normalize(),s[2].setComponents(l+a,f+u,m+x,T+b).normalize(),s[3].setComponents(l-a,f-u,m-x,T-b).normalize(),i)s[4].setComponents(c,d,p,y).normalize(),s[5].setComponents(l-c,f-d,m-p,T-y).normalize();else if(s[4].setComponents(l-c,f-d,m-p,T-y).normalize(),t===$i)s[5].setComponents(l+c,f+d,m+p,T+y).normalize();else if(t===gr)s[5].setComponents(c,d,p,y).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Ts.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{let t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Ts.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Ts)}intersectsSprite(e){Ts.center.set(0,0,0);let t=Ug.distanceTo(e.center);return Ts.radius=.7071067811865476+t,Ts.applyMatrix4(e.matrixWorld),this.intersectsSphere(Ts)}intersectsSphere(e){let t=this.planes,i=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(i)<s)return!1;return!0}intersectsBox(e){let t=this.planes;for(let i=0;i<6;i++){let s=t[i];if(qa.x=s.normal.x>0?e.max.x:e.min.x,qa.y=s.normal.y>0?e.max.y:e.min.y,qa.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(qa)<0)return!1}return!0}containsPoint(e){let t=this.planes;for(let i=0;i<6;i++)if(t[i].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}};var Nn=class extends yi{constructor(e){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new Te(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.linewidth=e.linewidth,this.linecap=e.linecap,this.linejoin=e.linejoin,this.fog=e.fog,this}},Sl=new P,bl=new P,rf=new rt,to=new Dn,Ya=new vi,qh=new P,of=new P,El=class extends ft{constructor(e=new ut,t=new Nn){super(),this.isLine=!0,this.type="Line",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}computeLineDistances(){let e=this.geometry;if(e.index===null){let t=e.attributes.position,i=[0];for(let s=1,r=t.count;s<r;s++)Sl.fromBufferAttribute(t,s-1),bl.fromBufferAttribute(t,s),i[s]=i[s-1],i[s]+=Sl.distanceTo(bl);e.setAttribute("lineDistance",new it(i,1))}else $e("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(e,t){let i=this.geometry,s=this.matrixWorld,r=e.params.Line.threshold,o=i.drawRange;if(i.boundingSphere===null&&i.computeBoundingSphere(),Ya.copy(i.boundingSphere),Ya.applyMatrix4(s),Ya.radius+=r,e.ray.intersectsSphere(Ya)===!1)return;rf.copy(s).invert(),to.copy(e.ray).applyMatrix4(rf);let a=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=a*a,l=this.isLineSegments?2:1,h=i.index,d=i.attributes.position;if(h!==null){let f=Math.max(0,o.start),g=Math.min(h.count,o.start+o.count);for(let x=f,p=g-1;x<p;x+=l){let m=h.getX(x),M=h.getX(x+1),b=$a(this,e,to,c,m,M,x);b&&t.push(b)}if(this.isLineLoop){let x=h.getX(g-1),p=h.getX(f),m=$a(this,e,to,c,x,p,g-1);m&&t.push(m)}}else{let f=Math.max(0,o.start),g=Math.min(d.count,o.start+o.count);for(let x=f,p=g-1;x<p;x+=l){let m=$a(this,e,to,c,x,x+1,x);m&&t.push(m)}if(this.isLineLoop){let x=$a(this,e,to,c,g-1,f,g-1);x&&t.push(x)}}}updateMorphTargets(){let t=this.geometry.morphAttributes,i=Object.keys(t);if(i.length>0){let s=t[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){let a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}};function $a(n,e,t,i,s,r,o){let a=n.geometry.attributes.position;if(Sl.fromBufferAttribute(a,s),bl.fromBufferAttribute(a,r),t.distanceSqToSegment(Sl,bl,qh,of)>i)return;qh.applyMatrix4(n.matrixWorld);let l=e.ray.origin.distanceTo(qh);if(!(l<e.near||l>e.far))return{distance:l,point:of.clone().applyMatrix4(n.matrixWorld),index:o,face:null,faceIndex:null,barycoord:null,object:n}}var af=new P,lf=new P,ns=class extends El{constructor(e,t){super(e,t),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){let e=this.geometry;if(e.index===null){let t=e.attributes.position,i=[];for(let s=0,r=t.count;s<r;s+=2)af.fromBufferAttribute(t,s),lf.fromBufferAttribute(t,s+1),i[s]=s===0?0:i[s-1],i[s+1]=i[s]+af.distanceTo(lf);e.setAttribute("lineDistance",new it(i,1))}else $e("LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}};var wl=class extends yi{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new Te(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}},cf=new rt,su=new Dn,Za=new vi,Ja=new P,vo=class extends ft{constructor(e=new ut,t=new wl){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){let i=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,o=i.drawRange;if(i.boundingSphere===null&&i.computeBoundingSphere(),Za.copy(i.boundingSphere),Za.applyMatrix4(s),Za.radius+=r,e.ray.intersectsSphere(Za)===!1)return;cf.copy(s).invert(),su.copy(e.ray).applyMatrix4(cf);let a=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=a*a,l=i.index,u=i.attributes.position;if(l!==null){let d=Math.max(0,o.start),f=Math.min(l.count,o.start+o.count);for(let g=d,x=f;g<x;g++){let p=l.getX(g);Ja.fromBufferAttribute(u,p),hf(Ja,p,c,s,e,t,this)}}else{let d=Math.max(0,o.start),f=Math.min(u.count,o.start+o.count);for(let g=d,x=f;g<x;g++)Ja.fromBufferAttribute(u,g),hf(Ja,g,c,s,e,t,this)}}updateMorphTargets(){let t=this.geometry.morphAttributes,i=Object.keys(t);if(i.length>0){let s=t[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){let a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}};function hf(n,e,t,i,s,r,o){let a=su.distanceSqToPoint(n);if(a<t){let c=new P;su.closestPointToPoint(n,c),c.applyMatrix4(i);let l=s.ray.origin.distanceTo(c);if(l<s.near||l>s.far)return;r.push({distance:l,distanceToRay:Math.sqrt(a),point:c,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}var yo=class extends fi{constructor(e=[],t=ds,i,s,r,o,a,c,l,h){super(e,t,i,s,r,o,a,c,l,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}},pn=class extends fi{constructor(e,t,i,s,r,o,a,c,l){super(e,t,i,s,r,o,a,c,l),this.isCanvasTexture=!0,this.needsUpdate=!0}};var ji=class extends fi{constructor(e,t,i=tn,s,r,o,a=Ot,c=Ot,l,h=fn,u=1){if(h!==fn&&h!==_n)throw new Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");let d={width:e,height:t,depth:u};super(d,s,r,o,a,c,h,i,l),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new vr(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){let t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}},Tl=class extends ji{constructor(e,t=tn,i=ds,s,r,o=Ot,a=Ot,c,l=fn){let h={width:e,height:e,depth:1},u=[h,h,h,h,h,h];super(e,e,t,i,s,r,o,a,c,l),this.image=u,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}},Mo=class extends fi{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}},Bt=class n extends ut{constructor(e=1,t=1,i=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:i,widthSegments:s,heightSegments:r,depthSegments:o};let a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);let c=[],l=[],h=[],u=[],d=0,f=0;g("z","y","x",-1,-1,i,t,e,o,r,0),g("z","y","x",1,-1,i,t,-e,o,r,1),g("x","z","y",1,1,e,i,t,s,o,2),g("x","z","y",1,-1,e,i,-t,s,o,3),g("x","y","z",1,-1,e,t,i,s,r,4),g("x","y","z",-1,-1,e,t,-i,s,r,5),this.setIndex(c),this.setAttribute("position",new it(l,3)),this.setAttribute("normal",new it(h,3)),this.setAttribute("uv",new it(u,2));function g(x,p,m,M,b,y,T,S,A,_,E){let C=y/A,I=T/_,L=y/2,V=T/2,q=S/2,N=A+1,Y=_+1,X=0,ne=0,ie=new P;for(let ge=0;ge<Y;ge++){let ue=ge*I-V;for(let xe=0;xe<N;xe++){let Ne=xe*C-L;ie[x]=Ne*M,ie[p]=ue*b,ie[m]=q,l.push(ie.x,ie.y,ie.z),ie[x]=0,ie[p]=0,ie[m]=S>0?1:-1,h.push(ie.x,ie.y,ie.z),u.push(xe/A),u.push(1-ge/_),X+=1}}for(let ge=0;ge<_;ge++)for(let ue=0;ue<A;ue++){let xe=d+ue+N*ge,Ne=d+ue+N*(ge+1),st=d+(ue+1)+N*(ge+1),Xe=d+(ue+1)+N*ge;c.push(xe,Ne,Xe),c.push(Ne,st,Xe),ne+=6}a.addGroup(f,ne,E),f+=ne,d+=X}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}};var Ki=class n extends ut{constructor(e=1,t=1,i=1,s=32,r=1,o=!1,a=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:i,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:c};let l=this;s=Math.floor(s),r=Math.floor(r);let h=[],u=[],d=[],f=[],g=0,x=[],p=i/2,m=0;M(),o===!1&&(e>0&&b(!0),t>0&&b(!1)),this.setIndex(h),this.setAttribute("position",new it(u,3)),this.setAttribute("normal",new it(d,3)),this.setAttribute("uv",new it(f,2));function M(){let y=new P,T=new P,S=0,A=(t-e)/i;for(let _=0;_<=r;_++){let E=[],C=_/r,I=C*(t-e)+e;for(let L=0;L<=s;L++){let V=L/s,q=V*c+a,N=Math.sin(q),Y=Math.cos(q);T.x=I*N,T.y=-C*i+p,T.z=I*Y,u.push(T.x,T.y,T.z),y.set(N,A,Y).normalize(),d.push(y.x,y.y,y.z),f.push(V,1-C),E.push(g++)}x.push(E)}for(let _=0;_<s;_++)for(let E=0;E<r;E++){let C=x[E][_],I=x[E+1][_],L=x[E+1][_+1],V=x[E][_+1];(e>0||E!==0)&&(h.push(C,I,V),S+=3),(t>0||E!==r-1)&&(h.push(I,L,V),S+=3)}l.addGroup(m,S,0),m+=S}function b(y){let T=g,S=new $,A=new P,_=0,E=y===!0?e:t,C=y===!0?1:-1;for(let L=1;L<=s;L++)u.push(0,p*C,0),d.push(0,C,0),f.push(.5,.5),g++;let I=g;for(let L=0;L<=s;L++){let q=L/s*c+a,N=Math.cos(q),Y=Math.sin(q);A.x=E*Y,A.y=p*C,A.z=E*N,u.push(A.x,A.y,A.z),d.push(0,C,0),S.x=N*.5+.5,S.y=Y*.5*C+.5,f.push(S.x,S.y),g++}for(let L=0;L<s;L++){let V=T+L,q=I+L;y===!0?h.push(q,q+1,V):h.push(q+1,q,V),_+=3}l.addGroup(m,_,y===!0?1:2),m+=_}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}},Er=class n extends Ki{constructor(e=1,t=1,i=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,i,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:i,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new n(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}},So=class n extends ut{constructor(e=[],t=[],i=1,s=0){super(),this.type="PolyhedronGeometry",this.parameters={vertices:e,indices:t,radius:i,detail:s};let r=[],o=[];a(s),l(i),h(),this.setAttribute("position",new it(r,3)),this.setAttribute("normal",new it(r.slice(),3)),this.setAttribute("uv",new it(o,2)),s===0?this.computeVertexNormals():this.normalizeNormals();function a(M){let b=new P,y=new P,T=new P;for(let S=0;S<t.length;S+=3)f(t[S+0],b),f(t[S+1],y),f(t[S+2],T),c(b,y,T,M)}function c(M,b,y,T){let S=T+1,A=[];for(let _=0;_<=S;_++){A[_]=[];let E=M.clone().lerp(y,_/S),C=b.clone().lerp(y,_/S),I=S-_;for(let L=0;L<=I;L++)L===0&&_===S?A[_][L]=E:A[_][L]=E.clone().lerp(C,L/I)}for(let _=0;_<S;_++)for(let E=0;E<2*(S-_)-1;E++){let C=Math.floor(E/2);E%2===0?(d(A[_][C+1]),d(A[_+1][C]),d(A[_][C])):(d(A[_][C+1]),d(A[_+1][C+1]),d(A[_+1][C]))}}function l(M){let b=new P;for(let y=0;y<r.length;y+=3)b.x=r[y+0],b.y=r[y+1],b.z=r[y+2],b.normalize().multiplyScalar(M),r[y+0]=b.x,r[y+1]=b.y,r[y+2]=b.z}function h(){let M=new P;for(let b=0;b<r.length;b+=3){M.x=r[b+0],M.y=r[b+1],M.z=r[b+2];let y=p(M)/2/Math.PI+.5,T=m(M)/Math.PI+.5;o.push(y,1-T)}g(),u()}function u(){for(let M=0;M<o.length;M+=6){let b=o[M+0],y=o[M+2],T=o[M+4],S=Math.max(b,y,T),A=Math.min(b,y,T);S>.9&&A<.1&&(b<.2&&(o[M+0]+=1),y<.2&&(o[M+2]+=1),T<.2&&(o[M+4]+=1))}}function d(M){r.push(M.x,M.y,M.z)}function f(M,b){let y=M*3;b.x=e[y+0],b.y=e[y+1],b.z=e[y+2]}function g(){let M=new P,b=new P,y=new P,T=new P,S=new $,A=new $,_=new $;for(let E=0,C=0;E<r.length;E+=9,C+=6){M.set(r[E+0],r[E+1],r[E+2]),b.set(r[E+3],r[E+4],r[E+5]),y.set(r[E+6],r[E+7],r[E+8]),S.set(o[C+0],o[C+1]),A.set(o[C+2],o[C+3]),_.set(o[C+4],o[C+5]),T.copy(M).add(b).add(y).divideScalar(3);let I=p(T);x(S,C+0,M,I),x(A,C+2,b,I),x(_,C+4,y,I)}}function x(M,b,y,T){T<0&&M.x===1&&(o[b]=M.x-1),y.x===0&&y.z===0&&(o[b]=T/2/Math.PI+.5)}function p(M){return Math.atan2(M.z,-M.x)}function m(M){return Math.atan2(-M.y,Math.sqrt(M.x*M.x+M.z*M.z))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.vertices,e.indices,e.radius,e.detail)}},bo=class n extends So{constructor(e=1,t=0){let i=(1+Math.sqrt(5))/2,s=1/i,r=[-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,0,-s,-i,0,-s,i,0,s,-i,0,s,i,-s,-i,0,-s,i,0,s,-i,0,s,i,0,-i,0,-s,i,0,-s,-i,0,s,i,0,s],o=[3,11,7,3,7,15,3,15,13,7,19,17,7,17,6,7,6,15,17,4,8,17,8,10,17,10,6,8,0,16,8,16,2,8,2,10,0,12,1,0,1,18,0,18,16,6,10,2,6,2,13,6,13,15,2,16,18,2,18,3,2,3,13,18,1,9,18,9,11,18,11,3,4,14,12,4,12,0,4,0,8,11,9,5,11,5,19,11,19,7,19,5,14,19,14,4,19,4,17,1,12,14,1,14,5,1,5,9];super(r,o,e,t),this.type="DodecahedronGeometry",this.parameters={radius:e,detail:t}}static fromJSON(e){return new n(e.radius,e.detail)}},ja=new P,Ka=new P,Yh=new P,Qa=new hn,Eo=class extends ut{constructor(e=null,t=1){if(super(),this.type="EdgesGeometry",this.parameters={geometry:e,thresholdAngle:t},e!==null){let s=Math.pow(10,4),r=Math.cos(pr*t),o=e.getIndex(),a=e.getAttribute("position"),c=o?o.count:a.count,l=[0,0,0],h=["a","b","c"],u=new Array(3),d={},f=[];for(let g=0;g<c;g+=3){o?(l[0]=o.getX(g),l[1]=o.getX(g+1),l[2]=o.getX(g+2)):(l[0]=g,l[1]=g+1,l[2]=g+2);let{a:x,b:p,c:m}=Qa;if(x.fromBufferAttribute(a,l[0]),p.fromBufferAttribute(a,l[1]),m.fromBufferAttribute(a,l[2]),Qa.getNormal(Yh),u[0]=`${Math.round(x.x*s)},${Math.round(x.y*s)},${Math.round(x.z*s)}`,u[1]=`${Math.round(p.x*s)},${Math.round(p.y*s)},${Math.round(p.z*s)}`,u[2]=`${Math.round(m.x*s)},${Math.round(m.y*s)},${Math.round(m.z*s)}`,!(u[0]===u[1]||u[1]===u[2]||u[2]===u[0]))for(let M=0;M<3;M++){let b=(M+1)%3,y=u[M],T=u[b],S=Qa[h[M]],A=Qa[h[b]],_=`${y}_${T}`,E=`${T}_${y}`;E in d&&d[E]?(Yh.dot(d[E].normal)<=r&&(f.push(S.x,S.y,S.z),f.push(A.x,A.y,A.z)),d[E]=null):_ in d||(d[_]={index0:l[M],index1:l[b],normal:Yh.clone()})}}for(let g in d)if(d[g]){let{index0:x,index1:p}=d[g];ja.fromBufferAttribute(a,x),Ka.fromBufferAttribute(a,p),f.push(ja.x,ja.y,ja.z),f.push(Ka.x,Ka.y,Ka.z)}this.setAttribute("position",new it(f,3))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}},Li=class{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){$e("Curve: .getPoint() not implemented.")}getPointAt(e,t){let i=this.getUtoTmapping(e);return this.getPoint(i,t)}getPoints(e=5){let t=[];for(let i=0;i<=e;i++)t.push(this.getPoint(i/e));return t}getSpacedPoints(e=5){let t=[];for(let i=0;i<=e;i++)t.push(this.getPointAt(i/e));return t}getLength(){let e=this.getLengths();return e[e.length-1]}getLengths(e=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===e+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;let t=[],i,s=this.getPoint(0),r=0;t.push(0);for(let o=1;o<=e;o++)i=this.getPoint(o/e),r+=i.distanceTo(s),t.push(r),s=i;return this.cacheArcLengths=t,t}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(e,t=null){let i=this.getLengths(),s=0,r=i.length,o;t?o=t:o=e*i[r-1];let a=0,c=r-1,l;for(;a<=c;)if(s=Math.floor(a+(c-a)/2),l=i[s]-o,l<0)a=s+1;else if(l>0)c=s-1;else{c=s;break}if(s=c,i[s]===o)return s/(r-1);let h=i[s],d=i[s+1]-h,f=(o-h)/d;return(s+f)/(r-1)}getTangent(e,t){let s=e-1e-4,r=e+1e-4;s<0&&(s=0),r>1&&(r=1);let o=this.getPoint(s),a=this.getPoint(r),c=t||(o.isVector2?new $:new P);return c.copy(a).sub(o).normalize(),c}getTangentAt(e,t){let i=this.getUtoTmapping(e);return this.getTangent(i,t)}computeFrenetFrames(e,t=!1){let i=new P,s=[],r=[],o=[],a=new P,c=new rt;for(let f=0;f<=e;f++){let g=f/e;s[f]=this.getTangentAt(g,new P)}r[0]=new P,o[0]=new P;let l=Number.MAX_VALUE,h=Math.abs(s[0].x),u=Math.abs(s[0].y),d=Math.abs(s[0].z);h<=l&&(l=h,i.set(1,0,0)),u<=l&&(l=u,i.set(0,1,0)),d<=l&&i.set(0,0,1),a.crossVectors(s[0],i).normalize(),r[0].crossVectors(s[0],a),o[0].crossVectors(s[0],r[0]);for(let f=1;f<=e;f++){if(r[f]=r[f-1].clone(),o[f]=o[f-1].clone(),a.crossVectors(s[f-1],s[f]),a.length()>Number.EPSILON){a.normalize();let g=Math.acos(je(s[f-1].dot(s[f]),-1,1));r[f].applyMatrix4(c.makeRotationAxis(a,g))}o[f].crossVectors(s[f],r[f])}if(t===!0){let f=Math.acos(je(r[0].dot(r[e]),-1,1));f/=e,s[0].dot(a.crossVectors(r[0],r[e]))>0&&(f=-f);for(let g=1;g<=e;g++)r[g].applyMatrix4(c.makeRotationAxis(s[g],f*g)),o[g].crossVectors(s[g],r[g])}return{tangents:s,normals:r,binormals:o}}clone(){return new this.constructor().copy(this)}copy(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}toJSON(){let e={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return e.arcLengthDivisions=this.arcLengthDivisions,e.type=this.type,e}fromJSON(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}},wr=class extends Li{constructor(e=0,t=0,i=1,s=1,r=0,o=Math.PI*2,a=!1,c=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=e,this.aY=t,this.xRadius=i,this.yRadius=s,this.aStartAngle=r,this.aEndAngle=o,this.aClockwise=a,this.aRotation=c}getPoint(e,t=new $){let i=t,s=Math.PI*2,r=this.aEndAngle-this.aStartAngle,o=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=s;for(;r>s;)r-=s;r<Number.EPSILON&&(o?r=0:r=s),this.aClockwise===!0&&!o&&(r===s?r=-s:r=r-s);let a=this.aStartAngle+e*r,c=this.aX+this.xRadius*Math.cos(a),l=this.aY+this.yRadius*Math.sin(a);if(this.aRotation!==0){let h=Math.cos(this.aRotation),u=Math.sin(this.aRotation),d=c-this.aX,f=l-this.aY;c=d*h-f*u+this.aX,l=d*u+f*h+this.aY}return i.set(c,l)}copy(e){return super.copy(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}toJSON(){let e=super.toJSON();return e.aX=this.aX,e.aY=this.aY,e.xRadius=this.xRadius,e.yRadius=this.yRadius,e.aStartAngle=this.aStartAngle,e.aEndAngle=this.aEndAngle,e.aClockwise=this.aClockwise,e.aRotation=this.aRotation,e}fromJSON(e){return super.fromJSON(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}},Al=class extends wr{constructor(e,t,i,s,r,o){super(e,t,i,i,s,r,o),this.isArcCurve=!0,this.type="ArcCurve"}};function Au(){let n=0,e=0,t=0,i=0;function s(r,o,a,c){n=r,e=a,t=-3*r+3*o-2*a-c,i=2*r-2*o+a+c}return{initCatmullRom:function(r,o,a,c,l){s(o,a,l*(a-r),l*(c-o))},initNonuniformCatmullRom:function(r,o,a,c,l,h,u){let d=(o-r)/l-(a-r)/(l+h)+(a-o)/h,f=(a-o)/h-(c-o)/(h+u)+(c-a)/u;d*=h,f*=h,s(o,a,d,f)},calc:function(r){let o=r*r,a=o*r;return n+e*r+t*o+i*a}}}var uf=new P,df=new P,$h=new Au,Zh=new Au,Jh=new Au,Rl=class extends Li{constructor(e=[],t=!1,i="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=e,this.closed=t,this.curveType=i,this.tension=s}getPoint(e,t=new P){let i=t,s=this.points,r=s.length,o=(r-(this.closed?0:1))*e,a=Math.floor(o),c=o-a;this.closed?a+=a>0?0:(Math.floor(Math.abs(a)/r)+1)*r:c===0&&a===r-1&&(a=r-2,c=1);let l,h;this.closed||a>0?l=s[(a-1)%r]:(df.subVectors(s[0],s[1]).add(s[0]),l=df);let u=s[a%r],d=s[(a+1)%r];if(this.closed||a+2<r?h=s[(a+2)%r]:(uf.subVectors(s[r-1],s[r-2]).add(s[r-1]),h=uf),this.curveType==="centripetal"||this.curveType==="chordal"){let f=this.curveType==="chordal"?.5:.25,g=Math.pow(l.distanceToSquared(u),f),x=Math.pow(u.distanceToSquared(d),f),p=Math.pow(d.distanceToSquared(h),f);x<1e-4&&(x=1),g<1e-4&&(g=x),p<1e-4&&(p=x),$h.initNonuniformCatmullRom(l.x,u.x,d.x,h.x,g,x,p),Zh.initNonuniformCatmullRom(l.y,u.y,d.y,h.y,g,x,p),Jh.initNonuniformCatmullRom(l.z,u.z,d.z,h.z,g,x,p)}else this.curveType==="catmullrom"&&($h.initCatmullRom(l.x,u.x,d.x,h.x,this.tension),Zh.initCatmullRom(l.y,u.y,d.y,h.y,this.tension),Jh.initCatmullRom(l.z,u.z,d.z,h.z,this.tension));return i.set($h.calc(c),Zh.calc(c),Jh.calc(c)),i}copy(e){super.copy(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(s.clone())}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}toJSON(){let e=super.toJSON();e.points=[];for(let t=0,i=this.points.length;t<i;t++){let s=this.points[t];e.points.push(s.toArray())}return e.closed=this.closed,e.curveType=this.curveType,e.tension=this.tension,e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(new P().fromArray(s))}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}};function ff(n,e,t,i,s){let r=(i-e)*.5,o=(s-t)*.5,a=n*n,c=n*a;return(2*t-2*i+r+o)*c+(-3*t+3*i-2*r-o)*a+r*n+t}function Ng(n,e){let t=1-n;return t*t*e}function Fg(n,e){return 2*(1-n)*n*e}function Og(n,e){return n*n*e}function so(n,e,t,i){return Ng(n,e)+Fg(n,t)+Og(n,i)}function Bg(n,e){let t=1-n;return t*t*t*e}function zg(n,e){let t=1-n;return 3*t*t*n*e}function kg(n,e){return 3*(1-n)*n*n*e}function Hg(n,e){return n*n*n*e}function ro(n,e,t,i,s){return Bg(n,e)+zg(n,t)+kg(n,i)+Hg(n,s)}var wo=class extends Li{constructor(e=new $,t=new $,i=new $,s=new $){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=e,this.v1=t,this.v2=i,this.v3=s}getPoint(e,t=new $){let i=t,s=this.v0,r=this.v1,o=this.v2,a=this.v3;return i.set(ro(e,s.x,r.x,o.x,a.x),ro(e,s.y,r.y,o.y,a.y)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}},Cl=class extends Li{constructor(e=new P,t=new P,i=new P,s=new P){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=e,this.v1=t,this.v2=i,this.v3=s}getPoint(e,t=new P){let i=t,s=this.v0,r=this.v1,o=this.v2,a=this.v3;return i.set(ro(e,s.x,r.x,o.x,a.x),ro(e,s.y,r.y,o.y,a.y),ro(e,s.z,r.z,o.z,a.z)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}},To=class extends Li{constructor(e=new $,t=new $){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=e,this.v2=t}getPoint(e,t=new $){let i=t;return e===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(e).add(this.v1)),i}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new $){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Pl=class extends Li{constructor(e=new P,t=new P){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=e,this.v2=t}getPoint(e,t=new P){let i=t;return e===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(e).add(this.v1)),i}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new P){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Ao=class extends Li{constructor(e=new $,t=new $,i=new $){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=e,this.v1=t,this.v2=i}getPoint(e,t=new $){let i=t,s=this.v0,r=this.v1,o=this.v2;return i.set(so(e,s.x,r.x,o.x),so(e,s.y,r.y,o.y)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Il=class extends Li{constructor(e=new P,t=new P,i=new P){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=e,this.v1=t,this.v2=i}getPoint(e,t=new P){let i=t,s=this.v0,r=this.v1,o=this.v2;return i.set(so(e,s.x,r.x,o.x),so(e,s.y,r.y,o.y),so(e,s.z,r.z,o.z)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Ro=class extends Li{constructor(e=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=e}getPoint(e,t=new $){let i=t,s=this.points,r=(s.length-1)*e,o=Math.floor(r),a=r-o,c=s[o===0?o:o-1],l=s[o],h=s[o>s.length-2?s.length-1:o+1],u=s[o>s.length-3?s.length-1:o+2];return i.set(ff(a,c.x,l.x,h.x,u.x),ff(a,c.y,l.y,h.y,u.y)),i}copy(e){super.copy(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(s.clone())}return this}toJSON(){let e=super.toJSON();e.points=[];for(let t=0,i=this.points.length;t<i;t++){let s=this.points[t];e.points.push(s.toArray())}return e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(new $().fromArray(s))}return this}},ru=Object.freeze({__proto__:null,ArcCurve:Al,CatmullRomCurve3:Rl,CubicBezierCurve:wo,CubicBezierCurve3:Cl,EllipseCurve:wr,LineCurve:To,LineCurve3:Pl,QuadraticBezierCurve:Ao,QuadraticBezierCurve3:Il,SplineCurve:Ro}),Dl=class extends Li{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(e){this.curves.push(e)}closePath(){let e=this.curves[0].getPoint(0),t=this.curves[this.curves.length-1].getPoint(1);if(!e.equals(t)){let i=e.isVector2===!0?"LineCurve":"LineCurve3";this.curves.push(new ru[i](t,e))}return this}getPoint(e,t){let i=e*this.getLength(),s=this.getCurveLengths(),r=0;for(;r<s.length;){if(s[r]>=i){let o=s[r]-i,a=this.curves[r],c=a.getLength(),l=c===0?0:1-o/c;return a.getPointAt(l,t)}r++}return null}getLength(){let e=this.getCurveLengths();return e[e.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;let e=[],t=0;for(let i=0,s=this.curves.length;i<s;i++)t+=this.curves[i].getLength(),e.push(t);return this.cacheLengths=e,e}getSpacedPoints(e=40){let t=[];for(let i=0;i<=e;i++)t.push(this.getPoint(i/e));return this.autoClose&&t.push(t[0]),t}getPoints(e=12){let t=[],i;for(let s=0,r=this.curves;s<r.length;s++){let o=r[s],a=o.isEllipseCurve?e*2:o.isLineCurve||o.isLineCurve3?1:o.isSplineCurve?e*o.points.length:e,c=o.getPoints(a);for(let l=0;l<c.length;l++){let h=c[l];i&&i.equals(h)||(t.push(h),i=h)}}return this.autoClose&&t.length>1&&!t[t.length-1].equals(t[0])&&t.push(t[0]),t}copy(e){super.copy(e),this.curves=[];for(let t=0,i=e.curves.length;t<i;t++){let s=e.curves[t];this.curves.push(s.clone())}return this.autoClose=e.autoClose,this}toJSON(){let e=super.toJSON();e.autoClose=this.autoClose,e.curves=[];for(let t=0,i=this.curves.length;t<i;t++){let s=this.curves[t];e.curves.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.autoClose=e.autoClose,this.curves=[];for(let t=0,i=e.curves.length;t<i;t++){let s=e.curves[t];this.curves.push(new ru[s.type]().fromJSON(s))}return this}},mn=class extends Dl{constructor(e){super(),this.type="Path",this.currentPoint=new $,e&&this.setFromPoints(e)}setFromPoints(e){this.moveTo(e[0].x,e[0].y);for(let t=1,i=e.length;t<i;t++)this.lineTo(e[t].x,e[t].y);return this}moveTo(e,t){return this.currentPoint.set(e,t),this}lineTo(e,t){let i=new To(this.currentPoint.clone(),new $(e,t));return this.curves.push(i),this.currentPoint.set(e,t),this}quadraticCurveTo(e,t,i,s){let r=new Ao(this.currentPoint.clone(),new $(e,t),new $(i,s));return this.curves.push(r),this.currentPoint.set(i,s),this}bezierCurveTo(e,t,i,s,r,o){let a=new wo(this.currentPoint.clone(),new $(e,t),new $(i,s),new $(r,o));return this.curves.push(a),this.currentPoint.set(r,o),this}splineThru(e){let t=[this.currentPoint.clone()].concat(e),i=new Ro(t);return this.curves.push(i),this.currentPoint.copy(e[e.length-1]),this}arc(e,t,i,s,r,o){let a=this.currentPoint.x,c=this.currentPoint.y;return this.absarc(e+a,t+c,i,s,r,o),this}absarc(e,t,i,s,r,o){return this.absellipse(e,t,i,i,s,r,o),this}ellipse(e,t,i,s,r,o,a,c){let l=this.currentPoint.x,h=this.currentPoint.y;return this.absellipse(e+l,t+h,i,s,r,o,a,c),this}absellipse(e,t,i,s,r,o,a,c){let l=new wr(e,t,i,s,r,o,a,c);if(this.curves.length>0){let u=l.getPoint(0);u.equals(this.currentPoint)||this.lineTo(u.x,u.y)}this.curves.push(l);let h=l.getPoint(1);return this.currentPoint.copy(h),this}copy(e){return super.copy(e),this.currentPoint.copy(e.currentPoint),this}toJSON(){let e=super.toJSON();return e.currentPoint=this.currentPoint.toArray(),e}fromJSON(e){return super.fromJSON(e),this.currentPoint.fromArray(e.currentPoint),this}},gn=class extends mn{constructor(e){super(e),this.uuid=dn(),this.type="Shape",this.holes=[]}getPointsHoles(e){let t=[];for(let i=0,s=this.holes.length;i<s;i++)t[i]=this.holes[i].getPoints(e);return t}extractPoints(e){return{shape:this.getPoints(e),holes:this.getPointsHoles(e)}}copy(e){super.copy(e),this.holes=[];for(let t=0,i=e.holes.length;t<i;t++){let s=e.holes[t];this.holes.push(s.clone())}return this}toJSON(){let e=super.toJSON();e.uuid=this.uuid,e.holes=[];for(let t=0,i=this.holes.length;t<i;t++){let s=this.holes[t];e.holes.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.uuid=e.uuid,this.holes=[];for(let t=0,i=e.holes.length;t<i;t++){let s=e.holes[t];this.holes.push(new mn().fromJSON(s))}return this}};function Vg(n,e,t=2){let i=e&&e.length,s=i?e[0]*t:n.length,r=op(n,0,s,t,!0),o=[];if(!r||r.next===r.prev)return o;let a,c,l;if(i&&(r=Yg(n,e,r,t)),n.length>80*t){a=n[0],c=n[1];let h=a,u=c;for(let d=t;d<s;d+=t){let f=n[d],g=n[d+1];f<a&&(a=f),g<c&&(c=g),f>h&&(h=f),g>u&&(u=g)}l=Math.max(h-a,u-c),l=l!==0?32767/l:0}return Co(r,o,t,a,c,l,0),o}function op(n,e,t,i,s){let r;if(s===s0(n,e,t,i)>0)for(let o=e;o<t;o+=i)r=pf(o/i|0,n[o],n[o+1],r);else for(let o=t-i;o>=e;o-=i)r=pf(o/i|0,n[o],n[o+1],r);return r&&Tr(r,r.next)&&(Io(r),r=r.next),r}function Ls(n,e){if(!n)return n;e||(e=n);let t=n,i;do if(i=!1,!t.steiner&&(Tr(t,t.next)||Ct(t.prev,t,t.next)===0)){if(Io(t),t=e=t.prev,t===t.next)break;i=!0}else t=t.next;while(i||t!==e);return e}function Co(n,e,t,i,s,r,o){if(!n)return;!o&&r&&Kg(n,i,s,r);let a=n;for(;n.prev!==n.next;){let c=n.prev,l=n.next;if(r?Wg(n,i,s,r):Gg(n)){e.push(c.i,n.i,l.i),Io(n),n=l.next,a=l.next;continue}if(n=l,n===a){o?o===1?(n=Xg(Ls(n),e),Co(n,e,t,i,s,r,2)):o===2&&qg(n,e,t,i,s,r):Co(Ls(n),e,t,i,s,r,1);break}}}function Gg(n){let e=n.prev,t=n,i=n.next;if(Ct(e,t,i)>=0)return!1;let s=e.x,r=t.x,o=i.x,a=e.y,c=t.y,l=i.y,h=Math.min(s,r,o),u=Math.min(a,c,l),d=Math.max(s,r,o),f=Math.max(a,c,l),g=i.next;for(;g!==e;){if(g.x>=h&&g.x<=d&&g.y>=u&&g.y<=f&&io(s,a,r,c,o,l,g.x,g.y)&&Ct(g.prev,g,g.next)>=0)return!1;g=g.next}return!0}function Wg(n,e,t,i){let s=n.prev,r=n,o=n.next;if(Ct(s,r,o)>=0)return!1;let a=s.x,c=r.x,l=o.x,h=s.y,u=r.y,d=o.y,f=Math.min(a,c,l),g=Math.min(h,u,d),x=Math.max(a,c,l),p=Math.max(h,u,d),m=ou(f,g,e,t,i),M=ou(x,p,e,t,i),b=n.prevZ,y=n.nextZ;for(;b&&b.z>=m&&y&&y.z<=M;){if(b.x>=f&&b.x<=x&&b.y>=g&&b.y<=p&&b!==s&&b!==o&&io(a,h,c,u,l,d,b.x,b.y)&&Ct(b.prev,b,b.next)>=0||(b=b.prevZ,y.x>=f&&y.x<=x&&y.y>=g&&y.y<=p&&y!==s&&y!==o&&io(a,h,c,u,l,d,y.x,y.y)&&Ct(y.prev,y,y.next)>=0))return!1;y=y.nextZ}for(;b&&b.z>=m;){if(b.x>=f&&b.x<=x&&b.y>=g&&b.y<=p&&b!==s&&b!==o&&io(a,h,c,u,l,d,b.x,b.y)&&Ct(b.prev,b,b.next)>=0)return!1;b=b.prevZ}for(;y&&y.z<=M;){if(y.x>=f&&y.x<=x&&y.y>=g&&y.y<=p&&y!==s&&y!==o&&io(a,h,c,u,l,d,y.x,y.y)&&Ct(y.prev,y,y.next)>=0)return!1;y=y.nextZ}return!0}function Xg(n,e){let t=n;do{let i=t.prev,s=t.next.next;!Tr(i,s)&&lp(i,t,t.next,s)&&Po(i,s)&&Po(s,i)&&(e.push(i.i,t.i,s.i),Io(t),Io(t.next),t=n=s),t=t.next}while(t!==n);return Ls(t)}function qg(n,e,t,i,s,r){let o=n;do{let a=o.next.next;for(;a!==o.prev;){if(o.i!==a.i&&t0(o,a)){let c=cp(o,a);o=Ls(o,o.next),c=Ls(c,c.next),Co(o,e,t,i,s,r,0),Co(c,e,t,i,s,r,0);return}a=a.next}o=o.next}while(o!==n)}function Yg(n,e,t,i){let s=[];for(let r=0,o=e.length;r<o;r++){let a=e[r]*i,c=r<o-1?e[r+1]*i:n.length,l=op(n,a,c,i,!1);l===l.next&&(l.steiner=!0),s.push(e0(l))}s.sort($g);for(let r=0;r<s.length;r++)t=Zg(s[r],t);return t}function $g(n,e){let t=n.x-e.x;if(t===0&&(t=n.y-e.y,t===0)){let i=(n.next.y-n.y)/(n.next.x-n.x),s=(e.next.y-e.y)/(e.next.x-e.x);t=i-s}return t}function Zg(n,e){let t=Jg(n,e);if(!t)return e;let i=cp(t,n);return Ls(i,i.next),Ls(t,t.next)}function Jg(n,e){let t=e,i=n.x,s=n.y,r=-1/0,o;if(Tr(n,t))return t;do{if(Tr(n,t.next))return t.next;if(s<=t.y&&s>=t.next.y&&t.next.y!==t.y){let u=t.x+(s-t.y)*(t.next.x-t.x)/(t.next.y-t.y);if(u<=i&&u>r&&(r=u,o=t.x<t.next.x?t:t.next,u===i))return o}t=t.next}while(t!==e);if(!o)return null;let a=o,c=o.x,l=o.y,h=1/0;t=o;do{if(i>=t.x&&t.x>=c&&i!==t.x&&ap(s<l?i:r,s,c,l,s<l?r:i,s,t.x,t.y)){let u=Math.abs(s-t.y)/(i-t.x);Po(t,n)&&(u<h||u===h&&(t.x>o.x||t.x===o.x&&jg(o,t)))&&(o=t,h=u)}t=t.next}while(t!==a);return o}function jg(n,e){return Ct(n.prev,n,e.prev)<0&&Ct(e.next,n,n.next)<0}function Kg(n,e,t,i){let s=n;do s.z===0&&(s.z=ou(s.x,s.y,e,t,i)),s.prevZ=s.prev,s.nextZ=s.next,s=s.next;while(s!==n);s.prevZ.nextZ=null,s.prevZ=null,Qg(s)}function Qg(n){let e,t=1;do{let i=n,s;n=null;let r=null;for(e=0;i;){e++;let o=i,a=0;for(let l=0;l<t&&(a++,o=o.nextZ,!!o);l++);let c=t;for(;a>0||c>0&&o;)a!==0&&(c===0||!o||i.z<=o.z)?(s=i,i=i.nextZ,a--):(s=o,o=o.nextZ,c--),r?r.nextZ=s:n=s,s.prevZ=r,r=s;i=o}r.nextZ=null,t*=2}while(e>1);return n}function ou(n,e,t,i,s){return n=(n-t)*s|0,e=(e-i)*s|0,n=(n|n<<8)&16711935,n=(n|n<<4)&252645135,n=(n|n<<2)&858993459,n=(n|n<<1)&1431655765,e=(e|e<<8)&16711935,e=(e|e<<4)&252645135,e=(e|e<<2)&858993459,e=(e|e<<1)&1431655765,n|e<<1}function e0(n){let e=n,t=n;do(e.x<t.x||e.x===t.x&&e.y<t.y)&&(t=e),e=e.next;while(e!==n);return t}function ap(n,e,t,i,s,r,o,a){return(s-o)*(e-a)>=(n-o)*(r-a)&&(n-o)*(i-a)>=(t-o)*(e-a)&&(t-o)*(r-a)>=(s-o)*(i-a)}function io(n,e,t,i,s,r,o,a){return!(n===o&&e===a)&&ap(n,e,t,i,s,r,o,a)}function t0(n,e){return n.next.i!==e.i&&n.prev.i!==e.i&&!i0(n,e)&&(Po(n,e)&&Po(e,n)&&n0(n,e)&&(Ct(n.prev,n,e.prev)||Ct(n,e.prev,e))||Tr(n,e)&&Ct(n.prev,n,n.next)>0&&Ct(e.prev,e,e.next)>0)}function Ct(n,e,t){return(e.y-n.y)*(t.x-e.x)-(e.x-n.x)*(t.y-e.y)}function Tr(n,e){return n.x===e.x&&n.y===e.y}function lp(n,e,t,i){let s=tl(Ct(n,e,t)),r=tl(Ct(n,e,i)),o=tl(Ct(t,i,n)),a=tl(Ct(t,i,e));return!!(s!==r&&o!==a||s===0&&el(n,t,e)||r===0&&el(n,i,e)||o===0&&el(t,n,i)||a===0&&el(t,e,i))}function el(n,e,t){return e.x<=Math.max(n.x,t.x)&&e.x>=Math.min(n.x,t.x)&&e.y<=Math.max(n.y,t.y)&&e.y>=Math.min(n.y,t.y)}function tl(n){return n>0?1:n<0?-1:0}function i0(n,e){let t=n;do{if(t.i!==n.i&&t.next.i!==n.i&&t.i!==e.i&&t.next.i!==e.i&&lp(t,t.next,n,e))return!0;t=t.next}while(t!==n);return!1}function Po(n,e){return Ct(n.prev,n,n.next)<0?Ct(n,e,n.next)>=0&&Ct(n,n.prev,e)>=0:Ct(n,e,n.prev)<0||Ct(n,n.next,e)<0}function n0(n,e){let t=n,i=!1,s=(n.x+e.x)/2,r=(n.y+e.y)/2;do t.y>r!=t.next.y>r&&t.next.y!==t.y&&s<(t.next.x-t.x)*(r-t.y)/(t.next.y-t.y)+t.x&&(i=!i),t=t.next;while(t!==n);return i}function cp(n,e){let t=au(n.i,n.x,n.y),i=au(e.i,e.x,e.y),s=n.next,r=e.prev;return n.next=e,e.prev=n,t.next=s,s.prev=t,i.next=t,t.prev=i,r.next=i,i.prev=r,i}function pf(n,e,t,i){let s=au(n,e,t);return i?(s.next=i.next,s.prev=i,i.next.prev=s,i.next=s):(s.prev=s,s.next=s),s}function Io(n){n.next.prev=n.prev,n.prev.next=n.next,n.prevZ&&(n.prevZ.nextZ=n.nextZ),n.nextZ&&(n.nextZ.prevZ=n.prevZ)}function au(n,e,t){return{i:n,x:e,y:t,prev:null,next:null,z:0,prevZ:null,nextZ:null,steiner:!1}}function s0(n,e,t,i){let s=0;for(let r=e,o=t-i;r<t;r+=i)s+=(n[o]-n[r])*(n[r+1]+n[o+1]),o=r;return s}var lu=class{static triangulate(e,t,i=2){return Vg(e,t,i)}},Rs=class n{static area(e){let t=e.length,i=0;for(let s=t-1,r=0;r<t;s=r++)i+=e[s].x*e[r].y-e[r].x*e[s].y;return i*.5}static isClockWise(e){return n.area(e)<0}static triangulateShape(e,t){let i=[],s=[],r=[];mf(e),gf(i,e);let o=e.length;t.forEach(mf);for(let c=0;c<t.length;c++)s.push(o),o+=t[c].length,gf(i,t[c]);let a=lu.triangulate(i,s);for(let c=0;c<a.length;c+=3)r.push(a.slice(c,c+3));return r}};function mf(n){let e=n.length;e>2&&n[e-1].equals(n[0])&&n.pop()}function gf(n,e){for(let t=0;t<e.length;t++)n.push(e[t].x),n.push(e[t].y)}var Fn=class n extends ut{constructor(e=new gn([new $(.5,.5),new $(-.5,.5),new $(-.5,-.5),new $(.5,-.5)]),t={}){super(),this.type="ExtrudeGeometry",this.parameters={shapes:e,options:t},e=Array.isArray(e)?e:[e];let i=this,s=[],r=[];for(let a=0,c=e.length;a<c;a++){let l=e[a];o(l)}this.setAttribute("position",new it(s,3)),this.setAttribute("uv",new it(r,2)),this.computeVertexNormals();function o(a){let c=[],l=t.curveSegments!==void 0?t.curveSegments:12,h=t.steps!==void 0?t.steps:1,u=t.depth!==void 0?t.depth:1,d=t.bevelEnabled!==void 0?t.bevelEnabled:!0,f=t.bevelThickness!==void 0?t.bevelThickness:.2,g=t.bevelSize!==void 0?t.bevelSize:f-.1,x=t.bevelOffset!==void 0?t.bevelOffset:0,p=t.bevelSegments!==void 0?t.bevelSegments:3,m=t.extrudePath,M=t.UVGenerator!==void 0?t.UVGenerator:r0,b,y=!1,T,S,A,_;if(m){b=m.getSpacedPoints(h),y=!0,d=!1;let O=m.isCatmullRomCurve3?m.closed:!1;T=m.computeFrenetFrames(h,O),S=new P,A=new P,_=new P}d||(p=0,f=0,g=0,x=0);let E=a.extractPoints(l),C=E.shape,I=E.holes;if(!Rs.isClockWise(C)){C=C.reverse();for(let O=0,H=I.length;O<H;O++){let Q=I[O];Rs.isClockWise(Q)&&(I[O]=Q.reverse())}}function V(O){let Q=10000000000000001e-36,W=O[0];for(let G=1;G<=O.length;G++){let se=G%O.length,ce=O[se],fe=ce.x-W.x,me=ce.y-W.y,D=fe*fe+me*me,Me=Math.max(Math.abs(ce.x),Math.abs(ce.y),Math.abs(W.x),Math.abs(W.y)),Ve=Q*Me*Me;if(D<=Ve){O.splice(se,1),G--;continue}W=ce}}V(C),I.forEach(V);let q=I.length,N=C;for(let O=0;O<q;O++){let H=I[O];C=C.concat(H)}function Y(O,H,Q){return H||Ze("ExtrudeGeometry: vec does not exist"),O.clone().addScaledVector(H,Q)}let X=C.length;function ne(O,H,Q){let W,G,se,ce=O.x-H.x,fe=O.y-H.y,me=Q.x-O.x,D=Q.y-O.y,Me=ce*ce+fe*fe,Ve=ce*D-fe*me;if(Math.abs(Ve)>Number.EPSILON){let R=Math.sqrt(Me),v=Math.sqrt(me*me+D*D),U=H.x-fe/R,B=H.y+ce/R,k=Q.x-D/v,pe=Q.y+me/v,_e=((k-U)*D-(pe-B)*me)/(ce*D-fe*me);W=U+ce*_e-O.x,G=B+fe*_e-O.y;let te=W*W+G*G;if(te<=2)return new $(W,G);se=Math.sqrt(te/2)}else{let R=!1;ce>Number.EPSILON?me>Number.EPSILON&&(R=!0):ce<-Number.EPSILON?me<-Number.EPSILON&&(R=!0):Math.sign(fe)===Math.sign(D)&&(R=!0),R?(W=-fe,G=ce,se=Math.sqrt(Me)):(W=ce,G=fe,se=Math.sqrt(Me/2))}return new $(W/se,G/se)}let ie=[];for(let O=0,H=N.length,Q=H-1,W=O+1;O<H;O++,Q++,W++)Q===H&&(Q=0),W===H&&(W=0),ie[O]=ne(N[O],N[Q],N[W]);let ge=[],ue,xe=ie.concat();for(let O=0,H=q;O<H;O++){let Q=I[O];ue=[];for(let W=0,G=Q.length,se=G-1,ce=W+1;W<G;W++,se++,ce++)se===G&&(se=0),ce===G&&(ce=0),ue[W]=ne(Q[W],Q[se],Q[ce]);ge.push(ue),xe=xe.concat(ue)}let Ne;if(p===0)Ne=Rs.triangulateShape(N,I);else{let O=[],H=[];for(let Q=0;Q<p;Q++){let W=Q/p,G=f*Math.cos(W*Math.PI/2),se=g*Math.sin(W*Math.PI/2)+x;for(let ce=0,fe=N.length;ce<fe;ce++){let me=Y(N[ce],ie[ce],se);Ae(me.x,me.y,-G),W===0&&O.push(me)}for(let ce=0,fe=q;ce<fe;ce++){let me=I[ce];ue=ge[ce];let D=[];for(let Me=0,Ve=me.length;Me<Ve;Me++){let R=Y(me[Me],ue[Me],se);Ae(R.x,R.y,-G),W===0&&D.push(R)}W===0&&H.push(D)}}Ne=Rs.triangulateShape(O,H)}let st=Ne.length,Xe=g+x;for(let O=0;O<X;O++){let H=d?Y(C[O],xe[O],Xe):C[O];y?(A.copy(T.normals[0]).multiplyScalar(H.x),S.copy(T.binormals[0]).multiplyScalar(H.y),_.copy(b[0]).add(A).add(S),Ae(_.x,_.y,_.z)):Ae(H.x,H.y,0)}for(let O=1;O<=h;O++)for(let H=0;H<X;H++){let Q=d?Y(C[H],xe[H],Xe):C[H];y?(A.copy(T.normals[O]).multiplyScalar(Q.x),S.copy(T.binormals[O]).multiplyScalar(Q.y),_.copy(b[O]).add(A).add(S),Ae(_.x,_.y,_.z)):Ae(Q.x,Q.y,u/h*O)}for(let O=p-1;O>=0;O--){let H=O/p,Q=f*Math.cos(H*Math.PI/2),W=g*Math.sin(H*Math.PI/2)+x;for(let G=0,se=N.length;G<se;G++){let ce=Y(N[G],ie[G],W);Ae(ce.x,ce.y,u+Q)}for(let G=0,se=I.length;G<se;G++){let ce=I[G];ue=ge[G];for(let fe=0,me=ce.length;fe<me;fe++){let D=Y(ce[fe],ue[fe],W);y?Ae(D.x,D.y+b[h-1].y,b[h-1].x+Q):Ae(D.x,D.y,u+Q)}}}j(),he();function j(){let O=s.length/3;if(d){let H=0,Q=X*H;for(let W=0;W<st;W++){let G=Ne[W];Fe(G[2]+Q,G[1]+Q,G[0]+Q)}H=h+p*2,Q=X*H;for(let W=0;W<st;W++){let G=Ne[W];Fe(G[0]+Q,G[1]+Q,G[2]+Q)}}else{for(let H=0;H<st;H++){let Q=Ne[H];Fe(Q[2],Q[1],Q[0])}for(let H=0;H<st;H++){let Q=Ne[H];Fe(Q[0]+X*h,Q[1]+X*h,Q[2]+X*h)}}i.addGroup(O,s.length/3-O,0)}function he(){let O=s.length/3,H=0;le(N,H),H+=N.length;for(let Q=0,W=I.length;Q<W;Q++){let G=I[Q];le(G,H),H+=G.length}i.addGroup(O,s.length/3-O,1)}function le(O,H){let Q=O.length;for(;--Q>=0;){let W=Q,G=Q-1;G<0&&(G=O.length-1);for(let se=0,ce=h+p*2;se<ce;se++){let fe=X*se,me=X*(se+1),D=H+W+fe,Me=H+G+fe,Ve=H+G+me,R=H+W+me;ke(D,Me,Ve,R)}}}function Ae(O,H,Q){c.push(O),c.push(H),c.push(Q)}function Fe(O,H,Q){ae(O),ae(H),ae(Q);let W=s.length/3,G=M.generateTopUV(i,s,W-3,W-2,W-1);ee(G[0]),ee(G[1]),ee(G[2])}function ke(O,H,Q,W){ae(O),ae(H),ae(W),ae(H),ae(Q),ae(W);let G=s.length/3,se=M.generateSideWallUV(i,s,G-6,G-3,G-2,G-1);ee(se[0]),ee(se[1]),ee(se[3]),ee(se[1]),ee(se[2]),ee(se[3])}function ae(O){s.push(c[O*3+0]),s.push(c[O*3+1]),s.push(c[O*3+2])}function ee(O){r.push(O.x),r.push(O.y)}}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}toJSON(){let e=super.toJSON(),t=this.parameters.shapes,i=this.parameters.options;return o0(t,i,e)}static fromJSON(e,t){let i=[];for(let r=0,o=e.shapes.length;r<o;r++){let a=t[e.shapes[r]];i.push(a)}let s=e.options.extrudePath;return s!==void 0&&(e.options.extrudePath=new ru[s.type]().fromJSON(s)),new n(i,e.options)}},r0={generateTopUV:function(n,e,t,i,s){let r=e[t*3],o=e[t*3+1],a=e[i*3],c=e[i*3+1],l=e[s*3],h=e[s*3+1];return[new $(r,o),new $(a,c),new $(l,h)]},generateSideWallUV:function(n,e,t,i,s,r){let o=e[t*3],a=e[t*3+1],c=e[t*3+2],l=e[i*3],h=e[i*3+1],u=e[i*3+2],d=e[s*3],f=e[s*3+1],g=e[s*3+2],x=e[r*3],p=e[r*3+1],m=e[r*3+2];return Math.abs(a-h)<Math.abs(o-l)?[new $(o,1-c),new $(l,1-u),new $(d,1-g),new $(x,1-m)]:[new $(a,1-c),new $(h,1-u),new $(f,1-g),new $(p,1-m)]}};function o0(n,e,t){if(t.shapes=[],Array.isArray(n))for(let i=0,s=n.length;i<s;i++){let r=n[i];t.shapes.push(r.uuid)}else t.shapes.push(n.uuid);return t.options=Object.assign({},e),e.extrudePath!==void 0&&(t.options.extrudePath=e.extrudePath.toJSON()),t}var Qi=class n extends So{constructor(e=1,t=0){let i=(1+Math.sqrt(5))/2,s=[-1,i,0,1,i,0,-1,-i,0,1,-i,0,0,-1,i,0,1,i,0,-1,-i,0,1,-i,i,0,-1,i,0,1,-i,0,-1,-i,0,1],r=[0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1];super(s,r,e,t),this.type="IcosahedronGeometry",this.parameters={radius:e,detail:t}}static fromJSON(e){return new n(e.radius,e.detail)}};var ki=class n extends ut{constructor(e=1,t=1,i=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:i,heightSegments:s};let r=e/2,o=t/2,a=Math.floor(i),c=Math.floor(s),l=a+1,h=c+1,u=e/a,d=t/c,f=[],g=[],x=[],p=[];for(let m=0;m<h;m++){let M=m*d-o;for(let b=0;b<l;b++){let y=b*u-r;g.push(y,-M,0),x.push(0,0,1),p.push(b/a),p.push(1-m/c)}}for(let m=0;m<c;m++)for(let M=0;M<a;M++){let b=M+l*m,y=M+l*(m+1),T=M+1+l*(m+1),S=M+1+l*m;f.push(b,y,S),f.push(y,T,S)}this.setIndex(f),this.setAttribute("position",new it(g,3)),this.setAttribute("normal",new it(x,3)),this.setAttribute("uv",new it(p,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.width,e.height,e.widthSegments,e.heightSegments)}},Do=class n extends ut{constructor(e=.5,t=1,i=32,s=1,r=0,o=Math.PI*2){super(),this.type="RingGeometry",this.parameters={innerRadius:e,outerRadius:t,thetaSegments:i,phiSegments:s,thetaStart:r,thetaLength:o},i=Math.max(3,i),s=Math.max(1,s);let a=[],c=[],l=[],h=[],u=e,d=(t-e)/s,f=new P,g=new $;for(let x=0;x<=s;x++){for(let p=0;p<=i;p++){let m=r+p/i*o;f.x=u*Math.cos(m),f.y=u*Math.sin(m),c.push(f.x,f.y,f.z),l.push(0,0,1),g.x=(f.x/t+1)/2,g.y=(f.y/t+1)/2,h.push(g.x,g.y)}u+=d}for(let x=0;x<s;x++){let p=x*(i+1);for(let m=0;m<i;m++){let M=m+p,b=M,y=M+i+1,T=M+i+2,S=M+1;a.push(b,y,S),a.push(y,T,S)}}this.setIndex(a),this.setAttribute("position",new it(c,3)),this.setAttribute("normal",new it(l,3)),this.setAttribute("uv",new it(h,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.innerRadius,e.outerRadius,e.thetaSegments,e.phiSegments,e.thetaStart,e.thetaLength)}};var Lo=class extends ut{constructor(e=null){if(super(),this.type="WireframeGeometry",this.parameters={geometry:e},e!==null){let t=[],i=new Set,s=new P,r=new P;if(e.index!==null){let o=e.attributes.position,a=e.index,c=e.groups;c.length===0&&(c=[{start:0,count:a.count,materialIndex:0}]);for(let l=0,h=c.length;l<h;++l){let u=c[l],d=u.start,f=u.count;for(let g=d,x=d+f;g<x;g+=3)for(let p=0;p<3;p++){let m=a.getX(g+p),M=a.getX(g+(p+1)%3);s.fromBufferAttribute(o,m),r.fromBufferAttribute(o,M),_f(s,r,i)===!0&&(t.push(s.x,s.y,s.z),t.push(r.x,r.y,r.z))}}}else{let o=e.attributes.position;for(let a=0,c=o.count/3;a<c;a++)for(let l=0;l<3;l++){let h=3*a+l,u=3*a+(l+1)%3;s.fromBufferAttribute(o,h),r.fromBufferAttribute(o,u),_f(s,r,i)===!0&&(t.push(s.x,s.y,s.z),t.push(r.x,r.y,r.z))}}this.setAttribute("position",new it(t,3))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}};function _f(n,e,t){let i=`${n.x},${n.y},${n.z}-${e.x},${e.y},${e.z}`,s=`${e.x},${e.y},${e.z}-${n.x},${n.y},${n.z}`;return t.has(i)===!0||t.has(s)===!0?!1:(t.add(i),t.add(s),!0)}var Uo=class extends yi{constructor(e){super(),this.isShadowMaterial=!0,this.type="ShadowMaterial",this.color=new Te(0),this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.fog=e.fog,this}};function Bs(n){let e={};for(let t in n){e[t]={};for(let i in n[t]){let s=n[t][i];if(xf(s))s.isRenderTargetTexture?($e("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][i]=null):e[t][i]=s.clone();else if(Array.isArray(s))if(xf(s[0])){let r=[];for(let o=0,a=s.length;o<a;o++)r[o]=s[o].clone();e[t][i]=r}else e[t][i]=s.slice();else e[t][i]=s}}return e}function hi(n){let e={};for(let t=0;t<n.length;t++){let i=Bs(n[t]);for(let s in i)e[s]=i[s]}return e}function xf(n){return n&&(n.isColor||n.isMatrix3||n.isMatrix4||n.isVector2||n.isVector3||n.isVector4||n.isTexture||n.isQuaternion)}function a0(n){let e=[];for(let t=0;t<n.length;t++)e.push(n[t].clone());return e}function Ru(n){let e=n.getRenderTarget();return e===null?n.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:ht.workingColorSpace}var mi={clone:Bs,merge:hi},l0=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,c0=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,bt=class extends yi{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=l0,this.fragmentShader=c0,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Bs(e.uniforms),this.uniformsGroups=a0(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){let t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(let s in this.uniforms){let o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;let i={};for(let s in this.extensions)this.extensions[s]===!0&&(i[s]=!0);return Object.keys(i).length>0&&(t.extensions=i),t}fromJSON(e,t){if(super.fromJSON(e,t),e.uniforms!==void 0)for(let i in e.uniforms){let s=e.uniforms[i];switch(this.uniforms[i]={},s.type){case"t":this.uniforms[i].value=t[s.value]||null;break;case"c":this.uniforms[i].value=new Te().setHex(s.value);break;case"v2":this.uniforms[i].value=new $().fromArray(s.value);break;case"v3":this.uniforms[i].value=new P().fromArray(s.value);break;case"v4":this.uniforms[i].value=new mt().fromArray(s.value);break;case"m3":this.uniforms[i].value=new tt().fromArray(s.value);break;case"m4":this.uniforms[i].value=new rt().fromArray(s.value);break;default:this.uniforms[i].value=s.value}}if(e.defines!==void 0&&(this.defines=e.defines),e.vertexShader!==void 0&&(this.vertexShader=e.vertexShader),e.fragmentShader!==void 0&&(this.fragmentShader=e.fragmentShader),e.glslVersion!==void 0&&(this.glslVersion=e.glslVersion),e.extensions!==void 0)for(let i in e.extensions)this.extensions[i]=e.extensions[i];return e.lights!==void 0&&(this.lights=e.lights),e.clipping!==void 0&&(this.clipping=e.clipping),this}},Ar=class extends bt{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}},Qe=class extends yi{constructor(e){super(),this.isMeshStandardMaterial=!0,this.type="MeshStandardMaterial",this.defines={STANDARD:""},this.color=new Te(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Te(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Lr,this.normalScale=new $(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ii,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.defines={STANDARD:""},this.color.copy(e.color),this.roughness=e.roughness,this.metalness=e.metalness,this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.roughnessMap=e.roughnessMap,this.metalnessMap=e.metalnessMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.envMapIntensity=e.envMapIntensity,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}},No=class extends Qe{constructor(e){super(),this.isMeshPhysicalMaterial=!0,this.defines={STANDARD:"",PHYSICAL:""},this.type="MeshPhysicalMaterial",this.anisotropyRotation=0,this.anisotropyMap=null,this.clearcoatMap=null,this.clearcoatRoughness=0,this.clearcoatRoughnessMap=null,this.clearcoatNormalScale=new $(1,1),this.clearcoatNormalMap=null,this.ior=1.5,Object.defineProperty(this,"reflectivity",{get:function(){return je(2.5*(this.ior-1)/(this.ior+1),0,1)},set:function(t){this.ior=(1+.4*t)/(1-.4*t)}}),this.iridescenceMap=null,this.iridescenceIOR=1.3,this.iridescenceThicknessRange=[100,400],this.iridescenceThicknessMap=null,this.sheenColor=new Te(0),this.sheenColorMap=null,this.sheenRoughness=1,this.sheenRoughnessMap=null,this.transmissionMap=null,this.thickness=0,this.thicknessMap=null,this.attenuationDistance=1/0,this.attenuationColor=new Te(1,1,1),this.specularIntensity=1,this.specularIntensityMap=null,this.specularColor=new Te(1,1,1),this.specularColorMap=null,this._anisotropy=0,this._clearcoat=0,this._dispersion=0,this._iridescence=0,this._sheen=0,this._transmission=0,this.setValues(e)}get anisotropy(){return this._anisotropy}set anisotropy(e){this._anisotropy>0!=e>0&&this.version++,this._anisotropy=e}get clearcoat(){return this._clearcoat}set clearcoat(e){this._clearcoat>0!=e>0&&this.version++,this._clearcoat=e}get iridescence(){return this._iridescence}set iridescence(e){this._iridescence>0!=e>0&&this.version++,this._iridescence=e}get dispersion(){return this._dispersion}set dispersion(e){this._dispersion>0!=e>0&&this.version++,this._dispersion=e}get sheen(){return this._sheen}set sheen(e){this._sheen>0!=e>0&&this.version++,this._sheen=e}get transmission(){return this._transmission}set transmission(e){this._transmission>0!=e>0&&this.version++,this._transmission=e}copy(e){return super.copy(e),this.defines={STANDARD:"",PHYSICAL:""},this.anisotropy=e.anisotropy,this.anisotropyRotation=e.anisotropyRotation,this.anisotropyMap=e.anisotropyMap,this.clearcoat=e.clearcoat,this.clearcoatMap=e.clearcoatMap,this.clearcoatRoughness=e.clearcoatRoughness,this.clearcoatRoughnessMap=e.clearcoatRoughnessMap,this.clearcoatNormalMap=e.clearcoatNormalMap,this.clearcoatNormalScale.copy(e.clearcoatNormalScale),this.dispersion=e.dispersion,this.ior=e.ior,this.iridescence=e.iridescence,this.iridescenceMap=e.iridescenceMap,this.iridescenceIOR=e.iridescenceIOR,this.iridescenceThicknessRange=[...e.iridescenceThicknessRange],this.iridescenceThicknessMap=e.iridescenceThicknessMap,this.sheen=e.sheen,this.sheenColor.copy(e.sheenColor),this.sheenColorMap=e.sheenColorMap,this.sheenRoughness=e.sheenRoughness,this.sheenRoughnessMap=e.sheenRoughnessMap,this.transmission=e.transmission,this.transmissionMap=e.transmissionMap,this.thickness=e.thickness,this.thicknessMap=e.thicknessMap,this.attenuationDistance=e.attenuationDistance,this.attenuationColor.copy(e.attenuationColor),this.specularIntensity=e.specularIntensity,this.specularIntensityMap=e.specularIntensityMap,this.specularColor.copy(e.specularColor),this.specularColorMap=e.specularColorMap,this}};var Fo=class extends yi{constructor(e){super(),this.isMeshNormalMaterial=!0,this.type="MeshNormalMaterial",this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Lr,this.normalScale=new $(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.flatShading=!1,this.setValues(e)}copy(e){return super.copy(e),this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.flatShading=e.flatShading,this}},Oo=class extends yi{constructor(e){super(),this.isMeshLambertMaterial=!0,this.type="MeshLambertMaterial",this.color=new Te(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Te(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Lr,this.normalScale=new $(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ii,this.combine=Jl,this.reflectivity=1,this.envMapIntensity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.envMapIntensity=e.envMapIntensity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}},Ll=class extends yi{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=qf,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}},Ul=class extends yi{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}};function il(n,e){return!n||n.constructor===e?n:typeof e.BYTES_PER_ELEMENT=="number"?new e(n):Array.prototype.slice.call(n)}var ss=class{constructor(e,t,i,s){this.parameterPositions=e,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new t.constructor(i),this.sampleValues=t,this.valueSize=i,this.settings=null,this.DefaultSettings_={}}evaluate(e){let t=this.parameterPositions,i=this._cachedIndex,s=t[i],r=t[i-1];i:{e:{let o;t:{n:if(!(e<s)){for(let a=i+2;;){if(s===void 0){if(e<r)break n;return i=t.length,this._cachedIndex=i,this.copySampleValue_(i-1)}if(i===a)break;if(r=s,s=t[++i],e<s)break e}o=t.length;break t}if(!(e>=r)){let a=t[1];e<a&&(i=2,r=a);for(let c=i-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(i===c)break;if(s=r,r=t[--i-1],e>=r)break e}o=i,i=0;break t}break i}for(;i<o;){let a=i+o>>>1;e<t[a]?o=a:i=a+1}if(s=t[i],r=t[i-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return i=t.length,this._cachedIndex=i,this.copySampleValue_(i-1)}this._cachedIndex=i,this.intervalChanged_(i,r,s)}return this.interpolate_(i,r,e,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(e){let t=this.resultBuffer,i=this.sampleValues,s=this.valueSize,r=e*s;for(let o=0;o!==s;++o)t[o]=i[r+o];return t}interpolate_(){throw new Error("THREE.Interpolant: Call to abstract method.")}intervalChanged_(){}},Nl=class extends ss{constructor(e,t,i,s){super(e,t,i,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:eu,endingEnd:eu}}intervalChanged_(e,t,i){let s=this.parameterPositions,r=e-2,o=e+1,a=s[r],c=s[o];if(a===void 0)switch(this.getSettings_().endingStart){case tu:r=e,a=2*t-i;break;case iu:r=s.length-2,a=t+s[r]-s[r+1];break;default:r=e,a=i}if(c===void 0)switch(this.getSettings_().endingEnd){case tu:o=e,c=2*i-t;break;case iu:o=1,c=i+s[1]-s[0];break;default:o=e-1,c=t}let l=(i-t)*.5,h=this.valueSize;this._weightPrev=l/(t-a),this._weightNext=l/(c-i),this._offsetPrev=r*h,this._offsetNext=o*h}interpolate_(e,t,i,s){let r=this.resultBuffer,o=this.sampleValues,a=this.valueSize,c=e*a,l=c-a,h=this._offsetPrev,u=this._offsetNext,d=this._weightPrev,f=this._weightNext,g=(i-t)/(s-t),x=g*g,p=x*g,m=-d*p+2*d*x-d*g,M=(1+d)*p+(-1.5-2*d)*x+(-.5+d)*g+1,b=(-1-f)*p+(1.5+f)*x+.5*g,y=f*p-f*x;for(let T=0;T!==a;++T)r[T]=m*o[h+T]+M*o[l+T]+b*o[c+T]+y*o[u+T];return r}},Fl=class extends ss{constructor(e,t,i,s){super(e,t,i,s)}interpolate_(e,t,i,s){let r=this.resultBuffer,o=this.sampleValues,a=this.valueSize,c=e*a,l=c-a,h=(i-t)/(s-t),u=1-h;for(let d=0;d!==a;++d)r[d]=o[l+d]*u+o[c+d]*h;return r}},Ol=class extends ss{constructor(e,t,i,s){super(e,t,i,s)}interpolate_(e){return this.copySampleValue_(e-1)}},Bl=class extends ss{interpolate_(e,t,i,s){let r=this.resultBuffer,o=this.sampleValues,a=this.valueSize,c=e*a,l=c-a,h=this.inTangents,u=this.outTangents;if(!h||!u){let g=(i-t)/(s-t),x=1-g;for(let p=0;p!==a;++p)r[p]=o[l+p]*x+o[c+p]*g;return r}let d=a*2,f=e-1;for(let g=0;g!==a;++g){let x=o[l+g],p=o[c+g],m=f*d+g*2,M=u[m],b=u[m+1],y=e*d+g*2,T=h[y],S=h[y+1],A=(i-t)/(s-t),_,E,C,I,L;for(let V=0;V<8;V++){_=A*A,E=_*A,C=1-A,I=C*C,L=I*C;let N=L*t+3*I*A*M+3*C*_*T+E*s-i;if(Math.abs(N)<1e-10)break;let Y=3*I*(M-t)+6*C*A*(T-M)+3*_*(s-T);if(Math.abs(Y)<1e-10)break;A=A-N/Y,A=Math.max(0,Math.min(1,A))}r[g]=L*x+3*I*A*b+3*C*_*S+E*p}return r}},Ui=class{constructor(e,t,i,s){if(e===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(t===void 0||t.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+e);this.name=e,this.times=il(t,this.TimeBufferType),this.values=il(i,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(e){let t=e.constructor,i;if(t.toJSON!==this.toJSON)i=t.toJSON(e);else{i={name:e.name,times:il(e.times,Array),values:il(e.values,Array)};let s=e.getInterpolation();s!==e.DefaultInterpolation&&(i.interpolation=s)}return i.type=e.ValueTypeName,i}InterpolantFactoryMethodDiscrete(e){return new Ol(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodLinear(e){return new Fl(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodSmooth(e){return new Nl(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodBezier(e){let t=new Bl(this.times,this.values,this.getValueSize(),e);return this.settings&&(t.inTangents=this.settings.inTangents,t.outTangents=this.settings.outTangents),t}setInterpolation(e){let t;switch(e){case oo:t=this.InterpolantFactoryMethodDiscrete;break;case _l:t=this.InterpolantFactoryMethodLinear;break;case ol:t=this.InterpolantFactoryMethodSmooth;break;case Qh:t=this.InterpolantFactoryMethodBezier;break}if(t===void 0){let i="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(e!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(i);return $e("KeyframeTrack:",i),this}return this.createInterpolant=t,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return oo;case this.InterpolantFactoryMethodLinear:return _l;case this.InterpolantFactoryMethodSmooth:return ol;case this.InterpolantFactoryMethodBezier:return Qh}}getValueSize(){return this.values.length/this.times.length}shift(e){if(e!==0){let t=this.times;for(let i=0,s=t.length;i!==s;++i)t[i]+=e}return this}scale(e){if(e!==1){let t=this.times;for(let i=0,s=t.length;i!==s;++i)t[i]*=e}return this}trim(e,t){let i=this.times,s=i.length,r=0,o=s-1;for(;r!==s&&i[r]<e;)++r;for(;o!==-1&&i[o]>t;)--o;if(++o,r!==0||o!==s){r>=o&&(o=Math.max(o,1),r=o-1);let a=this.getValueSize();this.times=i.slice(r,o),this.values=this.values.slice(r*a,o*a)}return this}validate(){let e=!0,t=this.getValueSize();t-Math.floor(t)!==0&&(Ze("KeyframeTrack: Invalid value size in track.",this),e=!1);let i=this.times,s=this.values,r=i.length;r===0&&(Ze("KeyframeTrack: Track is empty.",this),e=!1);let o=null;for(let a=0;a!==r;a++){let c=i[a];if(typeof c=="number"&&isNaN(c)){Ze("KeyframeTrack: Time is not a valid number.",this,a,c),e=!1;break}if(o!==null&&o>c){Ze("KeyframeTrack: Out of order keys.",this,a,c,o),e=!1;break}o=c}if(s!==void 0&&Qm(s))for(let a=0,c=s.length;a!==c;++a){let l=s[a];if(isNaN(l)){Ze("KeyframeTrack: Value is not a valid number.",this,a,l),e=!1;break}}return e}optimize(){let e=this.times.slice(),t=this.values.slice(),i=this.getValueSize(),s=this.getInterpolation()===ol,r=e.length-1,o=1;for(let a=1;a<r;++a){let c=!1,l=e[a],h=e[a+1];if(l!==h&&(a!==1||l!==e[0]))if(s)c=!0;else{let u=a*i,d=u-i,f=u+i;for(let g=0;g!==i;++g){let x=t[u+g];if(x!==t[d+g]||x!==t[f+g]){c=!0;break}}}if(c){if(a!==o){e[o]=e[a];let u=a*i,d=o*i;for(let f=0;f!==i;++f)t[d+f]=t[u+f]}++o}}if(r>0){e[o]=e[r];for(let a=r*i,c=o*i,l=0;l!==i;++l)t[c+l]=t[a+l];++o}return o!==e.length?(this.times=e.slice(0,o),this.values=t.slice(0,o*i)):(this.times=e,this.values=t),this}clone(){let e=this.times.slice(),t=this.values.slice(),i=this.constructor,s=new i(this.name,e,t);return s.createInterpolant=this.createInterpolant,s}};Ui.prototype.ValueTypeName="";Ui.prototype.TimeBufferType=Float32Array;Ui.prototype.ValueBufferType=Float32Array;Ui.prototype.DefaultInterpolation=_l;var rs=class extends Ui{constructor(e,t,i){super(e,t,i)}};rs.prototype.ValueTypeName="bool";rs.prototype.ValueBufferType=Array;rs.prototype.DefaultInterpolation=oo;rs.prototype.InterpolantFactoryMethodLinear=void 0;rs.prototype.InterpolantFactoryMethodSmooth=void 0;var zl=class extends Ui{constructor(e,t,i,s){super(e,t,i,s)}};zl.prototype.ValueTypeName="color";var kl=class extends Ui{constructor(e,t,i,s){super(e,t,i,s)}};kl.prototype.ValueTypeName="number";var Hl=class extends ss{constructor(e,t,i,s){super(e,t,i,s)}interpolate_(e,t,i,s){let r=this.resultBuffer,o=this.sampleValues,a=this.valueSize,c=(i-t)/(s-t),l=e*a;for(let h=l+a;l!==h;l+=4)Pi.slerpFlat(r,0,o,l-a,o,l,c);return r}},Bo=class extends Ui{constructor(e,t,i,s){super(e,t,i,s)}InterpolantFactoryMethodLinear(e){return new Hl(this.times,this.values,this.getValueSize(),e)}};Bo.prototype.ValueTypeName="quaternion";Bo.prototype.InterpolantFactoryMethodSmooth=void 0;var os=class extends Ui{constructor(e,t,i){super(e,t,i)}};os.prototype.ValueTypeName="string";os.prototype.ValueBufferType=Array;os.prototype.DefaultInterpolation=oo;os.prototype.InterpolantFactoryMethodLinear=void 0;os.prototype.InterpolantFactoryMethodSmooth=void 0;var Vl=class extends Ui{constructor(e,t,i,s){super(e,t,i,s)}};Vl.prototype.ValueTypeName="vector";var Gl=class{constructor(e,t,i){let s=this,r=!1,o=0,a=0,c,l=[];this.onStart=void 0,this.onLoad=e,this.onProgress=t,this.onError=i,this._abortController=null,this.itemStart=function(h){a++,r===!1&&s.onStart!==void 0&&s.onStart(h,o,a),r=!0},this.itemEnd=function(h){o++,s.onProgress!==void 0&&s.onProgress(h,o,a),o===a&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return h=h.normalize("NFC"),c?c(h):h},this.setURLModifier=function(h){return c=h,this},this.addHandler=function(h,u){return l.push(h,u),this},this.removeHandler=function(h){let u=l.indexOf(h);return u!==-1&&l.splice(u,2),this},this.getHandler=function(h){for(let u=0,d=l.length;u<d;u+=2){let f=l[u],g=l[u+1];if(f.global&&(f.lastIndex=0),f.test(h))return g}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}},hp=new Gl,Wl=class{constructor(e){this.manager=e!==void 0?e:hp,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(e,t){let i=this;return new Promise(function(s,r){i.load(e,s,t,r)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}};Wl.DEFAULT_MATERIAL_NAME="__DEFAULT";var Rr=class extends ft{constructor(e,t=1){super(),this.isLight=!0,this.type="Light",this.color=new Te(e),this.intensity=t}dispose(){this.dispatchEvent({type:"dispose"})}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){let t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,t}},zo=class extends Rr{constructor(e,t,i){super(e,i),this.isHemisphereLight=!0,this.type="HemisphereLight",this.position.copy(ft.DEFAULT_UP),this.updateMatrix(),this.groundColor=new Te(t)}copy(e,t){return super.copy(e,t),this.groundColor.copy(e.groundColor),this}toJSON(e){let t=super.toJSON(e);return t.object.groundColor=this.groundColor.getHex(),t}},jh=new rt,vf=new P,yf=new P,Xl=class{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.biasNode=null,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new $(512,512),this.mapType=ci,this.map=null,this.mapPass=null,this.matrix=new rt,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new br,this._frameExtents=new $(1,1),this._viewportCount=1,this._viewports=[new mt(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){let t=this.camera,i=this.matrix;vf.setFromMatrixPosition(e.matrixWorld),t.position.copy(vf),yf.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(yf),t.updateMatrixWorld(),jh.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix(jh,t.coordinateSystem,t.reversedDepth),t.coordinateSystem===gr||t.reversedDepth?i.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):i.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),i.multiply(jh)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this.biasNode=e.biasNode,this}clone(){return new this.constructor().copy(this)}toJSON(){let e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}},nl=new P,sl=new Pi,cn=new P,ko=class extends ft{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new rt,this.projectionMatrix=new rt,this.projectionMatrixInverse=new rt,this.coordinateSystem=$i,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(nl,sl,cn),cn.x===1&&cn.y===1&&cn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(nl,sl,cn.set(1,1,1)).invert()}updateWorldMatrix(e,t,i=!1){super.updateWorldMatrix(e,t,i),this.matrixWorld.decompose(nl,sl,cn),cn.x===1&&cn.y===1&&cn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(nl,sl,cn.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},is=new P,Mf=new $,Sf=new $,Qt=class extends ko{constructor(e=50,t=1,i=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=i,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){let t=.5*this.getFilmHeight()/e;this.fov=xr*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){let e=Math.tan(pr*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return xr*2*Math.atan(Math.tan(pr*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,i){is.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(is.x,is.y).multiplyScalar(-e/is.z),is.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(is.x,is.y).multiplyScalar(-e/is.z)}getViewSize(e,t){return this.getViewBounds(e,Mf,Sf),t.subVectors(Sf,Mf)}setViewOffset(e,t,i,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=this.near,t=e*Math.tan(pr*.5*this.fov)/this.zoom,i=2*t,s=this.aspect*i,r=-.5*s,o=this.view;if(this.view!==null&&this.view.enabled){let c=o.fullWidth,l=o.fullHeight;r+=o.offsetX*s/c,t-=o.offsetY*i/l,s*=o.width/c,i*=o.height/l}let a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-i,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}};var cu=class extends Xl{constructor(){super(new Qt(90,1,.5,500)),this.isPointLightShadow=!0}},Ho=class extends Rr{constructor(e,t,i=0,s=2){super(e,t),this.isPointLight=!0,this.type="PointLight",this.distance=i,this.decay=s,this.shadow=new cu}get power(){return this.intensity*4*Math.PI}set power(e){this.intensity=e/(4*Math.PI)}dispose(){super.dispose(),this.shadow.dispose()}copy(e,t){return super.copy(e,t),this.distance=e.distance,this.decay=e.decay,this.shadow=e.shadow.clone(),this}toJSON(e){let t=super.toJSON(e);return t.object.distance=this.distance,t.object.decay=this.decay,t.object.shadow=this.shadow.toJSON(),t}},as=class extends ko{constructor(e=-1,t=1,i=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=i,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,i,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=i-e,o=i+e,a=s+t,c=s-t;if(this.view!==null&&this.view.enabled){let l=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=l*this.view.offsetX,o=r+l*this.view.width,a-=h*this.view.offsetY,c=a-h*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,c,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}},hu=class extends Xl{constructor(){super(new as(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}},Cr=class extends Rr{constructor(e,t){super(e,t),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(ft.DEFAULT_UP),this.updateMatrix(),this.target=new ft,this.shadow=new hu}dispose(){super.dispose(),this.shadow.dispose()}copy(e){return super.copy(e),this.target=e.target.clone(),this.shadow=e.shadow.clone(),this}toJSON(e){let t=super.toJSON(e);return t.object.shadow=this.shadow.toJSON(),t.object.target=this.target.uuid,t}};var Vo=class extends ut{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){let e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}};var hr=-90,ur=1,ql=class extends ft{constructor(e,t,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new Qt(hr,ur,e,t);s.layers=this.layers,this.add(s);let r=new Qt(hr,ur,e,t);r.layers=this.layers,this.add(r);let o=new Qt(hr,ur,e,t);o.layers=this.layers,this.add(o);let a=new Qt(hr,ur,e,t);a.layers=this.layers,this.add(a);let c=new Qt(hr,ur,e,t);c.layers=this.layers,this.add(c);let l=new Qt(hr,ur,e,t);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){let e=this.coordinateSystem,t=this.children.concat(),[i,s,r,o,a,c]=t;for(let l of t)this.remove(l);if(e===$i)i.up.set(0,1,0),i.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(e===gr)i.up.set(0,-1,0),i.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(let l of t)this.add(l),l.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();let{renderTarget:i,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());let[r,o,a,c,l,h]=this.children,u=e.getRenderTarget(),d=e.getActiveCubeFace(),f=e.getActiveMipmapLevel(),g=e.xr.enabled;e.xr.enabled=!1;let x=i.texture.generateMipmaps;i.texture.generateMipmaps=!1;let p=!1;e.isWebGLRenderer===!0?p=e.state.buffers.depth.getReversed():p=e.reversedDepthBuffer,e.setRenderTarget(i,0,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,r),e.setRenderTarget(i,1,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,o),e.setRenderTarget(i,2,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,a),e.setRenderTarget(i,3,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,c),e.setRenderTarget(i,4,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,l),i.texture.generateMipmaps=x,e.setRenderTarget(i,5,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,h),e.setRenderTarget(u,d,f),e.xr.enabled=g,i.texture.needsPMREMUpdate=!0}},Yl=class extends Qt{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}},Go=class{constructor(){this._previousTime=0,this._currentTime=0,this._startTime=performance.now(),this._delta=0,this._elapsed=0,this._timescale=1,this._document=null,this._pageVisibilityHandler=null}connect(e){this._document=e,e.hidden!==void 0&&(this._pageVisibilityHandler=h0.bind(this),e.addEventListener("visibilitychange",this._pageVisibilityHandler,!1))}disconnect(){this._pageVisibilityHandler!==null&&(this._document.removeEventListener("visibilitychange",this._pageVisibilityHandler),this._pageVisibilityHandler=null),this._document=null}getDelta(){return this._delta/1e3}getElapsed(){return this._elapsed/1e3}getTimescale(){return this._timescale}setTimescale(e){return this._timescale=e,this}reset(){return this._currentTime=performance.now()-this._startTime,this}dispose(){this.disconnect()}update(e){return this._pageVisibilityHandler!==null&&this._document.hidden===!0?this._delta=0:(this._previousTime=this._currentTime,this._currentTime=(e!==void 0?e:performance.now())-this._startTime,this._delta=(this._currentTime-this._previousTime)*this._timescale,this._elapsed+=this._delta),this}};function h0(){this._document.hidden===!1&&this.reset()}var Cu="\\[\\]\\.:\\/",u0=new RegExp("["+Cu+"]","g"),Pu="[^"+Cu+"]",d0="[^"+Cu.replace("\\.","")+"]",f0=/((?:WC+[\/:])*)/.source.replace("WC",Pu),p0=/(WCOD+)?/.source.replace("WCOD",d0),m0=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",Pu),g0=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",Pu),_0=new RegExp("^"+f0+p0+m0+g0+"$"),x0=["material","materials","bones","map"],uu=class{constructor(e,t,i){let s=i||Et.parseTrackName(t);this._targetGroup=e,this._bindings=e.subscribe_(t,s)}getValue(e,t){this.bind();let i=this._targetGroup.nCachedObjects_,s=this._bindings[i];s!==void 0&&s.getValue(e,t)}setValue(e,t){let i=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=i.length;s!==r;++s)i[s].setValue(e,t)}bind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,i=e.length;t!==i;++t)e[t].bind()}unbind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,i=e.length;t!==i;++t)e[t].unbind()}},Et=class n{constructor(e,t,i){this.path=t,this.parsedPath=i||n.parseTrackName(t),this.node=n.findNode(e,this.parsedPath.nodeName),this.rootNode=e,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(e,t,i){return e&&e.isAnimationObjectGroup?new n.Composite(e,t,i):new n(e,t,i)}static sanitizeNodeName(e){return e.replace(/\s/g,"_").replace(u0,"")}static parseTrackName(e){let t=_0.exec(e);if(t===null)throw new Error("THREE.PropertyBinding: Cannot parse trackName: "+e);let i={nodeName:t[2],objectName:t[3],objectIndex:t[4],propertyName:t[5],propertyIndex:t[6]},s=i.nodeName&&i.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=i.nodeName.substring(s+1);x0.indexOf(r)!==-1&&(i.nodeName=i.nodeName.substring(0,s),i.objectName=r)}if(i.propertyName===null||i.propertyName.length===0)throw new Error("THREE.PropertyBinding: can not parse propertyName from trackName: "+e);return i}static findNode(e,t){if(t===void 0||t===""||t==="."||t===-1||t===e.name||t===e.uuid)return e;if(e.skeleton){let i=e.skeleton.getBoneByName(t);if(i!==void 0)return i}if(e.children){let i=function(r){for(let o=0;o<r.length;o++){let a=r[o];if(a.name===t||a.uuid===t)return a;let c=i(a.children);if(c)return c}return null},s=i(e.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,t){e[t]=this.targetObject[this.propertyName]}_getValue_array(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)e[t++]=i[s]}_getValue_arrayElement(e,t){e[t]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,t){this.resolvedProperty.toArray(e,t)}_setValue_direct(e,t){this.targetObject[this.propertyName]=e[t]}_setValue_direct_setNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[t++]}_setValue_array_setNeedsUpdate(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[t++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[t++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,t){this.resolvedProperty[this.propertyIndex]=e[t]}_setValue_arrayElement_setNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,t){this.resolvedProperty.fromArray(e,t)}_setValue_fromArray_setNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,t){this.bind(),this.getValue(e,t)}_setValue_unbound(e,t){this.bind(),this.setValue(e,t)}bind(){let e=this.node,t=this.parsedPath,i=t.objectName,s=t.propertyName,r=t.propertyIndex;if(e||(e=n.findNode(this.rootNode,t.nodeName),this.node=e),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!e){$e("PropertyBinding: No target node found for track: "+this.path+".");return}if(i){let l=t.objectIndex;switch(i){case"materials":if(!e.material){Ze("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.materials){Ze("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}e=e.material.materials;break;case"bones":if(!e.skeleton){Ze("PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}e=e.skeleton.bones;for(let h=0;h<e.length;h++)if(e[h].name===l){l=h;break}break;case"map":if("map"in e){e=e.map;break}if(!e.material){Ze("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.map){Ze("PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}e=e.material.map;break;default:if(e[i]===void 0){Ze("PropertyBinding: Can not bind to objectName of node undefined.",this);return}e=e[i]}if(l!==void 0){if(e[l]===void 0){Ze("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,e);return}e=e[l]}}let o=e[s];if(o===void 0){let l=t.nodeName;Ze("PropertyBinding: Trying to update property for track: "+l+"."+s+" but it wasn't found.",e);return}let a=this.Versioning.None;this.targetObject=e,e.isMaterial===!0?a=this.Versioning.NeedsUpdate:e.isObject3D===!0&&(a=this.Versioning.MatrixWorldNeedsUpdate);let c=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!e.geometry){Ze("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!e.geometry.morphAttributes){Ze("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}e.morphTargetDictionary[r]!==void 0&&(r=e.morphTargetDictionary[r])}c=this.BindingType.ArrayElement,this.resolvedProperty=o,this.propertyIndex=r}else o.fromArray!==void 0&&o.toArray!==void 0?(c=this.BindingType.HasFromToArray,this.resolvedProperty=o):Array.isArray(o)?(c=this.BindingType.EntireArray,this.resolvedProperty=o):this.propertyName=s;this.getValue=this.GetterByBindingType[c],this.setValue=this.SetterByBindingTypeAndVersioning[c][a]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};Et.Composite=uu;Et.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};Et.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};Et.prototype.GetterByBindingType=[Et.prototype._getValue_direct,Et.prototype._getValue_array,Et.prototype._getValue_arrayElement,Et.prototype._getValue_toArray];Et.prototype.SetterByBindingTypeAndVersioning=[[Et.prototype._setValue_direct,Et.prototype._setValue_direct_setNeedsUpdate,Et.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_array,Et.prototype._setValue_array_setNeedsUpdate,Et.prototype._setValue_array_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_arrayElement,Et.prototype._setValue_arrayElement_setNeedsUpdate,Et.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_fromArray,Et.prototype._setValue_fromArray_setNeedsUpdate,Et.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var TS=new Float32Array(1);var ls=class extends go{constructor(e,t,i=1){super(e,t),this.isInstancedInterleavedBuffer=!0,this.meshPerAttribute=i}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}clone(e){let t=super.clone(e);return t.meshPerAttribute=this.meshPerAttribute,t}toJSON(e){let t=super.toJSON(e);return t.isInstancedInterleavedBuffer=!0,t.meshPerAttribute=this.meshPerAttribute,t}};var bf=new rt,Wo=class{constructor(e,t,i=0,s=1/0){this.ray=new Dn(e,t),this.near=i,this.far=s,this.camera=null,this.layers=new yr,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(e,t){this.ray.set(e,t)}setFromCamera(e,t){t.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t):t.isOrthographicCamera?(this.ray.origin.set(e.x,e.y,t.projectionMatrix.elements[14]).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t):Ze("Raycaster: Unsupported camera type: "+t.type)}setFromXRController(e){return bf.identity().extractRotation(e.matrixWorld),this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(bf),this}intersectObject(e,t=!0,i=[]){return du(e,this,i,t),i.sort(Ef),i}intersectObjects(e,t=!0,i=[]){for(let s=0,r=e.length;s<r;s++)du(e[s],this,i,t);return i.sort(Ef),i}};function Ef(n,e){return n.distance-e.distance}function du(n,e,t,i){let s=!0;if(n.layers.test(e.layers)&&n.raycast(e,t)===!1&&(s=!1),s===!0&&i===!0){let r=n.children;for(let o=0,a=r.length;o<a;o++)du(r[o],e,t,!0)}}var Pr=class{constructor(e=1,t=0,i=0){this.radius=e,this.phi=t,this.theta=i}set(e,t,i){return this.radius=e,this.phi=t,this.theta=i,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=je(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,i){return this.radius=Math.sqrt(e*e+t*t+i*i),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,i),this.phi=Math.acos(je(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}};var Fu=class Fu{constructor(e,t,i,s){this.elements=[1,0,0,1],e!==void 0&&this.set(e,t,i,s)}identity(){return this.set(1,0,0,1),this}fromArray(e,t=0){for(let i=0;i<4;i++)this.elements[i]=e[i+t];return this}set(e,t,i,s){let r=this.elements;return r[0]=e,r[2]=t,r[1]=i,r[3]=s,this}};Fu.prototype.isMatrix2=!0;var fu=Fu;var wf=new P,rl=new P,dr=new P,fr=new P,Kh=new P,v0=new P,y0=new P,Xo=class{constructor(e=new P,t=new P){this.start=e,this.end=t}set(e,t){return this.start.copy(e),this.end.copy(t),this}copy(e){return this.start.copy(e.start),this.end.copy(e.end),this}getCenter(e){return e.addVectors(this.start,this.end).multiplyScalar(.5)}delta(e){return e.subVectors(this.end,this.start)}distanceSq(){return this.start.distanceToSquared(this.end)}distance(){return this.start.distanceTo(this.end)}at(e,t){return this.delta(t).multiplyScalar(e).add(this.start)}closestPointToPointParameter(e,t){wf.subVectors(e,this.start),rl.subVectors(this.end,this.start);let i=rl.dot(rl);if(i===0)return 0;let r=rl.dot(wf)/i;return t&&(r=je(r,0,1)),r}closestPointToPoint(e,t,i){let s=this.closestPointToPointParameter(e,t);return this.delta(i).multiplyScalar(s).add(this.start)}distanceSqToLine3(e,t=v0,i=y0){let s=10000000000000001e-32,r,o,a=this.start,c=e.start,l=this.end,h=e.end;dr.subVectors(l,a),fr.subVectors(h,c),Kh.subVectors(a,c);let u=dr.dot(dr),d=fr.dot(fr),f=fr.dot(Kh);if(u<=s&&d<=s)return t.copy(a),i.copy(c),t.sub(i),t.dot(t);if(u<=s)r=0,o=f/d,o=je(o,0,1);else{let g=dr.dot(Kh);if(d<=s)o=0,r=je(-g/u,0,1);else{let x=dr.dot(fr),p=u*d-x*x;p!==0?r=je((x*f-g*d)/p,0,1):r=0,o=(x*r+f)/d,o<0?(o=0,r=je(-g/u,0,1)):o>1&&(o=1,r=je((x-g)/u,0,1))}}return t.copy(a).addScaledVector(dr,r),i.copy(c).addScaledVector(fr,o),t.distanceToSquared(i)}applyMatrix4(e){return this.start.applyMatrix4(e),this.end.applyMatrix4(e),this}equals(e){return e.start.equals(this.start)&&e.end.equals(this.end)}clone(){return new this.constructor().copy(this)}};var qo=class extends Ji{constructor(e,t=null){super(),this.object=e,this.domElement=t,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(e){if(e===void 0){$e("Controls: connect() now requires an element.");return}this.domElement!==null&&this.disconnect(),this.domElement=e}disconnect(){}dispose(){}update(){}};function Iu(n,e,t,i){let s=M0(i);switch(t){case bu:return n*e;case nc:return n*e/s.components*s.byteLength;case sc:return n*e/s.components*s.byteLength;case ms:return n*e*2/s.components*s.byteLength;case rc:return n*e*2/s.components*s.byteLength;case Eu:return n*e*3/s.components*s.byteLength;case Si:return n*e*4/s.components*s.byteLength;case oc:return n*e*4/s.components*s.byteLength;case ia:case na:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*8;case sa:case ra:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case lc:case hc:return Math.max(n,16)*Math.max(e,8)/4;case ac:case cc:return Math.max(n,8)*Math.max(e,8)/2;case uc:case dc:case pc:case mc:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*8;case fc:case oa:case gc:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case _c:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case xc:return Math.floor((n+4)/5)*Math.floor((e+3)/4)*16;case vc:return Math.floor((n+4)/5)*Math.floor((e+4)/5)*16;case yc:return Math.floor((n+5)/6)*Math.floor((e+4)/5)*16;case Mc:return Math.floor((n+5)/6)*Math.floor((e+5)/6)*16;case Sc:return Math.floor((n+7)/8)*Math.floor((e+4)/5)*16;case bc:return Math.floor((n+7)/8)*Math.floor((e+5)/6)*16;case Ec:return Math.floor((n+7)/8)*Math.floor((e+7)/8)*16;case wc:return Math.floor((n+9)/10)*Math.floor((e+4)/5)*16;case Tc:return Math.floor((n+9)/10)*Math.floor((e+5)/6)*16;case Ac:return Math.floor((n+9)/10)*Math.floor((e+7)/8)*16;case Rc:return Math.floor((n+9)/10)*Math.floor((e+9)/10)*16;case Cc:return Math.floor((n+11)/12)*Math.floor((e+9)/10)*16;case Pc:return Math.floor((n+11)/12)*Math.floor((e+11)/12)*16;case Ic:case Dc:case Lc:return Math.ceil(n/4)*Math.ceil(e/4)*16;case Uc:case Nc:return Math.ceil(n/4)*Math.ceil(e/4)*8;case aa:case Fc:return Math.ceil(n/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function M0(n){switch(n){case ci:case vu:return{byteLength:1,components:1};case Dr:case yu:case ii:return{byteLength:2,components:1};case tc:case ic:return{byteLength:2,components:4};case tn:case ec:case Hi:return{byteLength:4,components:1};case Mu:case Su:return{byteLength:4,components:3}}throw new Error(`THREE.TextureUtils: Unknown texture type ${n}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"185"}}));typeof window<"u"&&(window.__THREE__?$e("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="185");/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function Lp(){let n=null,e=!1,t=null,i=null;function s(r,o){t(r,o),i=n.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&n!==null&&(i=n.requestAnimationFrame(s),e=!0)},stop:function(){n!==null&&n.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){n=r}}}function b0(n){let e=new WeakMap;function t(a,c){let l=a.array,h=a.usage,u=l.byteLength,d=n.createBuffer();n.bindBuffer(c,d),n.bufferData(c,l,h),a.onUploadCallback();let f;if(l instanceof Float32Array)f=n.FLOAT;else if(typeof Float16Array<"u"&&l instanceof Float16Array)f=n.HALF_FLOAT;else if(l instanceof Uint16Array)a.isFloat16BufferAttribute?f=n.HALF_FLOAT:f=n.UNSIGNED_SHORT;else if(l instanceof Int16Array)f=n.SHORT;else if(l instanceof Uint32Array)f=n.UNSIGNED_INT;else if(l instanceof Int32Array)f=n.INT;else if(l instanceof Int8Array)f=n.BYTE;else if(l instanceof Uint8Array)f=n.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)f=n.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:d,type:f,bytesPerElement:l.BYTES_PER_ELEMENT,version:a.version,size:u}}function i(a,c,l){let h=c.array,u=c.updateRanges;if(n.bindBuffer(l,a),u.length===0)n.bufferSubData(l,0,h);else{u.sort((f,g)=>f.start-g.start);let d=0;for(let f=1;f<u.length;f++){let g=u[d],x=u[f];x.start<=g.start+g.count+1?g.count=Math.max(g.count,x.start+x.count-g.start):(++d,u[d]=x)}u.length=d+1;for(let f=0,g=u.length;f<g;f++){let x=u[f];n.bufferSubData(l,x.start*h.BYTES_PER_ELEMENT,h,x.start,x.count)}c.clearUpdateRanges()}c.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);let c=e.get(a);c&&(n.deleteBuffer(c.buffer),e.delete(a))}function o(a,c){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){let h=e.get(a);(!h||h.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}let l=e.get(a);if(l===void 0)e.set(a,t(a,c));else if(l.version<a.version){if(l.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(l.buffer,a,c),l.version=a.version}}return{get:s,remove:r,update:o}}var E0=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,w0=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,T0=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,A0=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,R0=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,C0=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,P0=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,I0=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,D0=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,L0=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,U0=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,N0=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,F0=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,O0=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,B0=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,z0=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,k0=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,H0=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,V0=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,G0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,W0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,X0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,q0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,Y0=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
#define inverseTransformDirection transformDirectionByInverseViewMatrix
vec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) {
	return normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz );
}
vec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 viewMatrix ) {
	return normalize( ( vec4( dir, 0.0 ) * viewMatrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,$0=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Z0=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
#endif`,J0=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,j0=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,K0=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Q0=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,e_="gl_FragColor = linearToOutputTexel( gl_FragColor );",t_=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,i_=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * reflectVec );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,n_=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,s_=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,r_=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,o_=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,a_=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,l_=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,c_=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,h_=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,u_=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,d_=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,f_=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,p_=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,m_=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif
#include <lightprobes_pars_fragment>`,g_=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,__=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,x_=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,v_=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,y_=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,M_=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,S_=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
		vec3 iridescenceFresnelDielectric;
		vec3 iridescenceFresnelMetallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		return 0.5 / max( gv + gl, EPSILON );
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;
	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;
	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,b_=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
	#ifdef USE_LIGHT_PROBES_GRID
		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;
		vec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );
		irradiance += getLightProbeGridIrradiance( probeWorldPos, probeWorldNormal );
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,E_=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,w_=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,T_=`#ifdef USE_LIGHT_PROBES_GRID
uniform highp sampler3D probesSH;
uniform vec3 probesMin;
uniform vec3 probesMax;
uniform vec3 probesResolution;
vec3 getLightProbeGridIrradiance( vec3 worldPos, vec3 worldNormal ) {
	vec3 res = probesResolution;
	vec3 gridRange = probesMax - probesMin;
	vec3 resMinusOne = res - 1.0;
	vec3 probeSpacing = gridRange / resMinusOne;
	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;
	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.0, 1.0 );
	uvw = uvw * resMinusOne / res + 0.5 / res;
	float nz          = res.z;
	float paddedSlices = nz + 2.0;
	float atlasDepth  = 7.0 * paddedSlices;
	float uvZBase     = uvw.z * nz + 1.0;
	vec4 s0 = texture( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ) );
	vec4 s1 = texture( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ) );
	vec4 s2 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s3 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s4 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s5 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s6 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ) );
	vec3 c0 = s0.xyz;
	vec3 c1 = vec3( s0.w, s1.xy );
	vec3 c2 = vec3( s1.zw, s2.x );
	vec3 c3 = s2.yzw;
	vec3 c4 = s3.xyz;
	vec3 c5 = vec3( s3.w, s4.xy );
	vec3 c6 = vec3( s4.zw, s5.x );
	vec3 c7 = s5.yzw;
	vec3 c8 = s6.xyz;
	float x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;
	vec3 result = c0 * 0.886227;
	result += c1 * 2.0 * 0.511664 * y;
	result += c2 * 2.0 * 0.511664 * z;
	result += c3 * 2.0 * 0.511664 * x;
	result += c4 * 2.0 * 0.429043 * x * y;
	result += c5 * 2.0 * 0.429043 * y * z;
	result += c6 * ( 0.743125 * z * z - 0.247708 );
	result += c7 * 2.0 * 0.429043 * x * z;
	result += c8 * 0.429043 * ( x * x - y * y );
	return max( result, vec3( 0.0 ) );
}
#endif`,A_=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,R_=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,C_=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,P_=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,I_=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,D_=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,L_=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,U_=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,N_=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,F_=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,O_=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,B_=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,z_=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,k_=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,H_=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,V_=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#ifdef DOUBLE_SIDED
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#ifdef DOUBLE_SIDED
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,G_=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#if defined( USE_PACKED_NORMALMAP )
		mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );
	#endif
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,W_=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,X_=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,q_=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,Y_=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,$_=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Z_=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,J_=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,j_=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,K_=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,Q_=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,ex=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,tx=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,ix=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,nx=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,sx=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,rx=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,ox=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			vec2 sample0 = vogelDiskSample( 0, 5, phi );
			vec2 sample1 = vogelDiskSample( 1, 5, phi );
			vec2 sample2 = vogelDiskSample( 2, 5, phi );
			vec2 sample3 = vogelDiskSample( 3, 5, phi );
			vec2 sample4 = vogelDiskSample( 4, 5, phi );
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,ax=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,lx=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	#ifdef HAS_NORMAL
		vec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
	#else
		vec3 shadowWorldNormal = vec3( 0.0 );
	#endif
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,cx=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,hx=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,ux=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,dx=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,fx=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,px=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,mx=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,gx=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,_x=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,xx=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,vx=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,yx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Mx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Sx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,bx=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,Ex=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,wx=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Tx=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Ax=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Rx=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Cx=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Px=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Ix=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Dx=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Lx=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,Ux=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Nx=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Fx=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Ox=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Bx=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,zx=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,kx=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Hx=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Vx=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Gx=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Wx=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Xx=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,qx=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Yx=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,$x=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,Zx=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Jx=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,jx=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Kx=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,Qx=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,ev=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,tv=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,iv=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,nv=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,at={alphahash_fragment:E0,alphahash_pars_fragment:w0,alphamap_fragment:T0,alphamap_pars_fragment:A0,alphatest_fragment:R0,alphatest_pars_fragment:C0,aomap_fragment:P0,aomap_pars_fragment:I0,batching_pars_vertex:D0,batching_vertex:L0,begin_vertex:U0,beginnormal_vertex:N0,bsdfs:F0,iridescence_fragment:O0,bumpmap_pars_fragment:B0,clipping_planes_fragment:z0,clipping_planes_pars_fragment:k0,clipping_planes_pars_vertex:H0,clipping_planes_vertex:V0,color_fragment:G0,color_pars_fragment:W0,color_pars_vertex:X0,color_vertex:q0,common:Y0,cube_uv_reflection_fragment:$0,defaultnormal_vertex:Z0,displacementmap_pars_vertex:J0,displacementmap_vertex:j0,emissivemap_fragment:K0,emissivemap_pars_fragment:Q0,colorspace_fragment:e_,colorspace_pars_fragment:t_,envmap_fragment:i_,envmap_common_pars_fragment:n_,envmap_pars_fragment:s_,envmap_pars_vertex:r_,envmap_physical_pars_fragment:g_,envmap_vertex:o_,fog_vertex:a_,fog_pars_vertex:l_,fog_fragment:c_,fog_pars_fragment:h_,gradientmap_pars_fragment:u_,lightmap_pars_fragment:d_,lights_lambert_fragment:f_,lights_lambert_pars_fragment:p_,lights_pars_begin:m_,lights_toon_fragment:__,lights_toon_pars_fragment:x_,lights_phong_fragment:v_,lights_phong_pars_fragment:y_,lights_physical_fragment:M_,lights_physical_pars_fragment:S_,lights_fragment_begin:b_,lights_fragment_maps:E_,lights_fragment_end:w_,lightprobes_pars_fragment:T_,logdepthbuf_fragment:A_,logdepthbuf_pars_fragment:R_,logdepthbuf_pars_vertex:C_,logdepthbuf_vertex:P_,map_fragment:I_,map_pars_fragment:D_,map_particle_fragment:L_,map_particle_pars_fragment:U_,metalnessmap_fragment:N_,metalnessmap_pars_fragment:F_,morphinstance_vertex:O_,morphcolor_vertex:B_,morphnormal_vertex:z_,morphtarget_pars_vertex:k_,morphtarget_vertex:H_,normal_fragment_begin:V_,normal_fragment_maps:G_,normal_pars_fragment:W_,normal_pars_vertex:X_,normal_vertex:q_,normalmap_pars_fragment:Y_,clearcoat_normal_fragment_begin:$_,clearcoat_normal_fragment_maps:Z_,clearcoat_pars_fragment:J_,iridescence_pars_fragment:j_,opaque_fragment:K_,packing:Q_,premultiplied_alpha_fragment:ex,project_vertex:tx,dithering_fragment:ix,dithering_pars_fragment:nx,roughnessmap_fragment:sx,roughnessmap_pars_fragment:rx,shadowmap_pars_fragment:ox,shadowmap_pars_vertex:ax,shadowmap_vertex:lx,shadowmask_pars_fragment:cx,skinbase_vertex:hx,skinning_pars_vertex:ux,skinning_vertex:dx,skinnormal_vertex:fx,specularmap_fragment:px,specularmap_pars_fragment:mx,tonemapping_fragment:gx,tonemapping_pars_fragment:_x,transmission_fragment:xx,transmission_pars_fragment:vx,uv_pars_fragment:yx,uv_pars_vertex:Mx,uv_vertex:Sx,worldpos_vertex:bx,background_vert:Ex,background_frag:wx,backgroundCube_vert:Tx,backgroundCube_frag:Ax,cube_vert:Rx,cube_frag:Cx,depth_vert:Px,depth_frag:Ix,distance_vert:Dx,distance_frag:Lx,equirect_vert:Ux,equirect_frag:Nx,linedashed_vert:Fx,linedashed_frag:Ox,meshbasic_vert:Bx,meshbasic_frag:zx,meshlambert_vert:kx,meshlambert_frag:Hx,meshmatcap_vert:Vx,meshmatcap_frag:Gx,meshnormal_vert:Wx,meshnormal_frag:Xx,meshphong_vert:qx,meshphong_frag:Yx,meshphysical_vert:$x,meshphysical_frag:Zx,meshtoon_vert:Jx,meshtoon_frag:jx,points_vert:Kx,points_frag:Qx,shadow_vert:ev,shadow_frag:tv,sprite_vert:iv,sprite_frag:nv},be={common:{diffuse:{value:new Te(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new tt},alphaMap:{value:null},alphaMapTransform:{value:new tt},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new tt}},envmap:{envMap:{value:null},envMapRotation:{value:new tt},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new tt}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new tt}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new tt},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new tt},normalScale:{value:new $(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new tt},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new tt}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new tt}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new tt}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Te(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:null},probesMin:{value:new P},probesMax:{value:new P},probesResolution:{value:new P}},points:{diffuse:{value:new Te(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new tt},alphaTest:{value:0},uvTransform:{value:new tt}},sprite:{diffuse:{value:new Te(16777215)},opacity:{value:1},center:{value:new $(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new tt},alphaMap:{value:null},alphaMapTransform:{value:new tt},alphaTest:{value:0}}},gi={basic:{uniforms:hi([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.fog]),vertexShader:at.meshbasic_vert,fragmentShader:at.meshbasic_frag},lambert:{uniforms:hi([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.fog,be.lights,{emissive:{value:new Te(0)},envMapIntensity:{value:1}}]),vertexShader:at.meshlambert_vert,fragmentShader:at.meshlambert_frag},phong:{uniforms:hi([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.fog,be.lights,{emissive:{value:new Te(0)},specular:{value:new Te(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:at.meshphong_vert,fragmentShader:at.meshphong_frag},standard:{uniforms:hi([be.common,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.roughnessmap,be.metalnessmap,be.fog,be.lights,{emissive:{value:new Te(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:at.meshphysical_vert,fragmentShader:at.meshphysical_frag},toon:{uniforms:hi([be.common,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.gradientmap,be.fog,be.lights,{emissive:{value:new Te(0)}}]),vertexShader:at.meshtoon_vert,fragmentShader:at.meshtoon_frag},matcap:{uniforms:hi([be.common,be.bumpmap,be.normalmap,be.displacementmap,be.fog,{matcap:{value:null}}]),vertexShader:at.meshmatcap_vert,fragmentShader:at.meshmatcap_frag},points:{uniforms:hi([be.points,be.fog]),vertexShader:at.points_vert,fragmentShader:at.points_frag},dashed:{uniforms:hi([be.common,be.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:at.linedashed_vert,fragmentShader:at.linedashed_frag},depth:{uniforms:hi([be.common,be.displacementmap]),vertexShader:at.depth_vert,fragmentShader:at.depth_frag},normal:{uniforms:hi([be.common,be.bumpmap,be.normalmap,be.displacementmap,{opacity:{value:1}}]),vertexShader:at.meshnormal_vert,fragmentShader:at.meshnormal_frag},sprite:{uniforms:hi([be.sprite,be.fog]),vertexShader:at.sprite_vert,fragmentShader:at.sprite_frag},background:{uniforms:{uvTransform:{value:new tt},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:at.background_vert,fragmentShader:at.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new tt}},vertexShader:at.backgroundCube_vert,fragmentShader:at.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:at.cube_vert,fragmentShader:at.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:at.equirect_vert,fragmentShader:at.equirect_frag},distance:{uniforms:hi([be.common,be.displacementmap,{referencePosition:{value:new P},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:at.distance_vert,fragmentShader:at.distance_frag},shadow:{uniforms:hi([be.lights,be.fog,{color:{value:new Te(0)},opacity:{value:1}}]),vertexShader:at.shadow_vert,fragmentShader:at.shadow_frag}};gi.physical={uniforms:hi([gi.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new tt},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new tt},clearcoatNormalScale:{value:new $(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new tt},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new tt},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new tt},sheen:{value:0},sheenColor:{value:new Te(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new tt},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new tt},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new tt},transmissionSamplerSize:{value:new $},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new tt},attenuationDistance:{value:0},attenuationColor:{value:new Te(0)},specularColor:{value:new Te(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new tt},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new tt},anisotropyVector:{value:new $},anisotropyMap:{value:null},anisotropyMapTransform:{value:new tt}}]),vertexShader:at.meshphysical_vert,fragmentShader:at.meshphysical_frag};var zc={r:0,b:0,g:0},sv=new rt,Up=new tt;Up.set(-1,0,0,0,1,0,0,0,1);function rv(n,e,t,i,s,r){let o=new Te(0),a=s===!0?0:1,c,l,h=null,u=0,d=null;function f(M){let b=M.isScene===!0?M.background:null;if(b&&b.isTexture){let y=M.backgroundBlurriness>0;b=e.get(b,y)}return b}function g(M){let b=!1,y=f(M);y===null?p(o,a):y&&y.isColor&&(p(y,1),b=!0);let T=n.xr.getEnvironmentBlendMode();T==="additive"?t.buffers.color.setClear(0,0,0,1,r):T==="alpha-blend"&&t.buffers.color.setClear(0,0,0,0,r),(n.autoClear||b)&&(t.buffers.depth.setTest(!0),t.buffers.depth.setMask(!0),t.buffers.color.setMask(!0),n.clear(n.autoClearColor,n.autoClearDepth,n.autoClearStencil))}function x(M,b){let y=f(b);y&&(y.isCubeTexture||y.mapping===ea)?(l===void 0&&(l=new Ke(new Bt(1,1,1),new bt({name:"BackgroundCubeMaterial",uniforms:Bs(gi.backgroundCube.uniforms),vertexShader:gi.backgroundCube.vertexShader,fragmentShader:gi.backgroundCube.fragmentShader,side:ti,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),l.geometry.deleteAttribute("uv"),l.onBeforeRender=function(T,S,A){this.matrixWorld.copyPosition(A.matrixWorld)},Object.defineProperty(l.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(l)),l.material.uniforms.envMap.value=y,l.material.uniforms.backgroundBlurriness.value=b.backgroundBlurriness,l.material.uniforms.backgroundIntensity.value=b.backgroundIntensity,l.material.uniforms.backgroundRotation.value.setFromMatrix4(sv.makeRotationFromEuler(b.backgroundRotation)).transpose(),y.isCubeTexture&&y.isRenderTargetTexture===!1&&l.material.uniforms.backgroundRotation.value.premultiply(Up),l.material.toneMapped=ht.getTransfer(y.colorSpace)!==pt,(h!==y||u!==y.version||d!==n.toneMapping)&&(l.material.needsUpdate=!0,h=y,u=y.version,d=n.toneMapping),l.layers.enableAll(),M.unshift(l,l.geometry,l.material,0,0,null)):y&&y.isTexture&&(c===void 0&&(c=new Ke(new ki(2,2),new bt({name:"BackgroundMaterial",uniforms:Bs(gi.background.uniforms),vertexShader:gi.background.vertexShader,fragmentShader:gi.background.fragmentShader,side:Zi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(c)),c.material.uniforms.t2D.value=y,c.material.uniforms.backgroundIntensity.value=b.backgroundIntensity,c.material.toneMapped=ht.getTransfer(y.colorSpace)!==pt,y.matrixAutoUpdate===!0&&y.updateMatrix(),c.material.uniforms.uvTransform.value.copy(y.matrix),(h!==y||u!==y.version||d!==n.toneMapping)&&(c.material.needsUpdate=!0,h=y,u=y.version,d=n.toneMapping),c.layers.enableAll(),M.unshift(c,c.geometry,c.material,0,0,null))}function p(M,b){M.getRGB(zc,Ru(n)),t.buffers.color.setClear(zc.r,zc.g,zc.b,b,r)}function m(){l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return o},setClearColor:function(M,b=1){o.set(M),a=b,p(o,a)},getClearAlpha:function(){return a},setClearAlpha:function(M){a=M,p(o,a)},render:g,addToRenderList:x,dispose:m}}function ov(n,e){let t=n.getParameter(n.MAX_VERTEX_ATTRIBS),i={},s=d(null),r=s,o=!1;function a(I,L,V,q,N){let Y=!1,X=u(I,q,V,L);r!==X&&(r=X,l(r.object)),Y=f(I,q,V,N),Y&&g(I,q,V,N),N!==null&&e.update(N,n.ELEMENT_ARRAY_BUFFER),(Y||o)&&(o=!1,y(I,L,V,q),N!==null&&n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,e.get(N).buffer))}function c(){return n.createVertexArray()}function l(I){return n.bindVertexArray(I)}function h(I){return n.deleteVertexArray(I)}function u(I,L,V,q){let N=q.wireframe===!0,Y=i[L.id];Y===void 0&&(Y={},i[L.id]=Y);let X=I.isInstancedMesh===!0?I.id:0,ne=Y[X];ne===void 0&&(ne={},Y[X]=ne);let ie=ne[V.id];ie===void 0&&(ie={},ne[V.id]=ie);let ge=ie[N];return ge===void 0&&(ge=d(c()),ie[N]=ge),ge}function d(I){let L=[],V=[],q=[];for(let N=0;N<t;N++)L[N]=0,V[N]=0,q[N]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:L,enabledAttributes:V,attributeDivisors:q,object:I,attributes:{},index:null}}function f(I,L,V,q){let N=r.attributes,Y=L.attributes,X=0,ne=V.getAttributes();for(let ie in ne)if(ne[ie].location>=0){let ue=N[ie],xe=Y[ie];if(xe===void 0&&(ie==="instanceMatrix"&&I.instanceMatrix&&(xe=I.instanceMatrix),ie==="instanceColor"&&I.instanceColor&&(xe=I.instanceColor)),ue===void 0||ue.attribute!==xe||xe&&ue.data!==xe.data)return!0;X++}return r.attributesNum!==X||r.index!==q}function g(I,L,V,q){let N={},Y=L.attributes,X=0,ne=V.getAttributes();for(let ie in ne)if(ne[ie].location>=0){let ue=Y[ie];ue===void 0&&(ie==="instanceMatrix"&&I.instanceMatrix&&(ue=I.instanceMatrix),ie==="instanceColor"&&I.instanceColor&&(ue=I.instanceColor));let xe={};xe.attribute=ue,ue&&ue.data&&(xe.data=ue.data),N[ie]=xe,X++}r.attributes=N,r.attributesNum=X,r.index=q}function x(){let I=r.newAttributes;for(let L=0,V=I.length;L<V;L++)I[L]=0}function p(I){m(I,0)}function m(I,L){let V=r.newAttributes,q=r.enabledAttributes,N=r.attributeDivisors;V[I]=1,q[I]===0&&(n.enableVertexAttribArray(I),q[I]=1),N[I]!==L&&(n.vertexAttribDivisor(I,L),N[I]=L)}function M(){let I=r.newAttributes,L=r.enabledAttributes;for(let V=0,q=L.length;V<q;V++)L[V]!==I[V]&&(n.disableVertexAttribArray(V),L[V]=0)}function b(I,L,V,q,N,Y,X){X===!0?n.vertexAttribIPointer(I,L,V,N,Y):n.vertexAttribPointer(I,L,V,q,N,Y)}function y(I,L,V,q){x();let N=q.attributes,Y=V.getAttributes(),X=L.defaultAttributeValues;for(let ne in Y){let ie=Y[ne];if(ie.location>=0){let ge=N[ne];if(ge===void 0&&(ne==="instanceMatrix"&&I.instanceMatrix&&(ge=I.instanceMatrix),ne==="instanceColor"&&I.instanceColor&&(ge=I.instanceColor)),ge!==void 0){let ue=ge.normalized,xe=ge.itemSize,Ne=e.get(ge);if(Ne===void 0)continue;let st=Ne.buffer,Xe=Ne.type,j=Ne.bytesPerElement,he=Xe===n.INT||Xe===n.UNSIGNED_INT||ge.gpuType===ec;if(ge.isInterleavedBufferAttribute){let le=ge.data,Ae=le.stride,Fe=ge.offset;if(le.isInstancedInterleavedBuffer){for(let ke=0;ke<ie.locationSize;ke++)m(ie.location+ke,le.meshPerAttribute);I.isInstancedMesh!==!0&&q._maxInstanceCount===void 0&&(q._maxInstanceCount=le.meshPerAttribute*le.count)}else for(let ke=0;ke<ie.locationSize;ke++)p(ie.location+ke);n.bindBuffer(n.ARRAY_BUFFER,st);for(let ke=0;ke<ie.locationSize;ke++)b(ie.location+ke,xe/ie.locationSize,Xe,ue,Ae*j,(Fe+xe/ie.locationSize*ke)*j,he)}else{if(ge.isInstancedBufferAttribute){for(let le=0;le<ie.locationSize;le++)m(ie.location+le,ge.meshPerAttribute);I.isInstancedMesh!==!0&&q._maxInstanceCount===void 0&&(q._maxInstanceCount=ge.meshPerAttribute*ge.count)}else for(let le=0;le<ie.locationSize;le++)p(ie.location+le);n.bindBuffer(n.ARRAY_BUFFER,st);for(let le=0;le<ie.locationSize;le++)b(ie.location+le,xe/ie.locationSize,Xe,ue,xe*j,xe/ie.locationSize*le*j,he)}}else if(X!==void 0){let ue=X[ne];if(ue!==void 0)switch(ue.length){case 2:n.vertexAttrib2fv(ie.location,ue);break;case 3:n.vertexAttrib3fv(ie.location,ue);break;case 4:n.vertexAttrib4fv(ie.location,ue);break;default:n.vertexAttrib1fv(ie.location,ue)}}}}M()}function T(){E();for(let I in i){let L=i[I];for(let V in L){let q=L[V];for(let N in q){let Y=q[N];for(let X in Y)h(Y[X].object),delete Y[X];delete q[N]}}delete i[I]}}function S(I){if(i[I.id]===void 0)return;let L=i[I.id];for(let V in L){let q=L[V];for(let N in q){let Y=q[N];for(let X in Y)h(Y[X].object),delete Y[X];delete q[N]}}delete i[I.id]}function A(I){for(let L in i){let V=i[L];for(let q in V){let N=V[q];if(N[I.id]===void 0)continue;let Y=N[I.id];for(let X in Y)h(Y[X].object),delete Y[X];delete N[I.id]}}}function _(I){for(let L in i){let V=i[L],q=I.isInstancedMesh===!0?I.id:0,N=V[q];if(N!==void 0){for(let Y in N){let X=N[Y];for(let ne in X)h(X[ne].object),delete X[ne];delete N[Y]}delete V[q],Object.keys(V).length===0&&delete i[L]}}}function E(){C(),o=!0,r!==s&&(r=s,l(r.object))}function C(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:E,resetDefaultState:C,dispose:T,releaseStatesOfGeometry:S,releaseStatesOfObject:_,releaseStatesOfProgram:A,initAttributes:x,enableAttribute:p,disableUnusedAttributes:M}}function av(n,e,t){let i;function s(c){i=c}function r(c,l){n.drawArrays(i,c,l),t.update(l,i,1)}function o(c,l,h){h!==0&&(n.drawArraysInstanced(i,c,l,h),t.update(l,i,h))}function a(c,l,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,c,0,l,0,h);let d=0;for(let f=0;f<h;f++)d+=l[f];t.update(d,i,1)}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a}function lv(n,e,t,i){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){let A=e.get("EXT_texture_filter_anisotropic");s=n.getParameter(A.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(A){return!(A!==Si&&i.convert(A)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(A){let _=A===ii&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(A!==ci&&i.convert(A)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_TYPE)&&A!==Hi&&!_)}function c(A){if(A==="highp"){if(n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.HIGH_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.HIGH_FLOAT).precision>0)return"highp";A="mediump"}return A==="mediump"&&n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.MEDIUM_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=t.precision!==void 0?t.precision:"highp",h=c(l);h!==l&&($e("WebGLRenderer:",l,"not supported, using",h,"instead."),l=h);let u=t.logarithmicDepthBuffer===!0,d=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control");t.reversedDepthBuffer===!0&&d===!1&&$e("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.");let f=n.getParameter(n.MAX_TEXTURE_IMAGE_UNITS),g=n.getParameter(n.MAX_VERTEX_TEXTURE_IMAGE_UNITS),x=n.getParameter(n.MAX_TEXTURE_SIZE),p=n.getParameter(n.MAX_CUBE_MAP_TEXTURE_SIZE),m=n.getParameter(n.MAX_VERTEX_ATTRIBS),M=n.getParameter(n.MAX_VERTEX_UNIFORM_VECTORS),b=n.getParameter(n.MAX_VARYING_VECTORS),y=n.getParameter(n.MAX_FRAGMENT_UNIFORM_VECTORS),T=n.getParameter(n.MAX_SAMPLES),S=n.getParameter(n.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:c,textureFormatReadable:o,textureTypeReadable:a,precision:l,logarithmicDepthBuffer:u,reversedDepthBuffer:d,maxTextures:f,maxVertexTextures:g,maxTextureSize:x,maxCubemapSize:p,maxAttributes:m,maxVertexUniforms:M,maxVaryings:b,maxFragmentUniforms:y,maxSamples:T,samples:S}}function cv(n){let e=this,t=null,i=0,s=!1,r=!1,o=new Bi,a=new tt,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(u,d){let f=u.length!==0||d||i!==0||s;return s=d,i=u.length,f},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(u,d){t=h(u,d,0)},this.setState=function(u,d,f){let g=u.clippingPlanes,x=u.clipIntersection,p=u.clipShadows,m=n.get(u);if(!s||g===null||g.length===0||r&&!p)r?h(null):l();else{let M=r?0:i,b=M*4,y=m.clippingState||null;c.value=y,y=h(g,d,b,f);for(let T=0;T!==b;++T)y[T]=t[T];m.clippingState=y,this.numIntersection=x?this.numPlanes:0,this.numPlanes+=M}};function l(){c.value!==t&&(c.value=t,c.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function h(u,d,f,g){let x=u!==null?u.length:0,p=null;if(x!==0){if(p=c.value,g!==!0||p===null){let m=f+x*4,M=d.matrixWorldInverse;a.getNormalMatrix(M),(p===null||p.length<m)&&(p=new Float32Array(m));for(let b=0,y=f;b!==x;++b,y+=4)o.copy(u[b]).applyMatrix4(M,a),o.normal.toArray(p,y),p[y+3]=o.constant}c.value=p,c.needsUpdate=!0}return e.numPlanes=x,e.numIntersection=0,p}}var gs=4,up=[.125,.215,.35,.446,.526,.582],zs=20,hv=256,la=new as,dp=new Te,Ou=null,Bu=0,zu=0,ku=!1,uv=new P,Fr=class{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,i=.1,s=100,r={}){let{size:o=256,position:a=uv}=r;Ou=this._renderer.getRenderTarget(),Bu=this._renderer.getActiveCubeFace(),zu=this._renderer.getActiveMipmapLevel(),ku=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);let c=this._allocateTargets();return c.depthBuffer=!0,this._sceneToCubeUV(e,i,s,c,a),t>0&&this._blur(c,0,0,t),this._applyPMREM(c),this._cleanup(c),c}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=mp(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=pp(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Ou,Bu,zu),this._renderer.xr.enabled=ku,e.scissorTest=!1,Ur(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===ds||e.mapping===Os?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Ou=this._renderer.getRenderTarget(),Bu=this._renderer.getActiveCubeFace(),zu=this._renderer.getActiveMipmapLevel(),ku=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;let i=t||this._allocateTargets();return this._textureToCubeUV(e,i),this._applyPMREM(i),this._cleanup(i),i}_allocateTargets(){let e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,i={magFilter:ei,minFilter:ei,generateMipmaps:!1,type:ii,format:Si,colorSpace:ao,depthBuffer:!1},s=fp(e,t,i);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=fp(e,t,i);let{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=dv(r)),this._blurMaterial=pv(r,e,t),this._ggxMaterial=fv(r,e,t)}return s}_compileMaterial(e){let t=new Ke(new ut,e);this._renderer.compile(t,la)}_sceneToCubeUV(e,t,i,s,r){let c=new Qt(90,1,t,i),l=[1,-1,1,1,1,1],h=[1,1,1,-1,-1,-1],u=this._renderer,d=u.autoClear,f=u.toneMapping;u.getClearColor(dp),u.toneMapping=en,u.autoClear=!1,u.state.buffers.depth.getReversed()&&(u.setRenderTarget(s),u.clearDepth(),u.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Ke(new Bt,new Ln({name:"PMREM.Background",side:ti,depthWrite:!1,depthTest:!1})));let x=this._backgroundBox,p=x.material,m=!1,M=e.background;M?M.isColor&&(p.color.copy(M),e.background=null,m=!0):(p.color.copy(dp),m=!0);for(let b=0;b<6;b++){let y=b%3;y===0?(c.up.set(0,l[b],0),c.position.set(r.x,r.y,r.z),c.lookAt(r.x+h[b],r.y,r.z)):y===1?(c.up.set(0,0,l[b]),c.position.set(r.x,r.y,r.z),c.lookAt(r.x,r.y+h[b],r.z)):(c.up.set(0,l[b],0),c.position.set(r.x,r.y,r.z),c.lookAt(r.x,r.y,r.z+h[b]));let T=this._cubeSize;Ur(s,y*T,b>2?T:0,T,T),u.setRenderTarget(s),m&&u.render(x,c),u.render(e,c)}u.toneMapping=f,u.autoClear=d,e.background=M}_textureToCubeUV(e,t){let i=this._renderer,s=e.mapping===ds||e.mapping===Os;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=mp()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=pp());let r=s?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=r;let a=r.uniforms;a.envMap.value=e;let c=this._cubeSize;Ur(t,0,0,3*c,2*c),i.setRenderTarget(t),i.render(o,la)}_applyPMREM(e){let t=this._renderer,i=t.autoClear;t.autoClear=!1;let s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=i}_applyGGXFilter(e,t,i){let s=this._renderer,r=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[i];a.material=o;let c=o.uniforms,l=i/(this._lodMeshes.length-1),h=t/(this._lodMeshes.length-1),u=Math.sqrt(l*l-h*h),d=0+l*1.25,f=u*d,{_lodMax:g}=this,x=this._sizeLods[i],p=3*x*(i>g-gs?i-g+gs:0),m=4*(this._cubeSize-x);c.envMap.value=e.texture,c.roughness.value=f,c.mipInt.value=g-t,Ur(r,p,m,3*x,2*x),s.setRenderTarget(r),s.render(a,la),c.envMap.value=r.texture,c.roughness.value=0,c.mipInt.value=g-i,Ur(e,p,m,3*x,2*x),s.setRenderTarget(e),s.render(a,la)}_blur(e,t,i,s,r){let o=this._pingPongRenderTarget;this._halfBlur(e,o,t,i,s,"latitudinal",r),this._halfBlur(o,e,i,i,s,"longitudinal",r)}_halfBlur(e,t,i,s,r,o,a){let c=this._renderer,l=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&Ze("blur direction must be either latitudinal or longitudinal!");let h=3,u=this._lodMeshes[s];u.material=l;let d=l.uniforms,f=this._sizeLods[i]-1,g=isFinite(r)?Math.PI/(2*f):2*Math.PI/(2*zs-1),x=r/g,p=isFinite(r)?1+Math.floor(h*x):zs;p>zs&&$e(`sigmaRadians, ${r}, is too large and will clip, as it requested ${p} samples when the maximum is set to ${zs}`);let m=[],M=0;for(let A=0;A<zs;++A){let _=A/x,E=Math.exp(-_*_/2);m.push(E),A===0?M+=E:A<p&&(M+=2*E)}for(let A=0;A<m.length;A++)m[A]=m[A]/M;d.envMap.value=e.texture,d.samples.value=p,d.weights.value=m,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);let{_lodMax:b}=this;d.dTheta.value=g,d.mipInt.value=b-i;let y=this._sizeLods[s],T=3*y*(s>b-gs?s-b+gs:0),S=4*(this._cubeSize-y);Ur(t,T,S,3*y,2*y),c.setRenderTarget(t),c.render(u,la)}};function dv(n){let e=[],t=[],i=[],s=n,r=n-gs+1+up.length;for(let o=0;o<r;o++){let a=Math.pow(2,s);e.push(a);let c=1/a;o>n-gs?c=up[o-n+gs-1]:o===0&&(c=0),t.push(c);let l=1/(a-2),h=-l,u=1+l,d=[h,h,u,h,u,u,h,h,u,u,h,u],f=6,g=6,x=3,p=2,m=1,M=new Float32Array(x*g*f),b=new Float32Array(p*g*f),y=new Float32Array(m*g*f);for(let S=0;S<f;S++){let A=S%3*2/3-1,_=S>2?0:-1,E=[A,_,0,A+2/3,_,0,A+2/3,_+1,0,A,_,0,A+2/3,_+1,0,A,_+1,0];M.set(E,x*g*S),b.set(d,p*g*S);let C=[S,S,S,S,S,S];y.set(C,m*g*S)}let T=new ut;T.setAttribute("position",new Ut(M,x)),T.setAttribute("uv",new Ut(b,p)),T.setAttribute("faceIndex",new Ut(y,m)),i.push(new Ke(T,null)),s>gs&&s--}return{lodMeshes:i,sizeLods:e,sigmas:t}}function fp(n,e,t){let i=new Ht(n,e,t);return i.texture.mapping=ea,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function Ur(n,e,t,i,s){n.viewport.set(e,t,i,s),n.scissor.set(e,t,i,s)}function fv(n,e,t){return new bt({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:hv,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${n}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Gc(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 4.1: Orthonormal basis
				vec3 T1 = vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(V, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + V.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:zt,depthTest:!1,depthWrite:!1})}function pv(n,e,t){let i=new Float32Array(zs),s=new P(0,1,0);return new bt({name:"SphericalGaussianBlur",defines:{n:zs,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${n}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:i},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Gc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:zt,depthTest:!1,depthWrite:!1})}function pp(){return new bt({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Gc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:zt,depthTest:!1,depthWrite:!1})}function mp(){return new bt({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Gc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:zt,depthTest:!1,depthWrite:!1})}function Gc(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}var Hc=class extends Ht{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;let i={width:e,height:e,depth:1},s=[i,i,i,i,i,i];this.texture=new yo(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;let i={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new Bt(5,5,5),r=new bt({name:"CubemapFromEquirect",uniforms:Bs(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:ti,blending:zt});r.uniforms.tEquirect.value=t;let o=new Ke(s,r),a=t.minFilter;return t.minFilter===fs&&(t.minFilter=ei),new ql(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,i=!0,s=!0){let r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,i,s);e.setRenderTarget(r)}};function mv(n){let e=new WeakMap,t=new WeakMap,i=null;function s(d,f=!1){return d==null?null:f?o(d):r(d)}function r(d){if(d&&d.isTexture){let f=d.mapping;if(f===jl||f===Kl)if(e.has(d)){let g=e.get(d).texture;return a(g,d.mapping)}else{let g=d.image;if(g&&g.height>0){let x=new Hc(g.height);return x.fromEquirectangularTexture(n,d),e.set(d,x),d.addEventListener("dispose",l),a(x.texture,d.mapping)}else return null}}return d}function o(d){if(d&&d.isTexture){let f=d.mapping,g=f===jl||f===Kl,x=f===ds||f===Os;if(g||x){let p=t.get(d),m=p!==void 0?p.texture.pmremVersion:0;if(d.isRenderTargetTexture&&d.pmremVersion!==m)return i===null&&(i=new Fr(n)),p=g?i.fromEquirectangular(d,p):i.fromCubemap(d,p),p.texture.pmremVersion=d.pmremVersion,t.set(d,p),p.texture;if(p!==void 0)return p.texture;{let M=d.image;return g&&M&&M.height>0||x&&M&&c(M)?(i===null&&(i=new Fr(n)),p=g?i.fromEquirectangular(d):i.fromCubemap(d),p.texture.pmremVersion=d.pmremVersion,t.set(d,p),d.addEventListener("dispose",h),p.texture):null}}}return d}function a(d,f){return f===jl?d.mapping=ds:f===Kl&&(d.mapping=Os),d}function c(d){let f=0,g=6;for(let x=0;x<g;x++)d[x]!==void 0&&f++;return f===g}function l(d){let f=d.target;f.removeEventListener("dispose",l);let g=e.get(f);g!==void 0&&(e.delete(f),g.dispose())}function h(d){let f=d.target;f.removeEventListener("dispose",h);let g=t.get(f);g!==void 0&&(t.delete(f),g.dispose())}function u(){e=new WeakMap,t=new WeakMap,i!==null&&(i.dispose(),i=null)}return{get:s,dispose:u}}function gv(n){let e={};function t(i){if(e[i]!==void 0)return e[i];let s=n.getExtension(i);return e[i]=s,s}return{has:function(i){return t(i)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(i){let s=t(i);return s===null&&Cs("WebGLRenderer: "+i+" extension not supported."),s}}}function _v(n,e,t,i){let s={},r=new WeakMap;function o(u){let d=u.target;d.index!==null&&e.remove(d.index);for(let g in d.attributes)e.remove(d.attributes[g]);d.removeEventListener("dispose",o),delete s[d.id];let f=r.get(d);f&&(e.remove(f),r.delete(d)),i.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,t.memory.geometries--}function a(u,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,t.memory.geometries++),d}function c(u){let d=u.attributes;for(let f in d)e.update(d[f],n.ARRAY_BUFFER)}function l(u){let d=[],f=u.index,g=u.attributes.position,x=0;if(g===void 0)return;if(f!==null){let M=f.array;x=f.version;for(let b=0,y=M.length;b<y;b+=3){let T=M[b+0],S=M[b+1],A=M[b+2];d.push(T,S,S,A,A,T)}}else{let M=g.array;x=g.version;for(let b=0,y=M.length/3-1;b<y;b+=3){let T=b+0,S=b+1,A=b+2;d.push(T,S,S,A,A,T)}}let p=new(g.count>=65535?mo:po)(d,1);p.version=x;let m=r.get(u);m&&e.remove(m),r.set(u,p)}function h(u){let d=r.get(u);if(d){let f=u.index;f!==null&&d.version<f.version&&l(u)}else l(u);return r.get(u)}return{get:a,update:c,getWireframeAttribute:h}}function xv(n,e,t){let i;function s(u){i=u}let r,o;function a(u){r=u.type,o=u.bytesPerElement}function c(u,d){n.drawElements(i,d,r,u*o),t.update(d,i,1)}function l(u,d,f){f!==0&&(n.drawElementsInstanced(i,d,r,u*o,f),t.update(d,i,f))}function h(u,d,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,d,0,r,u,0,f);let x=0;for(let p=0;p<f;p++)x+=d[p];t.update(x,i,1)}this.setMode=s,this.setIndex=a,this.render=c,this.renderInstances=l,this.renderMultiDraw=h}function vv(n){let e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function i(r,o,a){switch(t.calls++,o){case n.TRIANGLES:t.triangles+=a*(r/3);break;case n.LINES:t.lines+=a*(r/2);break;case n.LINE_STRIP:t.lines+=a*(r-1);break;case n.LINE_LOOP:t.lines+=a*r;break;case n.POINTS:t.points+=a*r;break;default:Ze("WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:i}}function yv(n,e,t){let i=new WeakMap,s=new mt;function r(o,a,c){let l=o.morphTargetInfluences,h=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,u=h!==void 0?h.length:0,d=i.get(a);if(d===void 0||d.count!==u){let E=function(){A.dispose(),i.delete(a),a.removeEventListener("dispose",E)};d!==void 0&&d.texture.dispose();let f=a.morphAttributes.position!==void 0,g=a.morphAttributes.normal!==void 0,x=a.morphAttributes.color!==void 0,p=a.morphAttributes.position||[],m=a.morphAttributes.normal||[],M=a.morphAttributes.color||[],b=0;f===!0&&(b=1),g===!0&&(b=2),x===!0&&(b=3);let y=a.attributes.position.count*b,T=1;y>e.maxTextureSize&&(T=Math.ceil(y/e.maxTextureSize),y=e.maxTextureSize);let S=new Float32Array(y*T*4*u),A=new uo(S,y,T,u);A.type=Hi,A.needsUpdate=!0;let _=b*4;for(let C=0;C<u;C++){let I=p[C],L=m[C],V=M[C],q=y*T*4*C;for(let N=0;N<I.count;N++){let Y=N*_;f===!0&&(s.fromBufferAttribute(I,N),S[q+Y+0]=s.x,S[q+Y+1]=s.y,S[q+Y+2]=s.z,S[q+Y+3]=0),g===!0&&(s.fromBufferAttribute(L,N),S[q+Y+4]=s.x,S[q+Y+5]=s.y,S[q+Y+6]=s.z,S[q+Y+7]=0),x===!0&&(s.fromBufferAttribute(V,N),S[q+Y+8]=s.x,S[q+Y+9]=s.y,S[q+Y+10]=s.z,S[q+Y+11]=V.itemSize===4?s.w:1)}}d={count:u,texture:A,size:new $(y,T)},i.set(a,d),a.addEventListener("dispose",E)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)c.getUniforms().setValue(n,"morphTexture",o.morphTexture,t);else{let f=0;for(let x=0;x<l.length;x++)f+=l[x];let g=a.morphTargetsRelative?1:1-f;c.getUniforms().setValue(n,"morphTargetBaseInfluence",g),c.getUniforms().setValue(n,"morphTargetInfluences",l)}c.getUniforms().setValue(n,"morphTargetsTexture",d.texture,t),c.getUniforms().setValue(n,"morphTargetsTextureSize",d.size)}return{update:r}}function Mv(n,e,t,i,s){let r=new WeakMap;function o(l){let h=s.render.frame,u=l.geometry,d=e.get(l,u);if(r.get(d)!==h&&(e.update(d),r.set(d,h)),l.isInstancedMesh&&(l.hasEventListener("dispose",c)===!1&&l.addEventListener("dispose",c),r.get(l)!==h&&(t.update(l.instanceMatrix,n.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,n.ARRAY_BUFFER),r.set(l,h))),l.isSkinnedMesh){let f=l.skeleton;r.get(f)!==h&&(f.update(),r.set(f,h))}return d}function a(){r=new WeakMap}function c(l){let h=l.target;h.removeEventListener("dispose",c),i.releaseStatesOfObject(h),t.remove(h.instanceMatrix),h.instanceColor!==null&&t.remove(h.instanceColor)}return{update:o,dispose:a}}var Sv={[Zo]:"LINEAR_TONE_MAPPING",[Jo]:"REINHARD_TONE_MAPPING",[jo]:"CINEON_TONE_MAPPING",[us]:"ACES_FILMIC_TONE_MAPPING",[Qo]:"AGX_TONE_MAPPING",[Fs]:"NEUTRAL_TONE_MAPPING",[Ko]:"CUSTOM_TONE_MAPPING"};function bv(n,e,t,i,s,r){let o=new Ht(e,t,{type:n,depthBuffer:s,stencilBuffer:r,samples:i?4:0,depthTexture:s?new ji(e,t):void 0}),a=new Ht(e,t,{type:ii,depthBuffer:!1,stencilBuffer:!1}),c=new ut;c.setAttribute("position",new it([-1,3,0,-1,-1,0,3,-1,0],3)),c.setAttribute("uv",new it([0,2,0,0,2,0],2));let l=new Ar({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),h=new Ke(c,l),u=new as(-1,1,1,-1,0,1),d=null,f=null,g=!1,x,p=null,m=[],M=!1;this.setSize=function(b,y){o.setSize(b,y),a.setSize(b,y);for(let T=0;T<m.length;T++){let S=m[T];S.setSize&&S.setSize(b,y)}},this.setEffects=function(b){m=b,M=m.length>0&&m[0].isRenderPass===!0;let y=o.width,T=o.height;for(let S=0;S<m.length;S++){let A=m[S];A.setSize&&A.setSize(y,T)}},this.begin=function(b,y){if(g||b.toneMapping===en&&m.length===0)return!1;if(p=y,y!==null){let T=y.width,S=y.height;(o.width!==T||o.height!==S)&&this.setSize(T,S)}return M===!1&&b.setRenderTarget(o),x=b.toneMapping,b.toneMapping=en,!0},this.hasRenderPass=function(){return M},this.end=function(b,y){b.toneMapping=x,g=!0;let T=o,S=a;for(let A=0;A<m.length;A++){let _=m[A];if(_.enabled!==!1&&(_.render(b,S,T,y),_.needsSwap!==!1)){let E=T;T=S,S=E}}if(d!==b.outputColorSpace||f!==b.toneMapping){d=b.outputColorSpace,f=b.toneMapping,l.defines={},ht.getTransfer(d)===pt&&(l.defines.SRGB_TRANSFER="");let A=Sv[f];A&&(l.defines[A]=""),l.needsUpdate=!0}l.uniforms.tDiffuse.value=T.texture,b.setRenderTarget(p),b.render(h,u),p=null,g=!1},this.isCompositing=function(){return g},this.dispose=function(){o.depthTexture&&o.depthTexture.dispose(),o.dispose(),a.dispose(),c.dispose(),l.dispose()}}var Np=new fi,Gu=new ji(1,1),Fp=new uo,Op=new Ml,Bp=new yo,gp=[],_p=[],xp=new Float32Array(16),vp=new Float32Array(9),yp=new Float32Array(4);function Or(n,e,t){let i=n[0];if(i<=0||i>0)return n;let s=e*t,r=gp[s];if(r===void 0&&(r=new Float32Array(s),gp[s]=r),e!==0){i.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,n[o].toArray(r,a)}return r}function Wt(n,e){if(n.length!==e.length)return!1;for(let t=0,i=n.length;t<i;t++)if(n[t]!==e[t])return!1;return!0}function Xt(n,e){for(let t=0,i=e.length;t<i;t++)n[t]=e[t]}function Wc(n,e){let t=_p[e];t===void 0&&(t=new Int32Array(e),_p[e]=t);for(let i=0;i!==e;++i)t[i]=n.allocateTextureUnit();return t}function Ev(n,e){let t=this.cache;t[0]!==e&&(n.uniform1f(this.addr,e),t[0]=e)}function wv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Wt(t,e))return;n.uniform2fv(this.addr,e),Xt(t,e)}}function Tv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(n.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Wt(t,e))return;n.uniform3fv(this.addr,e),Xt(t,e)}}function Av(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Wt(t,e))return;n.uniform4fv(this.addr,e),Xt(t,e)}}function Rv(n,e){let t=this.cache,i=e.elements;if(i===void 0){if(Wt(t,e))return;n.uniformMatrix2fv(this.addr,!1,e),Xt(t,e)}else{if(Wt(t,i))return;yp.set(i),n.uniformMatrix2fv(this.addr,!1,yp),Xt(t,i)}}function Cv(n,e){let t=this.cache,i=e.elements;if(i===void 0){if(Wt(t,e))return;n.uniformMatrix3fv(this.addr,!1,e),Xt(t,e)}else{if(Wt(t,i))return;vp.set(i),n.uniformMatrix3fv(this.addr,!1,vp),Xt(t,i)}}function Pv(n,e){let t=this.cache,i=e.elements;if(i===void 0){if(Wt(t,e))return;n.uniformMatrix4fv(this.addr,!1,e),Xt(t,e)}else{if(Wt(t,i))return;xp.set(i),n.uniformMatrix4fv(this.addr,!1,xp),Xt(t,i)}}function Iv(n,e){let t=this.cache;t[0]!==e&&(n.uniform1i(this.addr,e),t[0]=e)}function Dv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Wt(t,e))return;n.uniform2iv(this.addr,e),Xt(t,e)}}function Lv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Wt(t,e))return;n.uniform3iv(this.addr,e),Xt(t,e)}}function Uv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Wt(t,e))return;n.uniform4iv(this.addr,e),Xt(t,e)}}function Nv(n,e){let t=this.cache;t[0]!==e&&(n.uniform1ui(this.addr,e),t[0]=e)}function Fv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Wt(t,e))return;n.uniform2uiv(this.addr,e),Xt(t,e)}}function Ov(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Wt(t,e))return;n.uniform3uiv(this.addr,e),Xt(t,e)}}function Bv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Wt(t,e))return;n.uniform4uiv(this.addr,e),Xt(t,e)}}function zv(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s);let r;this.type===n.SAMPLER_2D_SHADOW?(Gu.compareFunction=t.isReversedDepthBuffer()?Bc:Oc,r=Gu):r=Np,t.setTexture2D(e||r,s)}function kv(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTexture3D(e||Op,s)}function Hv(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTextureCube(e||Bp,s)}function Vv(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTexture2DArray(e||Fp,s)}function Gv(n){switch(n){case 5126:return Ev;case 35664:return wv;case 35665:return Tv;case 35666:return Av;case 35674:return Rv;case 35675:return Cv;case 35676:return Pv;case 5124:case 35670:return Iv;case 35667:case 35671:return Dv;case 35668:case 35672:return Lv;case 35669:case 35673:return Uv;case 5125:return Nv;case 36294:return Fv;case 36295:return Ov;case 36296:return Bv;case 35678:case 36198:case 36298:case 36306:case 35682:return zv;case 35679:case 36299:case 36307:return kv;case 35680:case 36300:case 36308:case 36293:return Hv;case 36289:case 36303:case 36311:case 36292:return Vv}}function Wv(n,e){n.uniform1fv(this.addr,e)}function Xv(n,e){let t=Or(e,this.size,2);n.uniform2fv(this.addr,t)}function qv(n,e){let t=Or(e,this.size,3);n.uniform3fv(this.addr,t)}function Yv(n,e){let t=Or(e,this.size,4);n.uniform4fv(this.addr,t)}function $v(n,e){let t=Or(e,this.size,4);n.uniformMatrix2fv(this.addr,!1,t)}function Zv(n,e){let t=Or(e,this.size,9);n.uniformMatrix3fv(this.addr,!1,t)}function Jv(n,e){let t=Or(e,this.size,16);n.uniformMatrix4fv(this.addr,!1,t)}function jv(n,e){n.uniform1iv(this.addr,e)}function Kv(n,e){n.uniform2iv(this.addr,e)}function Qv(n,e){n.uniform3iv(this.addr,e)}function ey(n,e){n.uniform4iv(this.addr,e)}function ty(n,e){n.uniform1uiv(this.addr,e)}function iy(n,e){n.uniform2uiv(this.addr,e)}function ny(n,e){n.uniform3uiv(this.addr,e)}function sy(n,e){n.uniform4uiv(this.addr,e)}function ry(n,e,t){let i=this.cache,s=e.length,r=Wc(t,s);Wt(i,r)||(n.uniform1iv(this.addr,r),Xt(i,r));let o;this.type===n.SAMPLER_2D_SHADOW?o=Gu:o=Np;for(let a=0;a!==s;++a)t.setTexture2D(e[a]||o,r[a])}function oy(n,e,t){let i=this.cache,s=e.length,r=Wc(t,s);Wt(i,r)||(n.uniform1iv(this.addr,r),Xt(i,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||Op,r[o])}function ay(n,e,t){let i=this.cache,s=e.length,r=Wc(t,s);Wt(i,r)||(n.uniform1iv(this.addr,r),Xt(i,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||Bp,r[o])}function ly(n,e,t){let i=this.cache,s=e.length,r=Wc(t,s);Wt(i,r)||(n.uniform1iv(this.addr,r),Xt(i,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||Fp,r[o])}function cy(n){switch(n){case 5126:return Wv;case 35664:return Xv;case 35665:return qv;case 35666:return Yv;case 35674:return $v;case 35675:return Zv;case 35676:return Jv;case 5124:case 35670:return jv;case 35667:case 35671:return Kv;case 35668:case 35672:return Qv;case 35669:case 35673:return ey;case 5125:return ty;case 36294:return iy;case 36295:return ny;case 36296:return sy;case 35678:case 36198:case 36298:case 36306:case 35682:return ry;case 35679:case 36299:case 36307:return oy;case 35680:case 36300:case 36308:case 36293:return ay;case 36289:case 36303:case 36311:case 36292:return ly}}var Wu=class{constructor(e,t,i){this.id=e,this.addr=i,this.cache=[],this.type=t.type,this.setValue=Gv(t.type)}},Xu=class{constructor(e,t,i){this.id=e,this.addr=i,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=cy(t.type)}},qu=class{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,i){let s=this.seq;for(let r=0,o=s.length;r!==o;++r){let a=s[r];a.setValue(e,t[a.id],i)}}},Hu=/(\w+)(\])?(\[|\.)?/g;function Mp(n,e){n.seq.push(e),n.map[e.id]=e}function hy(n,e,t){let i=n.name,s=i.length;for(Hu.lastIndex=0;;){let r=Hu.exec(i),o=Hu.lastIndex,a=r[1],c=r[2]==="]",l=r[3];if(c&&(a=a|0),l===void 0||l==="["&&o+2===s){Mp(t,l===void 0?new Wu(a,n,e):new Xu(a,n,e));break}else{let u=t.map[a];u===void 0&&(u=new qu(a),Mp(t,u)),t=u}}}var Nr=class{constructor(e,t){this.seq=[],this.map={};let i=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let o=0;o<i;++o){let a=e.getActiveUniform(t,o),c=e.getUniformLocation(t,a.name);hy(a,c,this)}let s=[],r=[];for(let o of this.seq)o.type===e.SAMPLER_2D_SHADOW||o.type===e.SAMPLER_CUBE_SHADOW||o.type===e.SAMPLER_2D_ARRAY_SHADOW?s.push(o):r.push(o);s.length>0&&(this.seq=s.concat(r))}setValue(e,t,i,s){let r=this.map[t];r!==void 0&&r.setValue(e,i,s)}setOptional(e,t,i){let s=t[i];s!==void 0&&this.setValue(e,i,s)}static upload(e,t,i,s){for(let r=0,o=t.length;r!==o;++r){let a=t[r],c=i[a.id];c.needsUpdate!==!1&&a.setValue(e,c.value,s)}}static seqWithValue(e,t){let i=[];for(let s=0,r=e.length;s!==r;++s){let o=e[s];o.id in t&&i.push(o)}return i}};function Sp(n,e,t){let i=n.createShader(e);return n.shaderSource(i,t),n.compileShader(i),i}var uy=37297,dy=0;function fy(n,e){let t=n.split(`
`),i=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){let a=o+1;i.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return i.join(`
`)}var bp=new tt;function py(n){ht._getMatrix(bp,ht.workingColorSpace,n);let e=`mat3( ${bp.elements.map(t=>t.toFixed(4))} )`;switch(ht.getTransfer(n)){case lo:return[e,"LinearTransferOETF"];case pt:return[e,"sRGBTransferOETF"];default:return $e("WebGLProgram: Unsupported color space: ",n),[e,"LinearTransferOETF"]}}function Ep(n,e,t){let i=n.getShaderParameter(e,n.COMPILE_STATUS),r=(n.getShaderInfoLog(e)||"").trim();if(i&&r==="")return"";let o=/ERROR: 0:(\d+)/.exec(r);if(o){let a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+fy(n.getShaderSource(e),a)}else return r}function my(n,e){let t=py(e);return[`vec4 ${n}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}var gy={[Zo]:"Linear",[Jo]:"Reinhard",[jo]:"Cineon",[us]:"ACESFilmic",[Qo]:"AgX",[Fs]:"Neutral",[Ko]:"Custom"};function _y(n,e){let t=gy[e];return t===void 0?($e("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+n+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+n+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}var kc=new P;function xy(){ht.getLuminanceCoefficients(kc);let n=kc.x.toFixed(4),e=kc.y.toFixed(4),t=kc.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${n}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function vy(n){return[n.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",n.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(ha).join(`
`)}function yy(n){let e=[];for(let t in n){let i=n[t];i!==!1&&e.push("#define "+t+" "+i)}return e.join(`
`)}function My(n,e){let t={},i=n.getProgramParameter(e,n.ACTIVE_ATTRIBUTES);for(let s=0;s<i;s++){let r=n.getActiveAttrib(e,s),o=r.name,a=1;r.type===n.FLOAT_MAT2&&(a=2),r.type===n.FLOAT_MAT3&&(a=3),r.type===n.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:n.getAttribLocation(e,o),locationSize:a}}return t}function ha(n){return n!==""}function wp(n,e){let t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return n.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function Tp(n,e){return n.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}var Sy=/^[ \t]*#include +<([\w\d./]+)>/gm;function Yu(n){return n.replace(Sy,Ey)}var by=new Map;function Ey(n,e){let t=at[e];if(t===void 0){let i=by.get(e);if(i!==void 0)t=at[i],$e('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,i);else throw new Error("THREE.WebGLProgram: Can not resolve #include <"+e+">")}return Yu(t)}var wy=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Ap(n){return n.replace(wy,Ty)}function Ty(n,e,t,i){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=i.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Rp(n){let e=`precision ${n.precision} float;
	precision ${n.precision} int;
	precision ${n.precision} sampler2D;
	precision ${n.precision} samplerCube;
	precision ${n.precision} sampler3D;
	precision ${n.precision} sampler2DArray;
	precision ${n.precision} sampler2DShadow;
	precision ${n.precision} samplerCubeShadow;
	precision ${n.precision} sampler2DArrayShadow;
	precision ${n.precision} isampler2D;
	precision ${n.precision} isampler3D;
	precision ${n.precision} isamplerCube;
	precision ${n.precision} isampler2DArray;
	precision ${n.precision} usampler2D;
	precision ${n.precision} usampler3D;
	precision ${n.precision} usamplerCube;
	precision ${n.precision} usampler2DArray;
	`;return n.precision==="highp"?e+=`
#define HIGH_PRECISION`:n.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:n.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}var Ay={[Us]:"SHADOWMAP_TYPE_PCF",[Ir]:"SHADOWMAP_TYPE_VSM"};function Ry(n){return Ay[n.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}var Cy={[ds]:"ENVMAP_TYPE_CUBE",[Os]:"ENVMAP_TYPE_CUBE",[ea]:"ENVMAP_TYPE_CUBE_UV"};function Py(n){return n.envMap===!1?"ENVMAP_TYPE_CUBE":Cy[n.envMapMode]||"ENVMAP_TYPE_CUBE"}var Iy={[Os]:"ENVMAP_MODE_REFRACTION"};function Dy(n){return n.envMap===!1?"ENVMAP_MODE_REFLECTION":Iy[n.envMapMode]||"ENVMAP_MODE_REFLECTION"}var Ly={[Jl]:"ENVMAP_BLENDING_MULTIPLY",[Gf]:"ENVMAP_BLENDING_MIX",[Wf]:"ENVMAP_BLENDING_ADD"};function Uy(n){return n.envMap===!1?"ENVMAP_BLENDING_NONE":Ly[n.combine]||"ENVMAP_BLENDING_NONE"}function Ny(n){let e=n.envMapCubeUVHeight;if(e===null)return null;let t=Math.log2(e)-2,i=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:i,maxMip:t}}function Fy(n,e,t,i){let s=n.getContext(),r=t.defines,o=t.vertexShader,a=t.fragmentShader,c=Ry(t),l=Py(t),h=Dy(t),u=Uy(t),d=Ny(t),f=vy(t),g=yy(r),x=s.createProgram(),p,m,M=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(p=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(ha).join(`
`),p.length>0&&(p+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(ha).join(`
`),m.length>0&&(m+=`
`)):(p=[Rp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+h:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexNormals?"#define HAS_NORMAL":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(ha).join(`
`),m=[Rp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+l:"",t.envMap?"#define "+h:"",t.envMap?"#define "+u:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor?"#define USE_COLOR":"",t.vertexAlphas||t.batchingColor?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.numLightProbeGrids>0?"#define USE_LIGHT_PROBES_GRID":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==en?"#define TONE_MAPPING":"",t.toneMapping!==en?at.tonemapping_pars_fragment:"",t.toneMapping!==en?_y("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",at.colorspace_pars_fragment,my("linearToOutputTexel",t.outputColorSpace),xy(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(ha).join(`
`)),o=Yu(o),o=wp(o,t),o=Tp(o,t),a=Yu(a),a=wp(a,t),a=Tp(a,t),o=Ap(o),a=Ap(a),t.isRawShaderMaterial!==!0&&(M=`#version 300 es
`,p=[f,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+p,m=["#define varying in",t.glslVersion===wu?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===wu?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);let b=M+p+o,y=M+m+a,T=Sp(s,s.VERTEX_SHADER,b),S=Sp(s,s.FRAGMENT_SHADER,y);s.attachShader(x,T),s.attachShader(x,S),t.index0AttributeName!==void 0?s.bindAttribLocation(x,0,t.index0AttributeName):t.hasPositionAttribute===!0&&s.bindAttribLocation(x,0,"position"),s.linkProgram(x);function A(I){if(n.debug.checkShaderErrors){let L=s.getProgramInfoLog(x)||"",V=s.getShaderInfoLog(T)||"",q=s.getShaderInfoLog(S)||"",N=L.trim(),Y=V.trim(),X=q.trim(),ne=!0,ie=!0;if(s.getProgramParameter(x,s.LINK_STATUS)===!1)if(ne=!1,typeof n.debug.onShaderError=="function")n.debug.onShaderError(s,x,T,S);else{let ge=Ep(s,T,"vertex"),ue=Ep(s,S,"fragment");Ze("WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(x,s.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+N+`
`+ge+`
`+ue)}else N!==""?$e("WebGLProgram: Program Info Log:",N):(Y===""||X==="")&&(ie=!1);ie&&(I.diagnostics={runnable:ne,programLog:N,vertexShader:{log:Y,prefix:p},fragmentShader:{log:X,prefix:m}})}s.deleteShader(T),s.deleteShader(S),_=new Nr(s,x),E=My(s,x)}let _;this.getUniforms=function(){return _===void 0&&A(this),_};let E;this.getAttributes=function(){return E===void 0&&A(this),E};let C=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return C===!1&&(C=s.getProgramParameter(x,uy)),C},this.destroy=function(){i.releaseStatesOfProgram(this),s.deleteProgram(x),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=dy++,this.cacheKey=e,this.usedTimes=1,this.program=x,this.vertexShader=T,this.fragmentShader=S,this}var Oy=0,$u=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e,t,i){let s=this._getShaderCacheForMaterial(e);return s.has(t)===!1&&(s.add(t),t.usedTimes++),s.has(i)===!1&&(s.add(i),i.usedTimes++),this}remove(e){let t=this.materialCache.get(e);for(let i of t)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(e),this}getVertexShaderStage(e){return this._getShaderStage(e.vertexShader)}getFragmentShaderStage(e){return this._getShaderStage(e.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){let t=this.materialCache,i=t.get(e);return i===void 0&&(i=new Set,t.set(e,i)),i}_getShaderStage(e){let t=this.shaderCache,i=t.get(e);return i===void 0&&(i=new Zu(e),t.set(e,i)),i}},Zu=class{constructor(e){this.id=Oy++,this.code=e,this.usedTimes=0}};function By(n){return n===ms||n===oa||n===aa}function zy(n,e,t,i,s,r){let o=new yr,a=new $u,c=new Set,l=[],h=new Map,u=i.logarithmicDepthBuffer,d=i.precision,f={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function g(_){return c.add(_),_===0?"uv":`uv${_}`}function x(_,E,C,I,L,V){let q=I.fog,N=L.geometry,Y=_.isMeshStandardMaterial||_.isMeshLambertMaterial||_.isMeshPhongMaterial?I.environment:null,X=_.isMeshStandardMaterial||_.isMeshLambertMaterial&&!_.envMap||_.isMeshPhongMaterial&&!_.envMap,ne=e.get(_.envMap||Y,X),ie=ne&&ne.mapping===ea?ne.image.height:null,ge=f[_.type];_.precision!==null&&(d=i.getMaxPrecision(_.precision),d!==_.precision&&$e("WebGLProgram.getParameters:",_.precision,"not supported, using",d,"instead."));let ue=N.morphAttributes.position||N.morphAttributes.normal||N.morphAttributes.color,xe=ue!==void 0?ue.length:0,Ne=0;N.morphAttributes.position!==void 0&&(Ne=1),N.morphAttributes.normal!==void 0&&(Ne=2),N.morphAttributes.color!==void 0&&(Ne=3);let st,Xe,j,he;if(ge){let Oe=gi[ge];st=Oe.vertexShader,Xe=Oe.fragmentShader}else{st=_.vertexShader,Xe=_.fragmentShader;let Oe=a.getVertexShaderStage(_),It=a.getFragmentShaderStage(_);a.update(_,Oe,It),j=Oe.id,he=It.id}let le=n.getRenderTarget(),Ae=n.state.buffers.depth.getReversed(),Fe=L.isInstancedMesh===!0,ke=L.isBatchedMesh===!0,ae=!!_.map,ee=!!_.matcap,O=!!ne,H=!!_.aoMap,Q=!!_.lightMap,W=!!_.bumpMap&&_.wireframe===!1,G=!!_.normalMap,se=!!_.displacementMap,ce=!!_.emissiveMap,fe=!!_.metalnessMap,me=!!_.roughnessMap,D=_.anisotropy>0,Me=_.clearcoat>0,Ve=_.dispersion>0,R=_.iridescence>0,v=_.sheen>0,U=_.transmission>0,B=D&&!!_.anisotropyMap,k=Me&&!!_.clearcoatMap,pe=Me&&!!_.clearcoatNormalMap,_e=Me&&!!_.clearcoatRoughnessMap,te=R&&!!_.iridescenceMap,re=R&&!!_.iridescenceThicknessMap,Se=v&&!!_.sheenColorMap,Ie=v&&!!_.sheenRoughnessMap,ve=!!_.specularMap,ye=!!_.specularColorMap,Be=!!_.specularIntensityMap,qe=U&&!!_.transmissionMap,Je=U&&!!_.thicknessMap,F=!!_.gradientMap,Ee=!!_.alphaMap,oe=_.alphaTest>0,we=!!_.alphaHash,Pe=!!_.extensions,de=en;_.toneMapped&&(le===null||le.isXRRenderTarget===!0)&&(de=n.toneMapping);let He={shaderID:ge,shaderType:_.type,shaderName:_.name,vertexShader:st,fragmentShader:Xe,defines:_.defines,customVertexShaderID:j,customFragmentShaderID:he,isRawShaderMaterial:_.isRawShaderMaterial===!0,glslVersion:_.glslVersion,precision:d,batching:ke,batchingColor:ke&&L._colorsTexture!==null,instancing:Fe,instancingColor:Fe&&L.instanceColor!==null,instancingMorph:Fe&&L.morphTexture!==null,outputColorSpace:le===null?n.outputColorSpace:le.isXRRenderTarget===!0?le.texture.colorSpace:ht.workingColorSpace,alphaToCoverage:!!_.alphaToCoverage,map:ae,matcap:ee,envMap:O,envMapMode:O&&ne.mapping,envMapCubeUVHeight:ie,aoMap:H,lightMap:Q,bumpMap:W,normalMap:G,displacementMap:se,emissiveMap:ce,normalMapObjectSpace:G&&_.normalMapType===Yf,normalMapTangentSpace:G&&_.normalMapType===Lr,packedNormalMap:G&&_.normalMapType===Lr&&By(_.normalMap.format),metalnessMap:fe,roughnessMap:me,anisotropy:D,anisotropyMap:B,clearcoat:Me,clearcoatMap:k,clearcoatNormalMap:pe,clearcoatRoughnessMap:_e,dispersion:Ve,iridescence:R,iridescenceMap:te,iridescenceThicknessMap:re,sheen:v,sheenColorMap:Se,sheenRoughnessMap:Ie,specularMap:ve,specularColorMap:ye,specularIntensityMap:Be,transmission:U,transmissionMap:qe,thicknessMap:Je,gradientMap:F,opaque:_.transparent===!1&&_.blending===Ps&&_.alphaToCoverage===!1,alphaMap:Ee,alphaTest:oe,alphaHash:we,combine:_.combine,mapUv:ae&&g(_.map.channel),aoMapUv:H&&g(_.aoMap.channel),lightMapUv:Q&&g(_.lightMap.channel),bumpMapUv:W&&g(_.bumpMap.channel),normalMapUv:G&&g(_.normalMap.channel),displacementMapUv:se&&g(_.displacementMap.channel),emissiveMapUv:ce&&g(_.emissiveMap.channel),metalnessMapUv:fe&&g(_.metalnessMap.channel),roughnessMapUv:me&&g(_.roughnessMap.channel),anisotropyMapUv:B&&g(_.anisotropyMap.channel),clearcoatMapUv:k&&g(_.clearcoatMap.channel),clearcoatNormalMapUv:pe&&g(_.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:_e&&g(_.clearcoatRoughnessMap.channel),iridescenceMapUv:te&&g(_.iridescenceMap.channel),iridescenceThicknessMapUv:re&&g(_.iridescenceThicknessMap.channel),sheenColorMapUv:Se&&g(_.sheenColorMap.channel),sheenRoughnessMapUv:Ie&&g(_.sheenRoughnessMap.channel),specularMapUv:ve&&g(_.specularMap.channel),specularColorMapUv:ye&&g(_.specularColorMap.channel),specularIntensityMapUv:Be&&g(_.specularIntensityMap.channel),transmissionMapUv:qe&&g(_.transmissionMap.channel),thicknessMapUv:Je&&g(_.thicknessMap.channel),alphaMapUv:Ee&&g(_.alphaMap.channel),vertexTangents:!!N.attributes.tangent&&(G||D),vertexNormals:!!N.attributes.normal,vertexColors:_.vertexColors,vertexAlphas:_.vertexColors===!0&&!!N.attributes.color&&N.attributes.color.itemSize===4,pointsUvs:L.isPoints===!0&&!!N.attributes.uv&&(ae||Ee),fog:!!q,useFog:_.fog===!0,fogExp2:!!q&&q.isFogExp2,flatShading:_.wireframe===!1&&(_.flatShading===!0||N.attributes.normal===void 0&&G===!1&&(_.isMeshLambertMaterial||_.isMeshPhongMaterial||_.isMeshStandardMaterial||_.isMeshPhysicalMaterial)),sizeAttenuation:_.sizeAttenuation===!0,logarithmicDepthBuffer:u,reversedDepthBuffer:Ae,skinning:L.isSkinnedMesh===!0,hasPositionAttribute:N.attributes.position!==void 0,morphTargets:N.morphAttributes.position!==void 0,morphNormals:N.morphAttributes.normal!==void 0,morphColors:N.morphAttributes.color!==void 0,morphTargetsCount:xe,morphTextureStride:Ne,numDirLights:E.directional.length,numPointLights:E.point.length,numSpotLights:E.spot.length,numSpotLightMaps:E.spotLightMap.length,numRectAreaLights:E.rectArea.length,numHemiLights:E.hemi.length,numDirLightShadows:E.directionalShadowMap.length,numPointLightShadows:E.pointShadowMap.length,numSpotLightShadows:E.spotShadowMap.length,numSpotLightShadowsWithMaps:E.numSpotLightShadowsWithMaps,numLightProbes:E.numLightProbes,numLightProbeGrids:V.length,numClippingPlanes:r.numPlanes,numClipIntersection:r.numIntersection,dithering:_.dithering,shadowMapEnabled:n.shadowMap.enabled&&C.length>0,shadowMapType:n.shadowMap.type,toneMapping:de,decodeVideoTexture:ae&&_.map.isVideoTexture===!0&&ht.getTransfer(_.map.colorSpace)===pt,decodeVideoTextureEmissive:ce&&_.emissiveMap.isVideoTexture===!0&&ht.getTransfer(_.emissiveMap.colorSpace)===pt,premultipliedAlpha:_.premultipliedAlpha,doubleSided:_.side===Mi,flipSided:_.side===ti,useDepthPacking:_.depthPacking>=0,depthPacking:_.depthPacking||0,index0AttributeName:_.index0AttributeName,extensionClipCullDistance:Pe&&_.extensions.clipCullDistance===!0&&t.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Pe&&_.extensions.multiDraw===!0||ke)&&t.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:t.has("KHR_parallel_shader_compile"),customProgramCacheKey:_.customProgramCacheKey()};return He.vertexUv1s=c.has(1),He.vertexUv2s=c.has(2),He.vertexUv3s=c.has(3),c.clear(),He}function p(_){let E=[];if(_.shaderID?E.push(_.shaderID):(E.push(_.customVertexShaderID),E.push(_.customFragmentShaderID)),_.defines!==void 0)for(let C in _.defines)E.push(C),E.push(_.defines[C]);return _.isRawShaderMaterial===!1&&(m(E,_),M(E,_),E.push(n.outputColorSpace)),E.push(_.customProgramCacheKey),E.join()}function m(_,E){_.push(E.precision),_.push(E.outputColorSpace),_.push(E.envMapMode),_.push(E.envMapCubeUVHeight),_.push(E.mapUv),_.push(E.alphaMapUv),_.push(E.lightMapUv),_.push(E.aoMapUv),_.push(E.bumpMapUv),_.push(E.normalMapUv),_.push(E.displacementMapUv),_.push(E.emissiveMapUv),_.push(E.metalnessMapUv),_.push(E.roughnessMapUv),_.push(E.anisotropyMapUv),_.push(E.clearcoatMapUv),_.push(E.clearcoatNormalMapUv),_.push(E.clearcoatRoughnessMapUv),_.push(E.iridescenceMapUv),_.push(E.iridescenceThicknessMapUv),_.push(E.sheenColorMapUv),_.push(E.sheenRoughnessMapUv),_.push(E.specularMapUv),_.push(E.specularColorMapUv),_.push(E.specularIntensityMapUv),_.push(E.transmissionMapUv),_.push(E.thicknessMapUv),_.push(E.combine),_.push(E.fogExp2),_.push(E.sizeAttenuation),_.push(E.morphTargetsCount),_.push(E.morphAttributeCount),_.push(E.numDirLights),_.push(E.numPointLights),_.push(E.numSpotLights),_.push(E.numSpotLightMaps),_.push(E.numHemiLights),_.push(E.numRectAreaLights),_.push(E.numDirLightShadows),_.push(E.numPointLightShadows),_.push(E.numSpotLightShadows),_.push(E.numSpotLightShadowsWithMaps),_.push(E.numLightProbes),_.push(E.shadowMapType),_.push(E.toneMapping),_.push(E.numClippingPlanes),_.push(E.numClipIntersection),_.push(E.depthPacking)}function M(_,E){o.disableAll(),E.instancing&&o.enable(0),E.instancingColor&&o.enable(1),E.instancingMorph&&o.enable(2),E.matcap&&o.enable(3),E.envMap&&o.enable(4),E.normalMapObjectSpace&&o.enable(5),E.normalMapTangentSpace&&o.enable(6),E.clearcoat&&o.enable(7),E.iridescence&&o.enable(8),E.alphaTest&&o.enable(9),E.vertexColors&&o.enable(10),E.vertexAlphas&&o.enable(11),E.vertexUv1s&&o.enable(12),E.vertexUv2s&&o.enable(13),E.vertexUv3s&&o.enable(14),E.vertexTangents&&o.enable(15),E.anisotropy&&o.enable(16),E.alphaHash&&o.enable(17),E.batching&&o.enable(18),E.dispersion&&o.enable(19),E.batchingColor&&o.enable(20),E.gradientMap&&o.enable(21),E.packedNormalMap&&o.enable(22),E.vertexNormals&&o.enable(23),_.push(o.mask),o.disableAll(),E.fog&&o.enable(0),E.useFog&&o.enable(1),E.flatShading&&o.enable(2),E.logarithmicDepthBuffer&&o.enable(3),E.reversedDepthBuffer&&o.enable(4),E.skinning&&o.enable(5),E.morphTargets&&o.enable(6),E.morphNormals&&o.enable(7),E.morphColors&&o.enable(8),E.premultipliedAlpha&&o.enable(9),E.shadowMapEnabled&&o.enable(10),E.doubleSided&&o.enable(11),E.flipSided&&o.enable(12),E.useDepthPacking&&o.enable(13),E.dithering&&o.enable(14),E.transmission&&o.enable(15),E.sheen&&o.enable(16),E.opaque&&o.enable(17),E.pointsUvs&&o.enable(18),E.decodeVideoTexture&&o.enable(19),E.decodeVideoTextureEmissive&&o.enable(20),E.alphaToCoverage&&o.enable(21),E.numLightProbeGrids>0&&o.enable(22),E.hasPositionAttribute&&o.enable(23),_.push(o.mask)}function b(_){let E=f[_.type],C;if(E){let I=gi[E];C=mi.clone(I.uniforms)}else C=_.uniforms;return C}function y(_,E){let C=h.get(E);return C!==void 0?++C.usedTimes:(C=new Fy(n,E,_,s),l.push(C),h.set(E,C)),C}function T(_){if(--_.usedTimes===0){let E=l.indexOf(_);l[E]=l[l.length-1],l.pop(),h.delete(_.cacheKey),_.destroy()}}function S(_){a.remove(_)}function A(){a.dispose()}return{getParameters:x,getProgramCacheKey:p,getUniforms:b,acquireProgram:y,releaseProgram:T,releaseShaderCache:S,programs:l,dispose:A}}function ky(){let n=new WeakMap;function e(o){return n.has(o)}function t(o){let a=n.get(o);return a===void 0&&(a={},n.set(o,a)),a}function i(o){n.delete(o)}function s(o,a,c){n.get(o)[a]=c}function r(){n=new WeakMap}return{has:e,get:t,remove:i,update:s,dispose:r}}function Hy(n,e){return n.groupOrder!==e.groupOrder?n.groupOrder-e.groupOrder:n.renderOrder!==e.renderOrder?n.renderOrder-e.renderOrder:n.material.id!==e.material.id?n.material.id-e.material.id:n.materialVariant!==e.materialVariant?n.materialVariant-e.materialVariant:n.z!==e.z?n.z-e.z:n.id-e.id}function Cp(n,e){return n.groupOrder!==e.groupOrder?n.groupOrder-e.groupOrder:n.renderOrder!==e.renderOrder?n.renderOrder-e.renderOrder:n.z!==e.z?e.z-n.z:n.id-e.id}function Pp(){let n=[],e=0,t=[],i=[],s=[];function r(){e=0,t.length=0,i.length=0,s.length=0}function o(d){let f=0;return d.isInstancedMesh&&(f+=2),d.isSkinnedMesh&&(f+=1),f}function a(d,f,g,x,p,m){let M=n[e];return M===void 0?(M={id:d.id,object:d,geometry:f,material:g,materialVariant:o(d),groupOrder:x,renderOrder:d.renderOrder,z:p,group:m},n[e]=M):(M.id=d.id,M.object=d,M.geometry=f,M.material=g,M.materialVariant=o(d),M.groupOrder=x,M.renderOrder=d.renderOrder,M.z=p,M.group=m),e++,M}function c(d,f,g,x,p,m){let M=a(d,f,g,x,p,m);g.transmission>0?i.push(M):g.transparent===!0?s.push(M):t.push(M)}function l(d,f,g,x,p,m){let M=a(d,f,g,x,p,m);g.transmission>0?i.unshift(M):g.transparent===!0?s.unshift(M):t.unshift(M)}function h(d,f,g){t.length>1&&t.sort(d||Hy),i.length>1&&i.sort(f||Cp),s.length>1&&s.sort(f||Cp),g&&(t.reverse(),i.reverse(),s.reverse())}function u(){for(let d=e,f=n.length;d<f;d++){let g=n[d];if(g.id===null)break;g.id=null,g.object=null,g.geometry=null,g.material=null,g.group=null}}return{opaque:t,transmissive:i,transparent:s,init:r,push:c,unshift:l,finish:u,sort:h}}function Vy(){let n=new WeakMap;function e(i,s){let r=n.get(i),o;return r===void 0?(o=new Pp,n.set(i,[o])):s>=r.length?(o=new Pp,r.push(o)):o=r[s],o}function t(){n=new WeakMap}return{get:e,dispose:t}}function Gy(){let n={};return{get:function(e){if(n[e.id]!==void 0)return n[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new P,color:new Te};break;case"SpotLight":t={position:new P,direction:new P,color:new Te,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new P,color:new Te,distance:0,decay:0};break;case"HemisphereLight":t={direction:new P,skyColor:new Te,groundColor:new Te};break;case"RectAreaLight":t={color:new Te,position:new P,halfWidth:new P,halfHeight:new P};break}return n[e.id]=t,t}}}function Wy(){let n={};return{get:function(e){if(n[e.id]!==void 0)return n[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new $};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new $};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new $,shadowCameraNear:1,shadowCameraFar:1e3};break}return n[e.id]=t,t}}}var Xy=0;function qy(n,e){return(e.castShadow?2:0)-(n.castShadow?2:0)+(e.map?1:0)-(n.map?1:0)}function Yy(n){let e=new Gy,t=Wy(),i={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)i.probe.push(new P);let s=new P,r=new rt,o=new rt;function a(l){let h=0,u=0,d=0;for(let E=0;E<9;E++)i.probe[E].set(0,0,0);let f=0,g=0,x=0,p=0,m=0,M=0,b=0,y=0,T=0,S=0,A=0;l.sort(qy);for(let E=0,C=l.length;E<C;E++){let I=l[E],L=I.color,V=I.intensity,q=I.distance,N=null;if(I.shadow&&I.shadow.map&&(I.shadow.map.texture.format===ms?N=I.shadow.map.texture:N=I.shadow.map.depthTexture||I.shadow.map.texture),I.isAmbientLight)h+=L.r*V,u+=L.g*V,d+=L.b*V;else if(I.isLightProbe){for(let Y=0;Y<9;Y++)i.probe[Y].addScaledVector(I.sh.coefficients[Y],V);A++}else if(I.isDirectionalLight){let Y=e.get(I);if(Y.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){let X=I.shadow,ne=t.get(I);ne.shadowIntensity=X.intensity,ne.shadowBias=X.bias,ne.shadowNormalBias=X.normalBias,ne.shadowRadius=X.radius,ne.shadowMapSize=X.mapSize,i.directionalShadow[f]=ne,i.directionalShadowMap[f]=N,i.directionalShadowMatrix[f]=I.shadow.matrix,M++}i.directional[f]=Y,f++}else if(I.isSpotLight){let Y=e.get(I);Y.position.setFromMatrixPosition(I.matrixWorld),Y.color.copy(L).multiplyScalar(V),Y.distance=q,Y.coneCos=Math.cos(I.angle),Y.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),Y.decay=I.decay,i.spot[x]=Y;let X=I.shadow;if(I.map&&(i.spotLightMap[T]=I.map,T++,X.updateMatrices(I),I.castShadow&&S++),i.spotLightMatrix[x]=X.matrix,I.castShadow){let ne=t.get(I);ne.shadowIntensity=X.intensity,ne.shadowBias=X.bias,ne.shadowNormalBias=X.normalBias,ne.shadowRadius=X.radius,ne.shadowMapSize=X.mapSize,i.spotShadow[x]=ne,i.spotShadowMap[x]=N,y++}x++}else if(I.isRectAreaLight){let Y=e.get(I);Y.color.copy(L).multiplyScalar(V),Y.halfWidth.set(I.width*.5,0,0),Y.halfHeight.set(0,I.height*.5,0),i.rectArea[p]=Y,p++}else if(I.isPointLight){let Y=e.get(I);if(Y.color.copy(I.color).multiplyScalar(I.intensity),Y.distance=I.distance,Y.decay=I.decay,I.castShadow){let X=I.shadow,ne=t.get(I);ne.shadowIntensity=X.intensity,ne.shadowBias=X.bias,ne.shadowNormalBias=X.normalBias,ne.shadowRadius=X.radius,ne.shadowMapSize=X.mapSize,ne.shadowCameraNear=X.camera.near,ne.shadowCameraFar=X.camera.far,i.pointShadow[g]=ne,i.pointShadowMap[g]=N,i.pointShadowMatrix[g]=I.shadow.matrix,b++}i.point[g]=Y,g++}else if(I.isHemisphereLight){let Y=e.get(I);Y.skyColor.copy(I.color).multiplyScalar(V),Y.groundColor.copy(I.groundColor).multiplyScalar(V),i.hemi[m]=Y,m++}}p>0&&(n.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=be.LTC_FLOAT_1,i.rectAreaLTC2=be.LTC_FLOAT_2):(i.rectAreaLTC1=be.LTC_HALF_1,i.rectAreaLTC2=be.LTC_HALF_2)),i.ambient[0]=h,i.ambient[1]=u,i.ambient[2]=d;let _=i.hash;(_.directionalLength!==f||_.pointLength!==g||_.spotLength!==x||_.rectAreaLength!==p||_.hemiLength!==m||_.numDirectionalShadows!==M||_.numPointShadows!==b||_.numSpotShadows!==y||_.numSpotMaps!==T||_.numLightProbes!==A)&&(i.directional.length=f,i.spot.length=x,i.rectArea.length=p,i.point.length=g,i.hemi.length=m,i.directionalShadow.length=M,i.directionalShadowMap.length=M,i.pointShadow.length=b,i.pointShadowMap.length=b,i.spotShadow.length=y,i.spotShadowMap.length=y,i.directionalShadowMatrix.length=M,i.pointShadowMatrix.length=b,i.spotLightMatrix.length=y+T-S,i.spotLightMap.length=T,i.numSpotLightShadowsWithMaps=S,i.numLightProbes=A,_.directionalLength=f,_.pointLength=g,_.spotLength=x,_.rectAreaLength=p,_.hemiLength=m,_.numDirectionalShadows=M,_.numPointShadows=b,_.numSpotShadows=y,_.numSpotMaps=T,_.numLightProbes=A,i.version=Xy++)}function c(l,h){let u=0,d=0,f=0,g=0,x=0,p=h.matrixWorldInverse;for(let m=0,M=l.length;m<M;m++){let b=l[m];if(b.isDirectionalLight){let y=i.directional[u];y.direction.setFromMatrixPosition(b.matrixWorld),s.setFromMatrixPosition(b.target.matrixWorld),y.direction.sub(s),y.direction.transformDirection(p),u++}else if(b.isSpotLight){let y=i.spot[f];y.position.setFromMatrixPosition(b.matrixWorld),y.position.applyMatrix4(p),y.direction.setFromMatrixPosition(b.matrixWorld),s.setFromMatrixPosition(b.target.matrixWorld),y.direction.sub(s),y.direction.transformDirection(p),f++}else if(b.isRectAreaLight){let y=i.rectArea[g];y.position.setFromMatrixPosition(b.matrixWorld),y.position.applyMatrix4(p),o.identity(),r.copy(b.matrixWorld),r.premultiply(p),o.extractRotation(r),y.halfWidth.set(b.width*.5,0,0),y.halfHeight.set(0,b.height*.5,0),y.halfWidth.applyMatrix4(o),y.halfHeight.applyMatrix4(o),g++}else if(b.isPointLight){let y=i.point[d];y.position.setFromMatrixPosition(b.matrixWorld),y.position.applyMatrix4(p),d++}else if(b.isHemisphereLight){let y=i.hemi[x];y.direction.setFromMatrixPosition(b.matrixWorld),y.direction.transformDirection(p),x++}}}return{setup:a,setupView:c,state:i}}function Ip(n){let e=new Yy(n),t=[],i=[],s=[];function r(d){u.camera=d,t.length=0,i.length=0,s.length=0}function o(d){t.push(d)}function a(d){i.push(d)}function c(d){s.push(d)}function l(){e.setup(t)}function h(d){e.setupView(t,d)}let u={lightsArray:t,shadowsArray:i,lightProbeGridArray:s,camera:null,lights:e,transmissionRenderTarget:{},textureUnits:0};return{init:r,state:u,setupLights:l,setupLightsView:h,pushLight:o,pushShadow:a,pushLightProbeGrid:c}}function $y(n){let e=new WeakMap;function t(s,r=0){let o=e.get(s),a;return o===void 0?(a=new Ip(n),e.set(s,[a])):r>=o.length?(a=new Ip(n),o.push(a)):a=o[r],a}function i(){e=new WeakMap}return{get:t,dispose:i}}var Zy=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,Jy=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,jy=[new P(1,0,0),new P(-1,0,0),new P(0,1,0),new P(0,-1,0),new P(0,0,1),new P(0,0,-1)],Ky=[new P(0,-1,0),new P(0,-1,0),new P(0,0,1),new P(0,0,-1),new P(0,-1,0),new P(0,-1,0)],Dp=new rt,ca=new P,Vu=new P;function Qy(n,e,t){let i=new br,s=new $,r=new $,o=new mt,a=new Ll,c=new Ul,l={},h=t.maxTextureSize,u={[Zi]:ti,[ti]:Zi,[Mi]:Mi},d=new bt({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new $},radius:{value:4}},vertexShader:Zy,fragmentShader:Jy}),f=d.clone();f.defines.HORIZONTAL_PASS=1;let g=new ut;g.setAttribute("position",new Ut(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let x=new Ke(g,d),p=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Us;let m=this.type;this.render=function(S,A,_){if(p.enabled===!1||p.autoUpdate===!1&&p.needsUpdate===!1||S.length===0)return;this.type===Rf&&($e("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),this.type=Us);let E=n.getRenderTarget(),C=n.getActiveCubeFace(),I=n.getActiveMipmapLevel(),L=n.state;L.setBlending(zt),L.buffers.depth.getReversed()===!0?L.buffers.color.setClear(0,0,0,0):L.buffers.color.setClear(1,1,1,1),L.buffers.depth.setTest(!0),L.setScissorTest(!1);let V=m!==this.type;V&&A.traverse(function(q){q.material&&(Array.isArray(q.material)?q.material.forEach(N=>N.needsUpdate=!0):q.material.needsUpdate=!0)});for(let q=0,N=S.length;q<N;q++){let Y=S[q],X=Y.shadow;if(X===void 0){$e("WebGLShadowMap:",Y,"has no shadow.");continue}if(X.autoUpdate===!1&&X.needsUpdate===!1)continue;s.copy(X.mapSize);let ne=X.getFrameExtents();s.multiply(ne),r.copy(X.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/ne.x),s.x=r.x*ne.x,X.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/ne.y),s.y=r.y*ne.y,X.mapSize.y=r.y));let ie=n.state.buffers.depth.getReversed();if(X.camera._reversedDepth=ie,X.map===null||V===!0){if(X.map!==null&&(X.map.depthTexture!==null&&(X.map.depthTexture.dispose(),X.map.depthTexture=null),X.map.dispose()),this.type===Ir){if(Y.isPointLight){$e("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}X.map=new Ht(s.x,s.y,{format:ms,type:ii,minFilter:ei,magFilter:ei,generateMipmaps:!1}),X.map.texture.name=Y.name+".shadowMap",X.map.depthTexture=new ji(s.x,s.y,Hi),X.map.depthTexture.name=Y.name+".shadowMapDepth",X.map.depthTexture.format=fn,X.map.depthTexture.compareFunction=null,X.map.depthTexture.minFilter=Ot,X.map.depthTexture.magFilter=Ot}else Y.isPointLight?(X.map=new Hc(s.x),X.map.depthTexture=new Tl(s.x,tn)):(X.map=new Ht(s.x,s.y),X.map.depthTexture=new ji(s.x,s.y,tn)),X.map.depthTexture.name=Y.name+".shadowMap",X.map.depthTexture.format=fn,this.type===Us?(X.map.depthTexture.compareFunction=ie?Bc:Oc,X.map.depthTexture.minFilter=ei,X.map.depthTexture.magFilter=ei):(X.map.depthTexture.compareFunction=null,X.map.depthTexture.minFilter=Ot,X.map.depthTexture.magFilter=Ot);X.camera.updateProjectionMatrix()}let ge=X.map.isWebGLCubeRenderTarget?6:1;for(let ue=0;ue<ge;ue++){if(X.map.isWebGLCubeRenderTarget)n.setRenderTarget(X.map,ue),n.clear();else{ue===0&&(n.setRenderTarget(X.map),n.clear());let xe=X.getViewport(ue);o.set(r.x*xe.x,r.y*xe.y,r.x*xe.z,r.y*xe.w),L.viewport(o)}if(Y.isPointLight){let xe=X.camera,Ne=X.matrix,st=Y.distance||xe.far;st!==xe.far&&(xe.far=st,xe.updateProjectionMatrix()),ca.setFromMatrixPosition(Y.matrixWorld),xe.position.copy(ca),Vu.copy(xe.position),Vu.add(jy[ue]),xe.up.copy(Ky[ue]),xe.lookAt(Vu),xe.updateMatrixWorld(),Ne.makeTranslation(-ca.x,-ca.y,-ca.z),Dp.multiplyMatrices(xe.projectionMatrix,xe.matrixWorldInverse),X._frustum.setFromProjectionMatrix(Dp,xe.coordinateSystem,xe.reversedDepth)}else X.updateMatrices(Y);i=X.getFrustum(),y(A,_,X.camera,Y,this.type)}X.isPointLightShadow!==!0&&this.type===Ir&&M(X,_),X.needsUpdate=!1}m=this.type,p.needsUpdate=!1,n.setRenderTarget(E,C,I)};function M(S,A){let _=e.update(x);d.defines.VSM_SAMPLES!==S.blurSamples&&(d.defines.VSM_SAMPLES=S.blurSamples,f.defines.VSM_SAMPLES=S.blurSamples,d.needsUpdate=!0,f.needsUpdate=!0),S.mapPass===null&&(S.mapPass=new Ht(s.x,s.y,{format:ms,type:ii})),d.uniforms.shadow_pass.value=S.map.depthTexture,d.uniforms.resolution.value=S.mapSize,d.uniforms.radius.value=S.radius,n.setRenderTarget(S.mapPass),n.clear(),n.renderBufferDirect(A,null,_,d,x,null),f.uniforms.shadow_pass.value=S.mapPass.texture,f.uniforms.resolution.value=S.mapSize,f.uniforms.radius.value=S.radius,n.setRenderTarget(S.map),n.clear(),n.renderBufferDirect(A,null,_,f,x,null)}function b(S,A,_,E){let C=null,I=_.isPointLight===!0?S.customDistanceMaterial:S.customDepthMaterial;if(I!==void 0)C=I;else if(C=_.isPointLight===!0?c:a,n.localClippingEnabled&&A.clipShadows===!0&&Array.isArray(A.clippingPlanes)&&A.clippingPlanes.length!==0||A.displacementMap&&A.displacementScale!==0||A.alphaMap&&A.alphaTest>0||A.map&&A.alphaTest>0||A.alphaToCoverage===!0){let L=C.uuid,V=A.uuid,q=l[L];q===void 0&&(q={},l[L]=q);let N=q[V];N===void 0&&(N=C.clone(),q[V]=N,A.addEventListener("dispose",T)),C=N}if(C.visible=A.visible,C.wireframe=A.wireframe,E===Ir?C.side=A.shadowSide!==null?A.shadowSide:A.side:C.side=A.shadowSide!==null?A.shadowSide:u[A.side],C.alphaMap=A.alphaMap,C.alphaTest=A.alphaToCoverage===!0?.5:A.alphaTest,C.map=A.map,C.clipShadows=A.clipShadows,C.clippingPlanes=A.clippingPlanes,C.clipIntersection=A.clipIntersection,C.displacementMap=A.displacementMap,C.displacementScale=A.displacementScale,C.displacementBias=A.displacementBias,C.wireframeLinewidth=A.wireframeLinewidth,C.linewidth=A.linewidth,_.isPointLight===!0&&C.isMeshDistanceMaterial===!0){let L=n.properties.get(C);L.light=_}return C}function y(S,A,_,E,C){if(S.visible===!1)return;if(S.layers.test(A.layers)&&(S.isMesh||S.isLine||S.isPoints)&&(S.castShadow||S.receiveShadow&&C===Ir)&&(!S.frustumCulled||i.intersectsObject(S))){S.modelViewMatrix.multiplyMatrices(_.matrixWorldInverse,S.matrixWorld);let V=e.update(S),q=S.material;if(Array.isArray(q)){let N=V.groups;for(let Y=0,X=N.length;Y<X;Y++){let ne=N[Y],ie=q[ne.materialIndex];if(ie&&ie.visible){let ge=b(S,ie,E,C);S.onBeforeShadow(n,S,A,_,V,ge,ne),n.renderBufferDirect(_,null,V,ge,S,ne),S.onAfterShadow(n,S,A,_,V,ge,ne)}}}else if(q.visible){let N=b(S,q,E,C);S.onBeforeShadow(n,S,A,_,V,N,null),n.renderBufferDirect(_,null,V,N,S,null),S.onAfterShadow(n,S,A,_,V,N,null)}}let L=S.children;for(let V=0,q=L.length;V<q;V++)y(L[V],A,_,E,C)}function T(S){S.target.removeEventListener("dispose",T);for(let _ in l){let E=l[_],C=S.target.uuid;C in E&&(E[C].dispose(),delete E[C])}}}function eM(n,e){function t(){let F=!1,Ee=new mt,oe=null,we=new mt(0,0,0,0);return{setMask:function(Pe){oe!==Pe&&!F&&(n.colorMask(Pe,Pe,Pe,Pe),oe=Pe)},setLocked:function(Pe){F=Pe},setClear:function(Pe,de,He,Oe,It){It===!0&&(Pe*=Oe,de*=Oe,He*=Oe),Ee.set(Pe,de,He,Oe),we.equals(Ee)===!1&&(n.clearColor(Pe,de,He,Oe),we.copy(Ee))},reset:function(){F=!1,oe=null,we.set(-1,0,0,0)}}}function i(){let F=!1,Ee=!1,oe=null,we=null,Pe=null;return{setReversed:function(de){if(Ee!==de){let He=e.get("EXT_clip_control");de?He.clipControlEXT(He.LOWER_LEFT_EXT,He.ZERO_TO_ONE_EXT):He.clipControlEXT(He.LOWER_LEFT_EXT,He.NEGATIVE_ONE_TO_ONE_EXT),Ee=de;let Oe=Pe;Pe=null,this.setClear(Oe)}},getReversed:function(){return Ee},setTest:function(de){de?le(n.DEPTH_TEST):Ae(n.DEPTH_TEST)},setMask:function(de){oe!==de&&!F&&(n.depthMask(de),oe=de)},setFunc:function(de){if(Ee&&(de=np[de]),we!==de){switch(de){case cl:n.depthFunc(n.NEVER);break;case hl:n.depthFunc(n.ALWAYS);break;case ul:n.depthFunc(n.LESS);break;case Is:n.depthFunc(n.LEQUAL);break;case dl:n.depthFunc(n.EQUAL);break;case fl:n.depthFunc(n.GEQUAL);break;case pl:n.depthFunc(n.GREATER);break;case ml:n.depthFunc(n.NOTEQUAL);break;default:n.depthFunc(n.LEQUAL)}we=de}},setLocked:function(de){F=de},setClear:function(de){Pe!==de&&(Pe=de,Ee&&(de=1-de),n.clearDepth(de))},reset:function(){F=!1,oe=null,we=null,Pe=null,Ee=!1}}}function s(){let F=!1,Ee=null,oe=null,we=null,Pe=null,de=null,He=null,Oe=null,It=null;return{setTest:function(Mt){F||(Mt?le(n.STENCIL_TEST):Ae(n.STENCIL_TEST))},setMask:function(Mt){Ee!==Mt&&!F&&(n.stencilMask(Mt),Ee=Mt)},setFunc:function(Mt,on,an){(oe!==Mt||we!==on||Pe!==an)&&(n.stencilFunc(Mt,on,an),oe=Mt,we=on,Pe=an)},setOp:function(Mt,on,an){(de!==Mt||He!==on||Oe!==an)&&(n.stencilOp(Mt,on,an),de=Mt,He=on,Oe=an)},setLocked:function(Mt){F=Mt},setClear:function(Mt){It!==Mt&&(n.clearStencil(Mt),It=Mt)},reset:function(){F=!1,Ee=null,oe=null,we=null,Pe=null,de=null,He=null,Oe=null,It=null}}}let r=new t,o=new i,a=new s,c=new WeakMap,l=new WeakMap,h={},u={},d={},f=new WeakMap,g=[],x=null,p=!1,m=null,M=null,b=null,y=null,T=null,S=null,A=null,_=new Te(0,0,0),E=0,C=!1,I=null,L=null,V=null,q=null,N=null,Y=n.getParameter(n.MAX_COMBINED_TEXTURE_IMAGE_UNITS),X=!1,ne=0,ie=n.getParameter(n.VERSION);ie.indexOf("WebGL")!==-1?(ne=parseFloat(/^WebGL (\d)/.exec(ie)[1]),X=ne>=1):ie.indexOf("OpenGL ES")!==-1&&(ne=parseFloat(/^OpenGL ES (\d)/.exec(ie)[1]),X=ne>=2);let ge=null,ue={},xe=n.getParameter(n.SCISSOR_BOX),Ne=n.getParameter(n.VIEWPORT),st=new mt().fromArray(xe),Xe=new mt().fromArray(Ne);function j(F,Ee,oe,we){let Pe=new Uint8Array(4),de=n.createTexture();n.bindTexture(F,de),n.texParameteri(F,n.TEXTURE_MIN_FILTER,n.NEAREST),n.texParameteri(F,n.TEXTURE_MAG_FILTER,n.NEAREST);for(let He=0;He<oe;He++)F===n.TEXTURE_3D||F===n.TEXTURE_2D_ARRAY?n.texImage3D(Ee,0,n.RGBA,1,1,we,0,n.RGBA,n.UNSIGNED_BYTE,Pe):n.texImage2D(Ee+He,0,n.RGBA,1,1,0,n.RGBA,n.UNSIGNED_BYTE,Pe);return de}let he={};he[n.TEXTURE_2D]=j(n.TEXTURE_2D,n.TEXTURE_2D,1),he[n.TEXTURE_CUBE_MAP]=j(n.TEXTURE_CUBE_MAP,n.TEXTURE_CUBE_MAP_POSITIVE_X,6),he[n.TEXTURE_2D_ARRAY]=j(n.TEXTURE_2D_ARRAY,n.TEXTURE_2D_ARRAY,1,1),he[n.TEXTURE_3D]=j(n.TEXTURE_3D,n.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),le(n.DEPTH_TEST),o.setFunc(Is),W(!1),G(pu),le(n.CULL_FACE),H(zt);function le(F){h[F]!==!0&&(n.enable(F),h[F]=!0)}function Ae(F){h[F]!==!1&&(n.disable(F),h[F]=!1)}function Fe(F,Ee){return d[F]!==Ee?(n.bindFramebuffer(F,Ee),d[F]=Ee,F===n.DRAW_FRAMEBUFFER&&(d[n.FRAMEBUFFER]=Ee),F===n.FRAMEBUFFER&&(d[n.DRAW_FRAMEBUFFER]=Ee),!0):!1}function ke(F,Ee){let oe=g,we=!1;if(F){oe=f.get(Ee),oe===void 0&&(oe=[],f.set(Ee,oe));let Pe=F.textures;if(oe.length!==Pe.length||oe[0]!==n.COLOR_ATTACHMENT0){for(let de=0,He=Pe.length;de<He;de++)oe[de]=n.COLOR_ATTACHMENT0+de;oe.length=Pe.length,we=!0}}else oe[0]!==n.BACK&&(oe[0]=n.BACK,we=!0);we&&n.drawBuffers(oe)}function ae(F){return x!==F?(n.useProgram(F),x=F,!0):!1}let ee={[Ci]:n.FUNC_ADD,[Cf]:n.FUNC_SUBTRACT,[Pf]:n.FUNC_REVERSE_SUBTRACT};ee[If]=n.MIN,ee[Df]=n.MAX;let O={[Ns]:n.ZERO,[Lf]:n.ONE,[Uf]:n.SRC_COLOR,[al]:n.SRC_ALPHA,[Bf]:n.SRC_ALPHA_SATURATE,[$o]:n.DST_COLOR,[Yo]:n.DST_ALPHA,[Nf]:n.ONE_MINUS_SRC_COLOR,[ll]:n.ONE_MINUS_SRC_ALPHA,[Of]:n.ONE_MINUS_DST_COLOR,[Ff]:n.ONE_MINUS_DST_ALPHA,[zf]:n.CONSTANT_COLOR,[kf]:n.ONE_MINUS_CONSTANT_COLOR,[Hf]:n.CONSTANT_ALPHA,[Vf]:n.ONE_MINUS_CONSTANT_ALPHA};function H(F,Ee,oe,we,Pe,de,He,Oe,It,Mt){if(F===zt){p===!0&&(Ae(n.BLEND),p=!1);return}if(p===!1&&(le(n.BLEND),p=!0),F!==Zl){if(F!==m||Mt!==C){if((M!==Ci||T!==Ci)&&(n.blendEquation(n.FUNC_ADD),M=Ci,T=Ci),Mt)switch(F){case Ps:n.blendFuncSeparate(n.ONE,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case mu:n.blendFunc(n.ONE,n.ONE);break;case gu:n.blendFuncSeparate(n.ZERO,n.ONE_MINUS_SRC_COLOR,n.ZERO,n.ONE);break;case _u:n.blendFuncSeparate(n.DST_COLOR,n.ONE_MINUS_SRC_ALPHA,n.ZERO,n.ONE);break;default:Ze("WebGLState: Invalid blending: ",F);break}else switch(F){case Ps:n.blendFuncSeparate(n.SRC_ALPHA,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case mu:n.blendFuncSeparate(n.SRC_ALPHA,n.ONE,n.ONE,n.ONE);break;case gu:Ze("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case _u:Ze("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Ze("WebGLState: Invalid blending: ",F);break}b=null,y=null,S=null,A=null,_.set(0,0,0),E=0,m=F,C=Mt}return}Pe=Pe||Ee,de=de||oe,He=He||we,(Ee!==M||Pe!==T)&&(n.blendEquationSeparate(ee[Ee],ee[Pe]),M=Ee,T=Pe),(oe!==b||we!==y||de!==S||He!==A)&&(n.blendFuncSeparate(O[oe],O[we],O[de],O[He]),b=oe,y=we,S=de,A=He),(Oe.equals(_)===!1||It!==E)&&(n.blendColor(Oe.r,Oe.g,Oe.b,It),_.copy(Oe),E=It),m=F,C=!1}function Q(F,Ee){F.side===Mi?Ae(n.CULL_FACE):le(n.CULL_FACE);let oe=F.side===ti;Ee&&(oe=!oe),W(oe),F.blending===Ps&&F.transparent===!1?H(zt):H(F.blending,F.blendEquation,F.blendSrc,F.blendDst,F.blendEquationAlpha,F.blendSrcAlpha,F.blendDstAlpha,F.blendColor,F.blendAlpha,F.premultipliedAlpha),o.setFunc(F.depthFunc),o.setTest(F.depthTest),o.setMask(F.depthWrite),r.setMask(F.colorWrite);let we=F.stencilWrite;a.setTest(we),we&&(a.setMask(F.stencilWriteMask),a.setFunc(F.stencilFunc,F.stencilRef,F.stencilFuncMask),a.setOp(F.stencilFail,F.stencilZFail,F.stencilZPass)),ce(F.polygonOffset,F.polygonOffsetFactor,F.polygonOffsetUnits),F.alphaToCoverage===!0?le(n.SAMPLE_ALPHA_TO_COVERAGE):Ae(n.SAMPLE_ALPHA_TO_COVERAGE)}function W(F){I!==F&&(F?n.frontFace(n.CW):n.frontFace(n.CCW),I=F)}function G(F){F!==Tf?(le(n.CULL_FACE),F!==L&&(F===pu?n.cullFace(n.BACK):F===Af?n.cullFace(n.FRONT):n.cullFace(n.FRONT_AND_BACK))):Ae(n.CULL_FACE),L=F}function se(F){F!==V&&(X&&n.lineWidth(F),V=F)}function ce(F,Ee,oe){F?(le(n.POLYGON_OFFSET_FILL),(q!==Ee||N!==oe)&&(q=Ee,N=oe,o.getReversed()&&(Ee=-Ee),n.polygonOffset(Ee,oe))):Ae(n.POLYGON_OFFSET_FILL)}function fe(F){F?le(n.SCISSOR_TEST):Ae(n.SCISSOR_TEST)}function me(F){F===void 0&&(F=n.TEXTURE0+Y-1),ge!==F&&(n.activeTexture(F),ge=F)}function D(F,Ee,oe){oe===void 0&&(ge===null?oe=n.TEXTURE0+Y-1:oe=ge);let we=ue[oe];we===void 0&&(we={type:void 0,texture:void 0},ue[oe]=we),(we.type!==F||we.texture!==Ee)&&(ge!==oe&&(n.activeTexture(oe),ge=oe),n.bindTexture(F,Ee||he[F]),we.type=F,we.texture=Ee)}function Me(){let F=ue[ge];F!==void 0&&F.type!==void 0&&(n.bindTexture(F.type,null),F.type=void 0,F.texture=void 0)}function Ve(){try{n.compressedTexImage2D(...arguments)}catch(F){Ze("WebGLState:",F)}}function R(){try{n.compressedTexImage3D(...arguments)}catch(F){Ze("WebGLState:",F)}}function v(){try{n.texSubImage2D(...arguments)}catch(F){Ze("WebGLState:",F)}}function U(){try{n.texSubImage3D(...arguments)}catch(F){Ze("WebGLState:",F)}}function B(){try{n.compressedTexSubImage2D(...arguments)}catch(F){Ze("WebGLState:",F)}}function k(){try{n.compressedTexSubImage3D(...arguments)}catch(F){Ze("WebGLState:",F)}}function pe(){try{n.texStorage2D(...arguments)}catch(F){Ze("WebGLState:",F)}}function _e(){try{n.texStorage3D(...arguments)}catch(F){Ze("WebGLState:",F)}}function te(){try{n.texImage2D(...arguments)}catch(F){Ze("WebGLState:",F)}}function re(){try{n.texImage3D(...arguments)}catch(F){Ze("WebGLState:",F)}}function Se(F){return u[F]!==void 0?u[F]:n.getParameter(F)}function Ie(F,Ee){u[F]!==Ee&&(n.pixelStorei(F,Ee),u[F]=Ee)}function ve(F){st.equals(F)===!1&&(n.scissor(F.x,F.y,F.z,F.w),st.copy(F))}function ye(F){Xe.equals(F)===!1&&(n.viewport(F.x,F.y,F.z,F.w),Xe.copy(F))}function Be(F,Ee){let oe=l.get(Ee);oe===void 0&&(oe=new WeakMap,l.set(Ee,oe));let we=oe.get(F);we===void 0&&(we=n.getUniformBlockIndex(Ee,F.name),oe.set(F,we))}function qe(F,Ee){let we=l.get(Ee).get(F);c.get(Ee)!==we&&(n.uniformBlockBinding(Ee,we,F.__bindingPointIndex),c.set(Ee,we))}function Je(){n.disable(n.BLEND),n.disable(n.CULL_FACE),n.disable(n.DEPTH_TEST),n.disable(n.POLYGON_OFFSET_FILL),n.disable(n.SCISSOR_TEST),n.disable(n.STENCIL_TEST),n.disable(n.SAMPLE_ALPHA_TO_COVERAGE),n.blendEquation(n.FUNC_ADD),n.blendFunc(n.ONE,n.ZERO),n.blendFuncSeparate(n.ONE,n.ZERO,n.ONE,n.ZERO),n.blendColor(0,0,0,0),n.colorMask(!0,!0,!0,!0),n.clearColor(0,0,0,0),n.depthMask(!0),n.depthFunc(n.LESS),o.setReversed(!1),n.clearDepth(1),n.stencilMask(4294967295),n.stencilFunc(n.ALWAYS,0,4294967295),n.stencilOp(n.KEEP,n.KEEP,n.KEEP),n.clearStencil(0),n.cullFace(n.BACK),n.frontFace(n.CCW),n.polygonOffset(0,0),n.activeTexture(n.TEXTURE0),n.bindFramebuffer(n.FRAMEBUFFER,null),n.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),n.bindFramebuffer(n.READ_FRAMEBUFFER,null),n.useProgram(null),n.lineWidth(1),n.scissor(0,0,n.canvas.width,n.canvas.height),n.viewport(0,0,n.canvas.width,n.canvas.height),n.pixelStorei(n.PACK_ALIGNMENT,4),n.pixelStorei(n.UNPACK_ALIGNMENT,4),n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,!1),n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,n.BROWSER_DEFAULT_WEBGL),n.pixelStorei(n.PACK_ROW_LENGTH,0),n.pixelStorei(n.PACK_SKIP_PIXELS,0),n.pixelStorei(n.PACK_SKIP_ROWS,0),n.pixelStorei(n.UNPACK_ROW_LENGTH,0),n.pixelStorei(n.UNPACK_IMAGE_HEIGHT,0),n.pixelStorei(n.UNPACK_SKIP_PIXELS,0),n.pixelStorei(n.UNPACK_SKIP_ROWS,0),n.pixelStorei(n.UNPACK_SKIP_IMAGES,0),h={},u={},ge=null,ue={},d={},f=new WeakMap,g=[],x=null,p=!1,m=null,M=null,b=null,y=null,T=null,S=null,A=null,_=new Te(0,0,0),E=0,C=!1,I=null,L=null,V=null,q=null,N=null,st.set(0,0,n.canvas.width,n.canvas.height),Xe.set(0,0,n.canvas.width,n.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:le,disable:Ae,bindFramebuffer:Fe,drawBuffers:ke,useProgram:ae,setBlending:H,setMaterial:Q,setFlipSided:W,setCullFace:G,setLineWidth:se,setPolygonOffset:ce,setScissorTest:fe,activeTexture:me,bindTexture:D,unbindTexture:Me,compressedTexImage2D:Ve,compressedTexImage3D:R,texImage2D:te,texImage3D:re,pixelStorei:Ie,getParameter:Se,updateUBOMapping:Be,uniformBlockBinding:qe,texStorage2D:pe,texStorage3D:_e,texSubImage2D:v,texSubImage3D:U,compressedTexSubImage2D:B,compressedTexSubImage3D:k,scissor:ve,viewport:ye,reset:Je}}function tM(n,e,t,i,s,r,o){let a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new $,h=new WeakMap,u=new Set,d,f=new WeakMap,g=!1;try{g=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(R,v){return g?new OffscreenCanvas(R,v):co("canvas")}function p(R,v,U){let B=1,k=Ve(R);if((k.width>U||k.height>U)&&(B=U/Math.max(k.width,k.height)),B<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){let pe=Math.floor(B*k.width),_e=Math.floor(B*k.height);d===void 0&&(d=x(pe,_e));let te=v?x(pe,_e):d;return te.width=pe,te.height=_e,te.getContext("2d").drawImage(R,0,0,pe,_e),$e("WebGLRenderer: Texture has been resized from ("+k.width+"x"+k.height+") to ("+pe+"x"+_e+")."),te}else return"data"in R&&$e("WebGLRenderer: Image in DataTexture is too big ("+k.width+"x"+k.height+")."),R;return R}function m(R){return R.generateMipmaps}function M(R){n.generateMipmap(R)}function b(R){return R.isWebGLCubeRenderTarget?n.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?n.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?n.TEXTURE_2D_ARRAY:n.TEXTURE_2D}function y(R,v,U,B,k,pe=!1){if(R!==null){if(n[R]!==void 0)return n[R];$e("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let _e;B&&(_e=e.get("EXT_texture_norm16"),_e||$e("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let te=v;if(v===n.RED&&(U===n.FLOAT&&(te=n.R32F),U===n.HALF_FLOAT&&(te=n.R16F),U===n.UNSIGNED_BYTE&&(te=n.R8),U===n.UNSIGNED_SHORT&&_e&&(te=_e.R16_EXT),U===n.SHORT&&_e&&(te=_e.R16_SNORM_EXT)),v===n.RED_INTEGER&&(U===n.UNSIGNED_BYTE&&(te=n.R8UI),U===n.UNSIGNED_SHORT&&(te=n.R16UI),U===n.UNSIGNED_INT&&(te=n.R32UI),U===n.BYTE&&(te=n.R8I),U===n.SHORT&&(te=n.R16I),U===n.INT&&(te=n.R32I)),v===n.RG&&(U===n.FLOAT&&(te=n.RG32F),U===n.HALF_FLOAT&&(te=n.RG16F),U===n.UNSIGNED_BYTE&&(te=n.RG8),U===n.UNSIGNED_SHORT&&_e&&(te=_e.RG16_EXT),U===n.SHORT&&_e&&(te=_e.RG16_SNORM_EXT)),v===n.RG_INTEGER&&(U===n.UNSIGNED_BYTE&&(te=n.RG8UI),U===n.UNSIGNED_SHORT&&(te=n.RG16UI),U===n.UNSIGNED_INT&&(te=n.RG32UI),U===n.BYTE&&(te=n.RG8I),U===n.SHORT&&(te=n.RG16I),U===n.INT&&(te=n.RG32I)),v===n.RGB_INTEGER&&(U===n.UNSIGNED_BYTE&&(te=n.RGB8UI),U===n.UNSIGNED_SHORT&&(te=n.RGB16UI),U===n.UNSIGNED_INT&&(te=n.RGB32UI),U===n.BYTE&&(te=n.RGB8I),U===n.SHORT&&(te=n.RGB16I),U===n.INT&&(te=n.RGB32I)),v===n.RGBA_INTEGER&&(U===n.UNSIGNED_BYTE&&(te=n.RGBA8UI),U===n.UNSIGNED_SHORT&&(te=n.RGBA16UI),U===n.UNSIGNED_INT&&(te=n.RGBA32UI),U===n.BYTE&&(te=n.RGBA8I),U===n.SHORT&&(te=n.RGBA16I),U===n.INT&&(te=n.RGBA32I)),v===n.RGB&&(U===n.UNSIGNED_SHORT&&_e&&(te=_e.RGB16_EXT),U===n.SHORT&&_e&&(te=_e.RGB16_SNORM_EXT),U===n.UNSIGNED_INT_5_9_9_9_REV&&(te=n.RGB9_E5),U===n.UNSIGNED_INT_10F_11F_11F_REV&&(te=n.R11F_G11F_B10F)),v===n.RGBA){let re=pe?lo:ht.getTransfer(k);U===n.FLOAT&&(te=n.RGBA32F),U===n.HALF_FLOAT&&(te=n.RGBA16F),U===n.UNSIGNED_BYTE&&(te=re===pt?n.SRGB8_ALPHA8:n.RGBA8),U===n.UNSIGNED_SHORT&&_e&&(te=_e.RGBA16_EXT),U===n.SHORT&&_e&&(te=_e.RGBA16_SNORM_EXT),U===n.UNSIGNED_SHORT_4_4_4_4&&(te=n.RGBA4),U===n.UNSIGNED_SHORT_5_5_5_1&&(te=n.RGB5_A1)}return(te===n.R16F||te===n.R32F||te===n.RG16F||te===n.RG32F||te===n.RGBA16F||te===n.RGBA32F)&&e.get("EXT_color_buffer_float"),te}function T(R,v){let U;return R?v===null||v===tn||v===ps?U=n.DEPTH24_STENCIL8:v===Hi?U=n.DEPTH32F_STENCIL8:v===Dr&&(U=n.DEPTH24_STENCIL8,$e("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):v===null||v===tn||v===ps?U=n.DEPTH_COMPONENT24:v===Hi?U=n.DEPTH_COMPONENT32F:v===Dr&&(U=n.DEPTH_COMPONENT16),U}function S(R,v){return m(R)===!0||R.isFramebufferTexture&&R.minFilter!==Ot&&R.minFilter!==ei?Math.log2(Math.max(v.width,v.height))+1:R.mipmaps!==void 0&&R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?v.mipmaps.length:1}function A(R){let v=R.target;v.removeEventListener("dispose",A),E(v),v.isVideoTexture&&h.delete(v),v.isHTMLTexture&&u.delete(v)}function _(R){let v=R.target;v.removeEventListener("dispose",_),I(v)}function E(R){let v=i.get(R);if(v.__webglInit===void 0)return;let U=R.source,B=f.get(U);if(B){let k=B[v.__cacheKey];k.usedTimes--,k.usedTimes===0&&C(R),Object.keys(B).length===0&&f.delete(U)}i.remove(R)}function C(R){let v=i.get(R);n.deleteTexture(v.__webglTexture);let U=R.source,B=f.get(U);delete B[v.__cacheKey],o.memory.textures--}function I(R){let v=i.get(R);if(R.depthTexture&&(R.depthTexture.dispose(),i.remove(R.depthTexture)),R.isWebGLCubeRenderTarget)for(let B=0;B<6;B++){if(Array.isArray(v.__webglFramebuffer[B]))for(let k=0;k<v.__webglFramebuffer[B].length;k++)n.deleteFramebuffer(v.__webglFramebuffer[B][k]);else n.deleteFramebuffer(v.__webglFramebuffer[B]);v.__webglDepthbuffer&&n.deleteRenderbuffer(v.__webglDepthbuffer[B])}else{if(Array.isArray(v.__webglFramebuffer))for(let B=0;B<v.__webglFramebuffer.length;B++)n.deleteFramebuffer(v.__webglFramebuffer[B]);else n.deleteFramebuffer(v.__webglFramebuffer);if(v.__webglDepthbuffer&&n.deleteRenderbuffer(v.__webglDepthbuffer),v.__webglMultisampledFramebuffer&&n.deleteFramebuffer(v.__webglMultisampledFramebuffer),v.__webglColorRenderbuffer)for(let B=0;B<v.__webglColorRenderbuffer.length;B++)v.__webglColorRenderbuffer[B]&&n.deleteRenderbuffer(v.__webglColorRenderbuffer[B]);v.__webglDepthRenderbuffer&&n.deleteRenderbuffer(v.__webglDepthRenderbuffer)}let U=R.textures;for(let B=0,k=U.length;B<k;B++){let pe=i.get(U[B]);pe.__webglTexture&&(n.deleteTexture(pe.__webglTexture),o.memory.textures--),i.remove(U[B])}i.remove(R)}let L=0;function V(){L=0}function q(){return L}function N(R){L=R}function Y(){let R=L;return R>=s.maxTextures&&$e("WebGLTextures: Trying to use "+R+" texture units while this GPU supports only "+s.maxTextures),L+=1,R}function X(R){let v=[];return v.push(R.wrapS),v.push(R.wrapT),v.push(R.wrapR||0),v.push(R.magFilter),v.push(R.minFilter),v.push(R.anisotropy),v.push(R.internalFormat),v.push(R.format),v.push(R.type),v.push(R.generateMipmaps),v.push(R.premultiplyAlpha),v.push(R.flipY),v.push(R.unpackAlignment),v.push(R.colorSpace),v.join()}function ne(R,v){let U=i.get(R);if(R.isVideoTexture&&D(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&U.__version!==R.version){let B=R.image;if(B===null)$e("WebGLRenderer: Texture marked for update but no image data found.");else if(B.complete===!1)$e("WebGLRenderer: Texture marked for update but image is incomplete");else{Ae(U,R,v);return}}else R.isExternalTexture&&(U.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(n.TEXTURE_2D,U.__webglTexture,n.TEXTURE0+v)}function ie(R,v){let U=i.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&U.__version!==R.version){Ae(U,R,v);return}else R.isExternalTexture&&(U.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(n.TEXTURE_2D_ARRAY,U.__webglTexture,n.TEXTURE0+v)}function ge(R,v){let U=i.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&U.__version!==R.version){Ae(U,R,v);return}t.bindTexture(n.TEXTURE_3D,U.__webglTexture,n.TEXTURE0+v)}function ue(R,v){let U=i.get(R);if(R.isCubeDepthTexture!==!0&&R.version>0&&U.__version!==R.version){Fe(U,R,v);return}t.bindTexture(n.TEXTURE_CUBE_MAP,U.__webglTexture,n.TEXTURE0+v)}let xe={[zi]:n.REPEAT,[un]:n.CLAMP_TO_EDGE,[gl]:n.MIRRORED_REPEAT},Ne={[Ot]:n.NEAREST,[Xf]:n.NEAREST_MIPMAP_NEAREST,[ta]:n.NEAREST_MIPMAP_LINEAR,[ei]:n.LINEAR,[Ql]:n.LINEAR_MIPMAP_NEAREST,[fs]:n.LINEAR_MIPMAP_LINEAR},st={[$f]:n.NEVER,[Qf]:n.ALWAYS,[Zf]:n.LESS,[Oc]:n.LEQUAL,[Jf]:n.EQUAL,[Bc]:n.GEQUAL,[jf]:n.GREATER,[Kf]:n.NOTEQUAL};function Xe(R,v){if(v.type===Hi&&e.has("OES_texture_float_linear")===!1&&(v.magFilter===ei||v.magFilter===Ql||v.magFilter===ta||v.magFilter===fs||v.minFilter===ei||v.minFilter===Ql||v.minFilter===ta||v.minFilter===fs)&&$e("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),n.texParameteri(R,n.TEXTURE_WRAP_S,xe[v.wrapS]),n.texParameteri(R,n.TEXTURE_WRAP_T,xe[v.wrapT]),(R===n.TEXTURE_3D||R===n.TEXTURE_2D_ARRAY)&&n.texParameteri(R,n.TEXTURE_WRAP_R,xe[v.wrapR]),n.texParameteri(R,n.TEXTURE_MAG_FILTER,Ne[v.magFilter]),n.texParameteri(R,n.TEXTURE_MIN_FILTER,Ne[v.minFilter]),v.compareFunction&&(n.texParameteri(R,n.TEXTURE_COMPARE_MODE,n.COMPARE_REF_TO_TEXTURE),n.texParameteri(R,n.TEXTURE_COMPARE_FUNC,st[v.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(v.magFilter===Ot||v.minFilter!==ta&&v.minFilter!==fs||v.type===Hi&&e.has("OES_texture_float_linear")===!1)return;if(v.anisotropy>1||i.get(v).__currentAnisotropy){let U=e.get("EXT_texture_filter_anisotropic");n.texParameterf(R,U.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(v.anisotropy,s.getMaxAnisotropy())),i.get(v).__currentAnisotropy=v.anisotropy}}}function j(R,v){let U=!1;R.__webglInit===void 0&&(R.__webglInit=!0,v.addEventListener("dispose",A));let B=v.source,k=f.get(B);k===void 0&&(k={},f.set(B,k));let pe=X(v);if(pe!==R.__cacheKey){k[pe]===void 0&&(k[pe]={texture:n.createTexture(),usedTimes:0},o.memory.textures++,U=!0),k[pe].usedTimes++;let _e=k[R.__cacheKey];_e!==void 0&&(k[R.__cacheKey].usedTimes--,_e.usedTimes===0&&C(v)),R.__cacheKey=pe,R.__webglTexture=k[pe].texture}return U}function he(R,v,U){return Math.floor(Math.floor(R/U)/v)}function le(R,v,U,B){let pe=R.updateRanges;if(pe.length===0)t.texSubImage2D(n.TEXTURE_2D,0,0,0,v.width,v.height,U,B,v.data);else{pe.sort((Ie,ve)=>Ie.start-ve.start);let _e=0;for(let Ie=1;Ie<pe.length;Ie++){let ve=pe[_e],ye=pe[Ie],Be=ve.start+ve.count,qe=he(ye.start,v.width,4),Je=he(ve.start,v.width,4);ye.start<=Be+1&&qe===Je&&he(ye.start+ye.count-1,v.width,4)===qe?ve.count=Math.max(ve.count,ye.start+ye.count-ve.start):(++_e,pe[_e]=ye)}pe.length=_e+1;let te=t.getParameter(n.UNPACK_ROW_LENGTH),re=t.getParameter(n.UNPACK_SKIP_PIXELS),Se=t.getParameter(n.UNPACK_SKIP_ROWS);t.pixelStorei(n.UNPACK_ROW_LENGTH,v.width);for(let Ie=0,ve=pe.length;Ie<ve;Ie++){let ye=pe[Ie],Be=Math.floor(ye.start/4),qe=Math.ceil(ye.count/4),Je=Be%v.width,F=Math.floor(Be/v.width),Ee=qe,oe=1;t.pixelStorei(n.UNPACK_SKIP_PIXELS,Je),t.pixelStorei(n.UNPACK_SKIP_ROWS,F),t.texSubImage2D(n.TEXTURE_2D,0,Je,F,Ee,oe,U,B,v.data)}R.clearUpdateRanges(),t.pixelStorei(n.UNPACK_ROW_LENGTH,te),t.pixelStorei(n.UNPACK_SKIP_PIXELS,re),t.pixelStorei(n.UNPACK_SKIP_ROWS,Se)}}function Ae(R,v,U){let B=n.TEXTURE_2D;(v.isDataArrayTexture||v.isCompressedArrayTexture)&&(B=n.TEXTURE_2D_ARRAY),v.isData3DTexture&&(B=n.TEXTURE_3D);let k=j(R,v),pe=v.source;t.bindTexture(B,R.__webglTexture,n.TEXTURE0+U);let _e=i.get(pe);if(pe.version!==_e.__version||k===!0){if(t.activeTexture(n.TEXTURE0+U),(typeof ImageBitmap<"u"&&v.image instanceof ImageBitmap)===!1){let oe=ht.getPrimaries(ht.workingColorSpace),we=v.colorSpace===On?null:ht.getPrimaries(v.colorSpace),Pe=v.colorSpace===On||oe===we?n.NONE:n.BROWSER_DEFAULT_WEBGL;t.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,v.flipY),t.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),t.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,Pe)}t.pixelStorei(n.UNPACK_ALIGNMENT,v.unpackAlignment);let re=p(v.image,!1,s.maxTextureSize);re=Me(v,re);let Se=r.convert(v.format,v.colorSpace),Ie=r.convert(v.type),ve=y(v.internalFormat,Se,Ie,v.normalized,v.colorSpace,v.isVideoTexture);Xe(B,v);let ye,Be=v.mipmaps,qe=v.isVideoTexture!==!0,Je=_e.__version===void 0||k===!0,F=pe.dataReady,Ee=S(v,re);if(v.isDepthTexture)ve=T(v.format===_n,v.type),Je&&(qe?t.texStorage2D(n.TEXTURE_2D,1,ve,re.width,re.height):t.texImage2D(n.TEXTURE_2D,0,ve,re.width,re.height,0,Se,Ie,null));else if(v.isDataTexture)if(Be.length>0){qe&&Je&&t.texStorage2D(n.TEXTURE_2D,Ee,ve,Be[0].width,Be[0].height);for(let oe=0,we=Be.length;oe<we;oe++)ye=Be[oe],qe?F&&t.texSubImage2D(n.TEXTURE_2D,oe,0,0,ye.width,ye.height,Se,Ie,ye.data):t.texImage2D(n.TEXTURE_2D,oe,ve,ye.width,ye.height,0,Se,Ie,ye.data);v.generateMipmaps=!1}else qe?(Je&&t.texStorage2D(n.TEXTURE_2D,Ee,ve,re.width,re.height),F&&le(v,re,Se,Ie)):t.texImage2D(n.TEXTURE_2D,0,ve,re.width,re.height,0,Se,Ie,re.data);else if(v.isCompressedTexture)if(v.isCompressedArrayTexture){qe&&Je&&t.texStorage3D(n.TEXTURE_2D_ARRAY,Ee,ve,Be[0].width,Be[0].height,re.depth);for(let oe=0,we=Be.length;oe<we;oe++)if(ye=Be[oe],v.format!==Si)if(Se!==null)if(qe){if(F)if(v.layerUpdates.size>0){let Pe=Iu(ye.width,ye.height,v.format,v.type);for(let de of v.layerUpdates){let He=ye.data.subarray(de*Pe/ye.data.BYTES_PER_ELEMENT,(de+1)*Pe/ye.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,oe,0,0,de,ye.width,ye.height,1,Se,He)}v.clearLayerUpdates()}else t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,oe,0,0,0,ye.width,ye.height,re.depth,Se,ye.data)}else t.compressedTexImage3D(n.TEXTURE_2D_ARRAY,oe,ve,ye.width,ye.height,re.depth,0,ye.data,0,0);else $e("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else qe?F&&t.texSubImage3D(n.TEXTURE_2D_ARRAY,oe,0,0,0,ye.width,ye.height,re.depth,Se,Ie,ye.data):t.texImage3D(n.TEXTURE_2D_ARRAY,oe,ve,ye.width,ye.height,re.depth,0,Se,Ie,ye.data)}else{qe&&Je&&t.texStorage2D(n.TEXTURE_2D,Ee,ve,Be[0].width,Be[0].height);for(let oe=0,we=Be.length;oe<we;oe++)ye=Be[oe],v.format!==Si?Se!==null?qe?F&&t.compressedTexSubImage2D(n.TEXTURE_2D,oe,0,0,ye.width,ye.height,Se,ye.data):t.compressedTexImage2D(n.TEXTURE_2D,oe,ve,ye.width,ye.height,0,ye.data):$e("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):qe?F&&t.texSubImage2D(n.TEXTURE_2D,oe,0,0,ye.width,ye.height,Se,Ie,ye.data):t.texImage2D(n.TEXTURE_2D,oe,ve,ye.width,ye.height,0,Se,Ie,ye.data)}else if(v.isDataArrayTexture)if(qe){if(Je&&t.texStorage3D(n.TEXTURE_2D_ARRAY,Ee,ve,re.width,re.height,re.depth),F)if(v.layerUpdates.size>0){let oe=Iu(re.width,re.height,v.format,v.type);for(let we of v.layerUpdates){let Pe=re.data.subarray(we*oe/re.data.BYTES_PER_ELEMENT,(we+1)*oe/re.data.BYTES_PER_ELEMENT);t.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,we,re.width,re.height,1,Se,Ie,Pe)}v.clearLayerUpdates()}else t.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,0,re.width,re.height,re.depth,Se,Ie,re.data)}else t.texImage3D(n.TEXTURE_2D_ARRAY,0,ve,re.width,re.height,re.depth,0,Se,Ie,re.data);else if(v.isData3DTexture)qe?(Je&&t.texStorage3D(n.TEXTURE_3D,Ee,ve,re.width,re.height,re.depth),F&&t.texSubImage3D(n.TEXTURE_3D,0,0,0,0,re.width,re.height,re.depth,Se,Ie,re.data)):t.texImage3D(n.TEXTURE_3D,0,ve,re.width,re.height,re.depth,0,Se,Ie,re.data);else if(v.isFramebufferTexture){if(Je)if(qe)t.texStorage2D(n.TEXTURE_2D,Ee,ve,re.width,re.height);else{let oe=re.width,we=re.height;for(let Pe=0;Pe<Ee;Pe++)t.texImage2D(n.TEXTURE_2D,Pe,ve,oe,we,0,Se,Ie,null),oe>>=1,we>>=1}}else if(v.isHTMLTexture){if("texElementImage2D"in n){let oe=n.canvas;if(oe.hasAttribute("layoutsubtree")||oe.setAttribute("layoutsubtree","true"),re.parentNode!==oe){oe.appendChild(re),u.add(v),oe.onpaint=we=>{let Pe=we.changedElements;for(let de of u)Pe.includes(de.image)&&(de.needsUpdate=!0)},oe.requestPaint();return}if(n.texElementImage2D.length===3)n.texElementImage2D(n.TEXTURE_2D,n.RGBA8,re);else{let Pe=n.RGBA,de=n.RGBA,He=n.UNSIGNED_BYTE;n.texElementImage2D(n.TEXTURE_2D,0,Pe,de,He,re)}n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MIN_FILTER,n.LINEAR),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE)}}else if(Be.length>0){if(qe&&Je){let oe=Ve(Be[0]);t.texStorage2D(n.TEXTURE_2D,Ee,ve,oe.width,oe.height)}for(let oe=0,we=Be.length;oe<we;oe++)ye=Be[oe],qe?F&&t.texSubImage2D(n.TEXTURE_2D,oe,0,0,Se,Ie,ye):t.texImage2D(n.TEXTURE_2D,oe,ve,Se,Ie,ye);v.generateMipmaps=!1}else if(qe){if(Je){let oe=Ve(re);t.texStorage2D(n.TEXTURE_2D,Ee,ve,oe.width,oe.height)}F&&t.texSubImage2D(n.TEXTURE_2D,0,0,0,Se,Ie,re)}else t.texImage2D(n.TEXTURE_2D,0,ve,Se,Ie,re);m(v)&&M(B),_e.__version=pe.version,v.onUpdate&&v.onUpdate(v)}R.__version=v.version}function Fe(R,v,U){if(v.image.length!==6)return;let B=j(R,v),k=v.source;t.bindTexture(n.TEXTURE_CUBE_MAP,R.__webglTexture,n.TEXTURE0+U);let pe=i.get(k);if(k.version!==pe.__version||B===!0){t.activeTexture(n.TEXTURE0+U);let _e=ht.getPrimaries(ht.workingColorSpace),te=v.colorSpace===On?null:ht.getPrimaries(v.colorSpace),re=v.colorSpace===On||_e===te?n.NONE:n.BROWSER_DEFAULT_WEBGL;t.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,v.flipY),t.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),t.pixelStorei(n.UNPACK_ALIGNMENT,v.unpackAlignment),t.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,re);let Se=v.isCompressedTexture||v.image[0].isCompressedTexture,Ie=v.image[0]&&v.image[0].isDataTexture,ve=[];for(let de=0;de<6;de++)!Se&&!Ie?ve[de]=p(v.image[de],!0,s.maxCubemapSize):ve[de]=Ie?v.image[de].image:v.image[de],ve[de]=Me(v,ve[de]);let ye=ve[0],Be=r.convert(v.format,v.colorSpace),qe=r.convert(v.type),Je=y(v.internalFormat,Be,qe,v.normalized,v.colorSpace),F=v.isVideoTexture!==!0,Ee=pe.__version===void 0||B===!0,oe=k.dataReady,we=S(v,ye);Xe(n.TEXTURE_CUBE_MAP,v);let Pe;if(Se){F&&Ee&&t.texStorage2D(n.TEXTURE_CUBE_MAP,we,Je,ye.width,ye.height);for(let de=0;de<6;de++){Pe=ve[de].mipmaps;for(let He=0;He<Pe.length;He++){let Oe=Pe[He];v.format!==Si?Be!==null?F?oe&&t.compressedTexSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,0,0,Oe.width,Oe.height,Be,Oe.data):t.compressedTexImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,Je,Oe.width,Oe.height,0,Oe.data):$e("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):F?oe&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,0,0,Oe.width,Oe.height,Be,qe,Oe.data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,Je,Oe.width,Oe.height,0,Be,qe,Oe.data)}}}else{if(Pe=v.mipmaps,F&&Ee){Pe.length>0&&we++;let de=Ve(ve[0]);t.texStorage2D(n.TEXTURE_CUBE_MAP,we,Je,de.width,de.height)}for(let de=0;de<6;de++)if(Ie){F?oe&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,ve[de].width,ve[de].height,Be,qe,ve[de].data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,Je,ve[de].width,ve[de].height,0,Be,qe,ve[de].data);for(let He=0;He<Pe.length;He++){let It=Pe[He].image[de].image;F?oe&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,0,0,It.width,It.height,Be,qe,It.data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,Je,It.width,It.height,0,Be,qe,It.data)}}else{F?oe&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,Be,qe,ve[de]):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,Je,Be,qe,ve[de]);for(let He=0;He<Pe.length;He++){let Oe=Pe[He];F?oe&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,0,0,Be,qe,Oe.image[de]):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,Je,Be,qe,Oe.image[de])}}}m(v)&&M(n.TEXTURE_CUBE_MAP),pe.__version=k.version,v.onUpdate&&v.onUpdate(v)}R.__version=v.version}function ke(R,v,U,B,k,pe){let _e=r.convert(U.format,U.colorSpace),te=r.convert(U.type),re=y(U.internalFormat,_e,te,U.normalized,U.colorSpace),Se=i.get(v),Ie=i.get(U);if(Ie.__renderTarget=v,!Se.__hasExternalTextures){let ve=Math.max(1,v.width>>pe),ye=Math.max(1,v.height>>pe);k===n.TEXTURE_3D||k===n.TEXTURE_2D_ARRAY?t.texImage3D(k,pe,re,ve,ye,v.depth,0,_e,te,null):t.texImage2D(k,pe,re,ve,ye,0,_e,te,null)}t.bindFramebuffer(n.FRAMEBUFFER,R),me(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,B,k,Ie.__webglTexture,0,fe(v)):(k===n.TEXTURE_2D||k>=n.TEXTURE_CUBE_MAP_POSITIVE_X&&k<=n.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&n.framebufferTexture2D(n.FRAMEBUFFER,B,k,Ie.__webglTexture,pe),t.bindFramebuffer(n.FRAMEBUFFER,null)}function ae(R,v,U){if(n.bindRenderbuffer(n.RENDERBUFFER,R),v.depthBuffer){let B=v.depthTexture,k=B&&B.isDepthTexture?B.type:null,pe=T(v.stencilBuffer,k),_e=v.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;me(v)?a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,fe(v),pe,v.width,v.height):U?n.renderbufferStorageMultisample(n.RENDERBUFFER,fe(v),pe,v.width,v.height):n.renderbufferStorage(n.RENDERBUFFER,pe,v.width,v.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,_e,n.RENDERBUFFER,R)}else{let B=v.textures;for(let k=0;k<B.length;k++){let pe=B[k],_e=r.convert(pe.format,pe.colorSpace),te=r.convert(pe.type),re=y(pe.internalFormat,_e,te,pe.normalized,pe.colorSpace);me(v)?a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,fe(v),re,v.width,v.height):U?n.renderbufferStorageMultisample(n.RENDERBUFFER,fe(v),re,v.width,v.height):n.renderbufferStorage(n.RENDERBUFFER,re,v.width,v.height)}}n.bindRenderbuffer(n.RENDERBUFFER,null)}function ee(R,v,U){let B=v.isWebGLCubeRenderTarget===!0;if(t.bindFramebuffer(n.FRAMEBUFFER,R),!(v.depthTexture&&v.depthTexture.isDepthTexture))throw new Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");let k=i.get(v.depthTexture);if(k.__renderTarget=v,(!k.__webglTexture||v.depthTexture.image.width!==v.width||v.depthTexture.image.height!==v.height)&&(v.depthTexture.image.width=v.width,v.depthTexture.image.height=v.height,v.depthTexture.needsUpdate=!0),B){if(k.__webglInit===void 0&&(k.__webglInit=!0,v.depthTexture.addEventListener("dispose",A)),k.__webglTexture===void 0){k.__webglTexture=n.createTexture(),t.bindTexture(n.TEXTURE_CUBE_MAP,k.__webglTexture),Xe(n.TEXTURE_CUBE_MAP,v.depthTexture);let Se=r.convert(v.depthTexture.format),Ie=r.convert(v.depthTexture.type),ve;v.depthTexture.format===fn?ve=n.DEPTH_COMPONENT24:v.depthTexture.format===_n&&(ve=n.DEPTH24_STENCIL8);for(let ye=0;ye<6;ye++)n.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ye,0,ve,v.width,v.height,0,Se,Ie,null)}}else ne(v.depthTexture,0);let pe=k.__webglTexture,_e=fe(v),te=B?n.TEXTURE_CUBE_MAP_POSITIVE_X+U:n.TEXTURE_2D,re=v.depthTexture.format===_n?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;if(v.depthTexture.format===fn)me(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,re,te,pe,0,_e):n.framebufferTexture2D(n.FRAMEBUFFER,re,te,pe,0);else if(v.depthTexture.format===_n)me(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,re,te,pe,0,_e):n.framebufferTexture2D(n.FRAMEBUFFER,re,te,pe,0);else throw new Error("THREE.WebGLTextures: Unknown depthTexture format.")}function O(R){let v=i.get(R),U=R.isWebGLCubeRenderTarget===!0;if(v.__boundDepthTexture!==R.depthTexture){let B=R.depthTexture;if(v.__depthDisposeCallback&&v.__depthDisposeCallback(),B){let k=()=>{delete v.__boundDepthTexture,delete v.__depthDisposeCallback,B.removeEventListener("dispose",k)};B.addEventListener("dispose",k),v.__depthDisposeCallback=k}v.__boundDepthTexture=B}if(R.depthTexture&&!v.__autoAllocateDepthBuffer)if(U)for(let B=0;B<6;B++)ee(v.__webglFramebuffer[B],R,B);else{let B=R.texture.mipmaps;B&&B.length>0?ee(v.__webglFramebuffer[0],R,0):ee(v.__webglFramebuffer,R,0)}else if(U){v.__webglDepthbuffer=[];for(let B=0;B<6;B++)if(t.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer[B]),v.__webglDepthbuffer[B]===void 0)v.__webglDepthbuffer[B]=n.createRenderbuffer(),ae(v.__webglDepthbuffer[B],R,!1);else{let k=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,pe=v.__webglDepthbuffer[B];n.bindRenderbuffer(n.RENDERBUFFER,pe),n.framebufferRenderbuffer(n.FRAMEBUFFER,k,n.RENDERBUFFER,pe)}}else{let B=R.texture.mipmaps;if(B&&B.length>0?t.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer[0]):t.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer),v.__webglDepthbuffer===void 0)v.__webglDepthbuffer=n.createRenderbuffer(),ae(v.__webglDepthbuffer,R,!1);else{let k=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,pe=v.__webglDepthbuffer;n.bindRenderbuffer(n.RENDERBUFFER,pe),n.framebufferRenderbuffer(n.FRAMEBUFFER,k,n.RENDERBUFFER,pe)}}t.bindFramebuffer(n.FRAMEBUFFER,null)}function H(R,v,U){let B=i.get(R);v!==void 0&&ke(B.__webglFramebuffer,R,R.texture,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,0),U!==void 0&&O(R)}function Q(R){let v=R.texture,U=i.get(R),B=i.get(v);R.addEventListener("dispose",_);let k=R.textures,pe=R.isWebGLCubeRenderTarget===!0,_e=k.length>1;if(_e||(B.__webglTexture===void 0&&(B.__webglTexture=n.createTexture()),B.__version=v.version,o.memory.textures++),pe){U.__webglFramebuffer=[];for(let te=0;te<6;te++)if(v.mipmaps&&v.mipmaps.length>0){U.__webglFramebuffer[te]=[];for(let re=0;re<v.mipmaps.length;re++)U.__webglFramebuffer[te][re]=n.createFramebuffer()}else U.__webglFramebuffer[te]=n.createFramebuffer()}else{if(v.mipmaps&&v.mipmaps.length>0){U.__webglFramebuffer=[];for(let te=0;te<v.mipmaps.length;te++)U.__webglFramebuffer[te]=n.createFramebuffer()}else U.__webglFramebuffer=n.createFramebuffer();if(_e)for(let te=0,re=k.length;te<re;te++){let Se=i.get(k[te]);Se.__webglTexture===void 0&&(Se.__webglTexture=n.createTexture(),o.memory.textures++)}if(R.samples>0&&me(R)===!1){U.__webglMultisampledFramebuffer=n.createFramebuffer(),U.__webglColorRenderbuffer=[],t.bindFramebuffer(n.FRAMEBUFFER,U.__webglMultisampledFramebuffer);for(let te=0;te<k.length;te++){let re=k[te];U.__webglColorRenderbuffer[te]=n.createRenderbuffer(),n.bindRenderbuffer(n.RENDERBUFFER,U.__webglColorRenderbuffer[te]);let Se=r.convert(re.format,re.colorSpace),Ie=r.convert(re.type),ve=y(re.internalFormat,Se,Ie,re.normalized,re.colorSpace,R.isXRRenderTarget===!0),ye=fe(R);n.renderbufferStorageMultisample(n.RENDERBUFFER,ye,ve,R.width,R.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+te,n.RENDERBUFFER,U.__webglColorRenderbuffer[te])}n.bindRenderbuffer(n.RENDERBUFFER,null),R.depthBuffer&&(U.__webglDepthRenderbuffer=n.createRenderbuffer(),ae(U.__webglDepthRenderbuffer,R,!0)),t.bindFramebuffer(n.FRAMEBUFFER,null)}}if(pe){t.bindTexture(n.TEXTURE_CUBE_MAP,B.__webglTexture),Xe(n.TEXTURE_CUBE_MAP,v);for(let te=0;te<6;te++)if(v.mipmaps&&v.mipmaps.length>0)for(let re=0;re<v.mipmaps.length;re++)ke(U.__webglFramebuffer[te][re],R,v,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+te,re);else ke(U.__webglFramebuffer[te],R,v,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+te,0);m(v)&&M(n.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(_e){for(let te=0,re=k.length;te<re;te++){let Se=k[te],Ie=i.get(Se),ve=n.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(ve=R.isWebGL3DRenderTarget?n.TEXTURE_3D:n.TEXTURE_2D_ARRAY),t.bindTexture(ve,Ie.__webglTexture),Xe(ve,Se),ke(U.__webglFramebuffer,R,Se,n.COLOR_ATTACHMENT0+te,ve,0),m(Se)&&M(ve)}t.unbindTexture()}else{let te=n.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(te=R.isWebGL3DRenderTarget?n.TEXTURE_3D:n.TEXTURE_2D_ARRAY),t.bindTexture(te,B.__webglTexture),Xe(te,v),v.mipmaps&&v.mipmaps.length>0)for(let re=0;re<v.mipmaps.length;re++)ke(U.__webglFramebuffer[re],R,v,n.COLOR_ATTACHMENT0,te,re);else ke(U.__webglFramebuffer,R,v,n.COLOR_ATTACHMENT0,te,0);m(v)&&M(te),t.unbindTexture()}R.depthBuffer&&O(R)}function W(R){let v=R.textures;for(let U=0,B=v.length;U<B;U++){let k=v[U];if(m(k)){let pe=b(R),_e=i.get(k).__webglTexture;t.bindTexture(pe,_e),M(pe),t.unbindTexture()}}}let G=[],se=[];function ce(R){if(R.samples>0){if(me(R)===!1){let v=R.textures,U=R.width,B=R.height,k=n.COLOR_BUFFER_BIT,pe=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,_e=i.get(R),te=v.length>1;if(te)for(let Se=0;Se<v.length;Se++)t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+Se,n.RENDERBUFFER,null),t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+Se,n.TEXTURE_2D,null,0);t.bindFramebuffer(n.READ_FRAMEBUFFER,_e.__webglMultisampledFramebuffer);let re=R.texture.mipmaps;re&&re.length>0?t.bindFramebuffer(n.DRAW_FRAMEBUFFER,_e.__webglFramebuffer[0]):t.bindFramebuffer(n.DRAW_FRAMEBUFFER,_e.__webglFramebuffer);for(let Se=0;Se<v.length;Se++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(k|=n.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(k|=n.STENCIL_BUFFER_BIT)),te){n.framebufferRenderbuffer(n.READ_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.RENDERBUFFER,_e.__webglColorRenderbuffer[Se]);let Ie=i.get(v[Se]).__webglTexture;n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,Ie,0)}n.blitFramebuffer(0,0,U,B,0,0,U,B,k,n.NEAREST),c===!0&&(G.length=0,se.length=0,G.push(n.COLOR_ATTACHMENT0+Se),R.depthBuffer&&R.resolveDepthBuffer===!1&&(G.push(pe),se.push(pe),n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,se)),n.invalidateFramebuffer(n.READ_FRAMEBUFFER,G))}if(t.bindFramebuffer(n.READ_FRAMEBUFFER,null),t.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),te)for(let Se=0;Se<v.length;Se++){t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+Se,n.RENDERBUFFER,_e.__webglColorRenderbuffer[Se]);let Ie=i.get(v[Se]).__webglTexture;t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+Se,n.TEXTURE_2D,Ie,0)}t.bindFramebuffer(n.DRAW_FRAMEBUFFER,_e.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.resolveDepthBuffer===!1&&c){let v=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,[v])}}}function fe(R){return Math.min(s.maxSamples,R.samples)}function me(R){let v=i.get(R);return R.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&v.__useRenderToTexture!==!1}function D(R){let v=o.render.frame;h.get(R)!==v&&(h.set(R,v),R.update())}function Me(R,v){let U=R.colorSpace,B=R.format,k=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||U!==ao&&U!==On&&(ht.getTransfer(U)===pt?(B!==Si||k!==ci)&&$e("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Ze("WebGLTextures: Unsupported texture color space:",U)),v}function Ve(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(l.width=R.naturalWidth||R.width,l.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(l.width=R.displayWidth,l.height=R.displayHeight):(l.width=R.width,l.height=R.height),l}this.allocateTextureUnit=Y,this.resetTextureUnits=V,this.getTextureUnits=q,this.setTextureUnits=N,this.setTexture2D=ne,this.setTexture2DArray=ie,this.setTexture3D=ge,this.setTextureCube=ue,this.rebindTextures=H,this.setupRenderTarget=Q,this.updateRenderTargetMipmap=W,this.updateMultisampleRenderTarget=ce,this.setupDepthRenderbuffer=O,this.setupFrameBufferTexture=ke,this.useMultisampledRTT=me,this.isReversedDepthBuffer=function(){return t.buffers.depth.getReversed()}}function iM(n,e){function t(i,s=On){let r,o=ht.getTransfer(s);if(i===ci)return n.UNSIGNED_BYTE;if(i===tc)return n.UNSIGNED_SHORT_4_4_4_4;if(i===ic)return n.UNSIGNED_SHORT_5_5_5_1;if(i===Mu)return n.UNSIGNED_INT_5_9_9_9_REV;if(i===Su)return n.UNSIGNED_INT_10F_11F_11F_REV;if(i===vu)return n.BYTE;if(i===yu)return n.SHORT;if(i===Dr)return n.UNSIGNED_SHORT;if(i===ec)return n.INT;if(i===tn)return n.UNSIGNED_INT;if(i===Hi)return n.FLOAT;if(i===ii)return n.HALF_FLOAT;if(i===bu)return n.ALPHA;if(i===Eu)return n.RGB;if(i===Si)return n.RGBA;if(i===fn)return n.DEPTH_COMPONENT;if(i===_n)return n.DEPTH_STENCIL;if(i===nc)return n.RED;if(i===sc)return n.RED_INTEGER;if(i===ms)return n.RG;if(i===rc)return n.RG_INTEGER;if(i===oc)return n.RGBA_INTEGER;if(i===ia||i===na||i===sa||i===ra)if(o===pt)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(i===ia)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===na)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===sa)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===ra)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(i===ia)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===na)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===sa)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===ra)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===ac||i===lc||i===cc||i===hc)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(i===ac)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===lc)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===cc)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===hc)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===uc||i===dc||i===fc||i===pc||i===mc||i===oa||i===gc)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(i===uc||i===dc)return o===pt?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(i===fc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC;if(i===pc)return r.COMPRESSED_R11_EAC;if(i===mc)return r.COMPRESSED_SIGNED_R11_EAC;if(i===oa)return r.COMPRESSED_RG11_EAC;if(i===gc)return r.COMPRESSED_SIGNED_RG11_EAC}else return null;if(i===_c||i===xc||i===vc||i===yc||i===Mc||i===Sc||i===bc||i===Ec||i===wc||i===Tc||i===Ac||i===Rc||i===Cc||i===Pc)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(i===_c)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===xc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===vc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===yc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===Mc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===Sc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===bc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===Ec)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===wc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===Tc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===Ac)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===Rc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===Cc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===Pc)return o===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===Ic||i===Dc||i===Lc)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(i===Ic)return o===pt?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===Dc)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===Lc)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===Uc||i===Nc||i===aa||i===Fc)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(i===Uc)return r.COMPRESSED_RED_RGTC1_EXT;if(i===Nc)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===aa)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===Fc)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===ps?n.UNSIGNED_INT_24_8:n[i]!==void 0?n[i]:null}return{convert:t}}var nM=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,sM=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`,Ju=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){let i=new Mo(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=i}}getMesh(e){if(this.texture!==null&&this.mesh===null){let t=e.cameras[0].viewport,i=new bt({vertexShader:nM,fragmentShader:sM,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new Ke(new ki(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},ju=class extends Ji{constructor(e,t){super();let i=this,s=null,r=1,o=null,a="local-floor",c=1,l=null,h=null,u=null,d=null,f=null,g=null,x=typeof XRWebGLBinding<"u",p=new Ju,m={},M=t.getContextAttributes(),b=null,y=null,T=[],S=[],A=new $,_=null,E=new Qt;E.viewport=new mt;let C=new Qt;C.viewport=new mt;let I=[E,C],L=new Yl,V=null,q=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(j){let he=T[j];return he===void 0&&(he=new Mr,T[j]=he),he.getTargetRaySpace()},this.getControllerGrip=function(j){let he=T[j];return he===void 0&&(he=new Mr,T[j]=he),he.getGripSpace()},this.getHand=function(j){let he=T[j];return he===void 0&&(he=new Mr,T[j]=he),he.getHandSpace()};function N(j){let he=S.indexOf(j.inputSource);if(he===-1)return;let le=T[he];le!==void 0&&(le.update(j.inputSource,j.frame,l||o),le.dispatchEvent({type:j.type,data:j.inputSource}))}function Y(){s.removeEventListener("select",N),s.removeEventListener("selectstart",N),s.removeEventListener("selectend",N),s.removeEventListener("squeeze",N),s.removeEventListener("squeezestart",N),s.removeEventListener("squeezeend",N),s.removeEventListener("end",Y),s.removeEventListener("inputsourceschange",X);for(let j=0;j<T.length;j++){let he=S[j];he!==null&&(S[j]=null,T[j].disconnect(he))}V=null,q=null,p.reset();for(let j in m)delete m[j];e.setRenderTarget(b),f=null,d=null,u=null,s=null,y=null,Xe.stop(),i.isPresenting=!1,e.setPixelRatio(_),e.setSize(A.width,A.height,!1),i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(j){r=j,i.isPresenting===!0&&$e("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(j){a=j,i.isPresenting===!0&&$e("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||o},this.setReferenceSpace=function(j){l=j},this.getBaseLayer=function(){return d!==null?d:f},this.getBinding=function(){return u===null&&x&&(u=new XRWebGLBinding(s,t)),u},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(j){if(s=j,s!==null){if(b=e.getRenderTarget(),s.addEventListener("select",N),s.addEventListener("selectstart",N),s.addEventListener("selectend",N),s.addEventListener("squeeze",N),s.addEventListener("squeezestart",N),s.addEventListener("squeezeend",N),s.addEventListener("end",Y),s.addEventListener("inputsourceschange",X),M.xrCompatible!==!0&&await t.makeXRCompatible(),_=e.getPixelRatio(),e.getSize(A),x&&"createProjectionLayer"in XRWebGLBinding.prototype){let le=null,Ae=null,Fe=null;M.depth&&(Fe=M.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,le=M.stencil?_n:fn,Ae=M.stencil?ps:tn);let ke={colorFormat:t.RGBA8,depthFormat:Fe,scaleFactor:r};u=this.getBinding(),d=u.createProjectionLayer(ke),s.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),y=new Ht(d.textureWidth,d.textureHeight,{format:Si,type:ci,depthTexture:new ji(d.textureWidth,d.textureHeight,Ae,void 0,void 0,void 0,void 0,void 0,void 0,le),stencilBuffer:M.stencil,colorSpace:e.outputColorSpace,samples:M.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{let le={antialias:M.antialias,alpha:!0,depth:M.depth,stencil:M.stencil,framebufferScaleFactor:r};f=new XRWebGLLayer(s,t,le),s.updateRenderState({baseLayer:f}),e.setPixelRatio(1),e.setSize(f.framebufferWidth,f.framebufferHeight,!1),y=new Ht(f.framebufferWidth,f.framebufferHeight,{format:Si,type:ci,colorSpace:e.outputColorSpace,stencilBuffer:M.stencil,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}y.isXRRenderTarget=!0,this.setFoveation(c),l=null,o=await s.requestReferenceSpace(a),Xe.setContext(s),Xe.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return p.getDepthTexture()};function X(j){for(let he=0;he<j.removed.length;he++){let le=j.removed[he],Ae=S.indexOf(le);Ae>=0&&(S[Ae]=null,T[Ae].disconnect(le))}for(let he=0;he<j.added.length;he++){let le=j.added[he],Ae=S.indexOf(le);if(Ae===-1){for(let ke=0;ke<T.length;ke++)if(ke>=S.length){S.push(le),Ae=ke;break}else if(S[ke]===null){S[ke]=le,Ae=ke;break}if(Ae===-1)break}let Fe=T[Ae];Fe&&Fe.connect(le)}}let ne=new P,ie=new P;function ge(j,he,le){ne.setFromMatrixPosition(he.matrixWorld),ie.setFromMatrixPosition(le.matrixWorld);let Ae=ne.distanceTo(ie),Fe=he.projectionMatrix.elements,ke=le.projectionMatrix.elements,ae=Fe[14]/(Fe[10]-1),ee=Fe[14]/(Fe[10]+1),O=(Fe[9]+1)/Fe[5],H=(Fe[9]-1)/Fe[5],Q=(Fe[8]-1)/Fe[0],W=(ke[8]+1)/ke[0],G=ae*Q,se=ae*W,ce=Ae/(-Q+W),fe=ce*-Q;if(he.matrixWorld.decompose(j.position,j.quaternion,j.scale),j.translateX(fe),j.translateZ(ce),j.matrixWorld.compose(j.position,j.quaternion,j.scale),j.matrixWorldInverse.copy(j.matrixWorld).invert(),Fe[10]===-1)j.projectionMatrix.copy(he.projectionMatrix),j.projectionMatrixInverse.copy(he.projectionMatrixInverse);else{let me=ae+ce,D=ee+ce,Me=G-fe,Ve=se+(Ae-fe),R=O*ee/D*me,v=H*ee/D*me;j.projectionMatrix.makePerspective(Me,Ve,R,v,me,D),j.projectionMatrixInverse.copy(j.projectionMatrix).invert()}}function ue(j,he){he===null?j.matrixWorld.copy(j.matrix):j.matrixWorld.multiplyMatrices(he.matrixWorld,j.matrix),j.matrixWorldInverse.copy(j.matrixWorld).invert()}this.updateCamera=function(j){if(s===null)return;let he=j.near,le=j.far;p.texture!==null&&(p.depthNear>0&&(he=p.depthNear),p.depthFar>0&&(le=p.depthFar)),L.near=C.near=E.near=he,L.far=C.far=E.far=le,(V!==L.near||q!==L.far)&&(s.updateRenderState({depthNear:L.near,depthFar:L.far}),V=L.near,q=L.far),L.layers.mask=j.layers.mask|6,E.layers.mask=L.layers.mask&-5,C.layers.mask=L.layers.mask&-3;let Ae=j.parent,Fe=L.cameras;ue(L,Ae);for(let ke=0;ke<Fe.length;ke++)ue(Fe[ke],Ae);Fe.length===2?ge(L,E,C):L.projectionMatrix.copy(E.projectionMatrix),xe(j,L,Ae)};function xe(j,he,le){le===null?j.matrix.copy(he.matrixWorld):(j.matrix.copy(le.matrixWorld),j.matrix.invert(),j.matrix.multiply(he.matrixWorld)),j.matrix.decompose(j.position,j.quaternion,j.scale),j.updateMatrixWorld(!0),j.projectionMatrix.copy(he.projectionMatrix),j.projectionMatrixInverse.copy(he.projectionMatrixInverse),j.isPerspectiveCamera&&(j.fov=xr*2*Math.atan(1/j.projectionMatrix.elements[5]),j.zoom=1)}this.getCamera=function(){return L},this.getFoveation=function(){if(!(d===null&&f===null))return c},this.setFoveation=function(j){c=j,d!==null&&(d.fixedFoveation=j),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=j)},this.hasDepthSensing=function(){return p.texture!==null},this.getDepthSensingMesh=function(){return p.getMesh(L)},this.getCameraTexture=function(j){return m[j]};let Ne=null;function st(j,he){if(h=he.getViewerPose(l||o),g=he,h!==null){let le=h.views;f!==null&&(e.setRenderTargetFramebuffer(y,f.framebuffer),e.setRenderTarget(y));let Ae=!1;le.length!==L.cameras.length&&(L.cameras.length=0,Ae=!0);for(let ee=0;ee<le.length;ee++){let O=le[ee],H=null;if(f!==null)H=f.getViewport(O);else{let W=u.getViewSubImage(d,O);H=W.viewport,ee===0&&(e.setRenderTargetTextures(y,W.colorTexture,W.depthStencilTexture),e.setRenderTarget(y))}let Q=I[ee];Q===void 0&&(Q=new Qt,Q.layers.enable(ee),Q.viewport=new mt,I[ee]=Q),Q.matrix.fromArray(O.transform.matrix),Q.matrix.decompose(Q.position,Q.quaternion,Q.scale),Q.projectionMatrix.fromArray(O.projectionMatrix),Q.projectionMatrixInverse.copy(Q.projectionMatrix).invert(),Q.viewport.set(H.x,H.y,H.width,H.height),ee===0&&(L.matrix.copy(Q.matrix),L.matrix.decompose(L.position,L.quaternion,L.scale)),Ae===!0&&L.cameras.push(Q)}let Fe=s.enabledFeatures;if(Fe&&Fe.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&x){u=i.getBinding();let ee=u.getDepthInformation(le[0]);ee&&ee.isValid&&ee.texture&&p.init(ee,s.renderState)}if(Fe&&Fe.includes("camera-access")&&x){e.state.unbindTexture(),u=i.getBinding();for(let ee=0;ee<le.length;ee++){let O=le[ee].camera;if(O){let H=m[O];H||(H=new Mo,m[O]=H);let Q=u.getCameraImage(O);H.sourceTexture=Q}}}}for(let le=0;le<T.length;le++){let Ae=S[le],Fe=T[le];Ae!==null&&Fe!==void 0&&Fe.update(Ae,he,l||o)}Ne&&Ne(j,he),he.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:he}),g=null}let Xe=new Lp;Xe.setAnimationLoop(st),this.setAnimationLoop=function(j){Ne=j},this.dispose=function(){}}},rM=new rt,zp=new tt;zp.set(-1,0,0,0,1,0,0,0,1);function oM(n,e){function t(p,m){p.matrixAutoUpdate===!0&&p.updateMatrix(),m.value.copy(p.matrix)}function i(p,m){m.color.getRGB(p.fogColor.value,Ru(n)),m.isFog?(p.fogNear.value=m.near,p.fogFar.value=m.far):m.isFogExp2&&(p.fogDensity.value=m.density)}function s(p,m,M,b,y){m.isNodeMaterial?m.uniformsNeedUpdate=!1:m.isMeshBasicMaterial?r(p,m):m.isMeshLambertMaterial?(r(p,m),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)):m.isMeshToonMaterial?(r(p,m),u(p,m)):m.isMeshPhongMaterial?(r(p,m),h(p,m),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)):m.isMeshStandardMaterial?(r(p,m),d(p,m),m.isMeshPhysicalMaterial&&f(p,m,y)):m.isMeshMatcapMaterial?(r(p,m),g(p,m)):m.isMeshDepthMaterial?r(p,m):m.isMeshDistanceMaterial?(r(p,m),x(p,m)):m.isMeshNormalMaterial?r(p,m):m.isLineBasicMaterial?(o(p,m),m.isLineDashedMaterial&&a(p,m)):m.isPointsMaterial?c(p,m,M,b):m.isSpriteMaterial?l(p,m):m.isShadowMaterial?(p.color.value.copy(m.color),p.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(p,m){p.opacity.value=m.opacity,m.color&&p.diffuse.value.copy(m.color),m.emissive&&p.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(p.map.value=m.map,t(m.map,p.mapTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.bumpMap&&(p.bumpMap.value=m.bumpMap,t(m.bumpMap,p.bumpMapTransform),p.bumpScale.value=m.bumpScale,m.side===ti&&(p.bumpScale.value*=-1)),m.normalMap&&(p.normalMap.value=m.normalMap,t(m.normalMap,p.normalMapTransform),p.normalScale.value.copy(m.normalScale),m.side===ti&&p.normalScale.value.negate()),m.displacementMap&&(p.displacementMap.value=m.displacementMap,t(m.displacementMap,p.displacementMapTransform),p.displacementScale.value=m.displacementScale,p.displacementBias.value=m.displacementBias),m.emissiveMap&&(p.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,p.emissiveMapTransform)),m.specularMap&&(p.specularMap.value=m.specularMap,t(m.specularMap,p.specularMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest);let M=e.get(m),b=M.envMap,y=M.envMapRotation;b&&(p.envMap.value=b,p.envMapRotation.value.setFromMatrix4(rM.makeRotationFromEuler(y)).transpose(),b.isCubeTexture&&b.isRenderTargetTexture===!1&&p.envMapRotation.value.premultiply(zp),p.reflectivity.value=m.reflectivity,p.ior.value=m.ior,p.refractionRatio.value=m.refractionRatio),m.lightMap&&(p.lightMap.value=m.lightMap,p.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,p.lightMapTransform)),m.aoMap&&(p.aoMap.value=m.aoMap,p.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,p.aoMapTransform))}function o(p,m){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,m.map&&(p.map.value=m.map,t(m.map,p.mapTransform))}function a(p,m){p.dashSize.value=m.dashSize,p.totalSize.value=m.dashSize+m.gapSize,p.scale.value=m.scale}function c(p,m,M,b){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,p.size.value=m.size*M,p.scale.value=b*.5,m.map&&(p.map.value=m.map,t(m.map,p.uvTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest)}function l(p,m){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,p.rotation.value=m.rotation,m.map&&(p.map.value=m.map,t(m.map,p.mapTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest)}function h(p,m){p.specular.value.copy(m.specular),p.shininess.value=Math.max(m.shininess,1e-4)}function u(p,m){m.gradientMap&&(p.gradientMap.value=m.gradientMap)}function d(p,m){p.metalness.value=m.metalness,m.metalnessMap&&(p.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,p.metalnessMapTransform)),p.roughness.value=m.roughness,m.roughnessMap&&(p.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,p.roughnessMapTransform)),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)}function f(p,m,M){p.ior.value=m.ior,m.sheen>0&&(p.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),p.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(p.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,p.sheenColorMapTransform)),m.sheenRoughnessMap&&(p.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,p.sheenRoughnessMapTransform))),m.clearcoat>0&&(p.clearcoat.value=m.clearcoat,p.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(p.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,p.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(p.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,p.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(p.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,p.clearcoatNormalMapTransform),p.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===ti&&p.clearcoatNormalScale.value.negate())),m.dispersion>0&&(p.dispersion.value=m.dispersion),m.iridescence>0&&(p.iridescence.value=m.iridescence,p.iridescenceIOR.value=m.iridescenceIOR,p.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],p.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(p.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,p.iridescenceMapTransform)),m.iridescenceThicknessMap&&(p.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,p.iridescenceThicknessMapTransform))),m.transmission>0&&(p.transmission.value=m.transmission,p.transmissionSamplerMap.value=M.texture,p.transmissionSamplerSize.value.set(M.width,M.height),m.transmissionMap&&(p.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,p.transmissionMapTransform)),p.thickness.value=m.thickness,m.thicknessMap&&(p.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,p.thicknessMapTransform)),p.attenuationDistance.value=m.attenuationDistance,p.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(p.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(p.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,p.anisotropyMapTransform))),p.specularIntensity.value=m.specularIntensity,p.specularColor.value.copy(m.specularColor),m.specularColorMap&&(p.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,p.specularColorMapTransform)),m.specularIntensityMap&&(p.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,p.specularIntensityMapTransform))}function g(p,m){m.matcap&&(p.matcap.value=m.matcap)}function x(p,m){let M=e.get(m).light;p.referencePosition.value.setFromMatrixPosition(M.matrixWorld),p.nearDistance.value=M.shadow.camera.near,p.farDistance.value=M.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:s}}function aM(n,e,t,i){let s={},r={},o=[],a=n.getParameter(n.MAX_UNIFORM_BUFFER_BINDINGS);function c(y,T){let S=T.program;i.uniformBlockBinding(y,S)}function l(y,T){let S=s[y.id];S===void 0&&(p(y),S=h(y),s[y.id]=S,y.addEventListener("dispose",M));let A=T.program;i.updateUBOMapping(y,A);let _=e.render.frame;r[y.id]!==_&&(d(y),r[y.id]=_)}function h(y){let T=u();y.__bindingPointIndex=T;let S=n.createBuffer(),A=y.__size,_=y.usage;return n.bindBuffer(n.UNIFORM_BUFFER,S),n.bufferData(n.UNIFORM_BUFFER,A,_),n.bindBuffer(n.UNIFORM_BUFFER,null),n.bindBufferBase(n.UNIFORM_BUFFER,T,S),S}function u(){for(let y=0;y<a;y++)if(o.indexOf(y)===-1)return o.push(y),y;return Ze("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(y){let T=s[y.id],S=y.uniforms,A=y.__cache;n.bindBuffer(n.UNIFORM_BUFFER,T);for(let _=0,E=S.length;_<E;_++){let C=S[_];if(Array.isArray(C))for(let I=0,L=C.length;I<L;I++)f(C[I],_,I,A);else f(C,_,0,A)}n.bindBuffer(n.UNIFORM_BUFFER,null)}function f(y,T,S,A){if(x(y,T,S,A)===!0){let _=y.__offset,E=y.value;if(Array.isArray(E)){let C=0;for(let I=0;I<E.length;I++){let L=E[I],V=m(L);g(L,y.__data,C),typeof L!="number"&&typeof L!="boolean"&&!L.isMatrix3&&!ArrayBuffer.isView(L)&&(C+=V.storage/Float32Array.BYTES_PER_ELEMENT)}}else g(E,y.__data,0);n.bufferSubData(n.UNIFORM_BUFFER,_,y.__data)}}function g(y,T,S){typeof y=="number"||typeof y=="boolean"?T[0]=y:y.isMatrix3?(T[0]=y.elements[0],T[1]=y.elements[1],T[2]=y.elements[2],T[3]=0,T[4]=y.elements[3],T[5]=y.elements[4],T[6]=y.elements[5],T[7]=0,T[8]=y.elements[6],T[9]=y.elements[7],T[10]=y.elements[8],T[11]=0):ArrayBuffer.isView(y)?T.set(new y.constructor(y.buffer,y.byteOffset,T.length)):y.toArray(T,S)}function x(y,T,S,A){let _=y.value,E=T+"_"+S;if(A[E]===void 0)return typeof _=="number"||typeof _=="boolean"?A[E]=_:ArrayBuffer.isView(_)?A[E]=_.slice():A[E]=_.clone(),!0;{let C=A[E];if(typeof _=="number"||typeof _=="boolean"){if(C!==_)return A[E]=_,!0}else{if(ArrayBuffer.isView(_))return!0;if(C.equals(_)===!1)return C.copy(_),!0}}return!1}function p(y){let T=y.uniforms,S=0,A=16;for(let E=0,C=T.length;E<C;E++){let I=Array.isArray(T[E])?T[E]:[T[E]];for(let L=0,V=I.length;L<V;L++){let q=I[L],N=Array.isArray(q.value)?q.value:[q.value];for(let Y=0,X=N.length;Y<X;Y++){let ne=N[Y],ie=m(ne),ge=S%A,ue=ge%ie.boundary,xe=ge+ue;S+=ue,xe!==0&&A-xe<ie.storage&&(S+=A-xe),q.__data=new Float32Array(ie.storage/Float32Array.BYTES_PER_ELEMENT),q.__offset=S,S+=ie.storage}}}let _=S%A;return _>0&&(S+=A-_),y.__size=S,y.__cache={},this}function m(y){let T={boundary:0,storage:0};return typeof y=="number"||typeof y=="boolean"?(T.boundary=4,T.storage=4):y.isVector2?(T.boundary=8,T.storage=8):y.isVector3||y.isColor?(T.boundary=16,T.storage=12):y.isVector4?(T.boundary=16,T.storage=16):y.isMatrix3?(T.boundary=48,T.storage=48):y.isMatrix4?(T.boundary=64,T.storage=64):y.isTexture?$e("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(y)?(T.boundary=16,T.storage=y.byteLength):$e("WebGLRenderer: Unsupported uniform value type.",y),T}function M(y){let T=y.target;T.removeEventListener("dispose",M);let S=o.indexOf(T.__bindingPointIndex);o.splice(S,1),n.deleteBuffer(s[T.id]),delete s[T.id],delete r[T.id]}function b(){for(let y in s)n.deleteBuffer(s[y]);o=[],s={},r={}}return{bind:c,update:l,dispose:b}}var lM=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),vn=null;function cM(){return vn===null&&(vn=new Un(lM,16,16,ms,ii),vn.name="DFG_LUT",vn.minFilter=ei,vn.magFilter=ei,vn.wrapS=un,vn.wrapT=un,vn.generateMipmaps=!1,vn.needsUpdate=!0),vn}var Vc=class{constructor(e={}){let{canvas:t=ep(),context:i=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:u=!1,reversedDepthBuffer:d=!1,outputBufferType:f=ci}=e;this.isWebGLRenderer=!0;let g;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");g=i.getContextAttributes().alpha}else g=o;let x=f,p=new Set([oc,rc,sc]),m=new Set([ci,tn,Dr,ps,tc,ic]),M=new Uint32Array(4),b=new Int32Array(4),y=new P,T=null,S=null,A=[],_=[],E=null;this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=en,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let C=this,I=!1,L=null,V=null,q=null,N=null;this._outputColorSpace=Lt;let Y=0,X=0,ne=null,ie=-1,ge=null,ue=new mt,xe=new mt,Ne=null,st=new Te(0),Xe=0,j=t.width,he=t.height,le=1,Ae=null,Fe=null,ke=new mt(0,0,j,he),ae=new mt(0,0,j,he),ee=!1,O=new br,H=!1,Q=!1,W=new rt,G=new P,se=new mt,ce={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},fe=!1;function me(){return ne===null?le:1}let D=i;function Me(w,z){return t.getContext(w,z)}try{let w={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:h,failIfMajorPerformanceCaveat:u};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${"185"}`),t.addEventListener("webglcontextlost",It,!1),t.addEventListener("webglcontextrestored",Mt,!1),t.addEventListener("webglcontextcreationerror",on,!1),D===null){let z="webgl2";if(D=Me(z,w),D===null)throw Me(z)?new Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes."):new Error("THREE.WebGLRenderer: Error creating WebGL context.")}}catch(w){throw Ze("WebGLRenderer: "+w.message),w}let Ve,R,v,U,B,k,pe,_e,te,re,Se,Ie,ve,ye,Be,qe,Je,F,Ee,oe,we,Pe,de;function He(){Ve=new gv(D),Ve.init(),we=new iM(D,Ve),R=new lv(D,Ve,e,we),v=new eM(D,Ve),R.reversedDepthBuffer&&d&&v.buffers.depth.setReversed(!0),V=D.createFramebuffer(),q=D.createFramebuffer(),N=D.createFramebuffer(),U=new vv(D),B=new ky,k=new tM(D,Ve,v,B,R,we,U),pe=new mv(C),_e=new b0(D),Pe=new ov(D,_e),te=new _v(D,_e,U,Pe),re=new Mv(D,te,_e,Pe,U),F=new yv(D,R,k),Be=new cv(B),Se=new zy(C,pe,Ve,R,Pe,Be),Ie=new oM(C,B),ve=new Vy,ye=new $y(Ve),Je=new rv(C,pe,v,re,g,c),qe=new Qy(C,re,R),de=new aM(D,U,R,v),Ee=new av(D,Ve,U),oe=new xv(D,Ve,U),U.programs=Se.programs,C.capabilities=R,C.extensions=Ve,C.properties=B,C.renderLists=ve,C.shadowMap=qe,C.state=v,C.info=U}He(),x!==ci&&(E=new bv(x,t.width,t.height,a,s,r));let Oe=new ju(C,D);this.xr=Oe,this.getContext=function(){return D},this.getContextAttributes=function(){return D.getContextAttributes()},this.forceContextLoss=function(){let w=Ve.get("WEBGL_lose_context");w&&w.loseContext()},this.forceContextRestore=function(){let w=Ve.get("WEBGL_lose_context");w&&w.restoreContext()},this.getPixelRatio=function(){return le},this.setPixelRatio=function(w){w!==void 0&&(le=w,this.setSize(j,he,!1))},this.getSize=function(w){return w.set(j,he)},this.setSize=function(w,z,K=!0){if(Oe.isPresenting){$e("WebGLRenderer: Can't change size while VR device is presenting.");return}j=w,he=z,t.width=Math.floor(w*le),t.height=Math.floor(z*le),K===!0&&(t.style.width=w+"px",t.style.height=z+"px"),E!==null&&E.setSize(t.width,t.height),this.setViewport(0,0,w,z)},this.getDrawingBufferSize=function(w){return w.set(j*le,he*le).floor()},this.setDrawingBufferSize=function(w,z,K){j=w,he=z,le=K,t.width=Math.floor(w*K),t.height=Math.floor(z*K),this.setViewport(0,0,w,z)},this.setEffects=function(w){if(x===ci){Ze("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(w){for(let z=0;z<w.length;z++)if(w[z].isOutputPass===!0){$e("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}E.setEffects(w||[])},this.getCurrentViewport=function(w){return w.copy(ue)},this.getViewport=function(w){return w.copy(ke)},this.setViewport=function(w,z,K,Z){w.isVector4?ke.set(w.x,w.y,w.z,w.w):ke.set(w,z,K,Z),v.viewport(ue.copy(ke).multiplyScalar(le).round())},this.getScissor=function(w){return w.copy(ae)},this.setScissor=function(w,z,K,Z){w.isVector4?ae.set(w.x,w.y,w.z,w.w):ae.set(w,z,K,Z),v.scissor(xe.copy(ae).multiplyScalar(le).round())},this.getScissorTest=function(){return ee},this.setScissorTest=function(w){v.setScissorTest(ee=w)},this.setOpaqueSort=function(w){Ae=w},this.setTransparentSort=function(w){Fe=w},this.getClearColor=function(w){return w.copy(Je.getClearColor())},this.setClearColor=function(){Je.setClearColor(...arguments)},this.getClearAlpha=function(){return Je.getClearAlpha()},this.setClearAlpha=function(){Je.setClearAlpha(...arguments)},this.clear=function(w=!0,z=!0,K=!0){let Z=0;if(w){let J=!1;if(ne!==null){let Ce=ne.texture.format;J=p.has(Ce)}if(J){let Ce=ne.texture.type,Ue=m.has(Ce),Re=Je.getClearColor(),ze=Je.getClearAlpha(),Ge=Re.r,nt=Re.g,lt=Re.b;Ue?(M[0]=Ge,M[1]=nt,M[2]=lt,M[3]=ze,D.clearBufferuiv(D.COLOR,0,M)):(b[0]=Ge,b[1]=nt,b[2]=lt,b[3]=ze,D.clearBufferiv(D.COLOR,0,b))}else Z|=D.COLOR_BUFFER_BIT}z&&(Z|=D.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),K&&(Z|=D.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),Z!==0&&D.clear(Z)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(w){w.setRenderer(this),L=w},this.dispose=function(){t.removeEventListener("webglcontextlost",It,!1),t.removeEventListener("webglcontextrestored",Mt,!1),t.removeEventListener("webglcontextcreationerror",on,!1),Je.dispose(),ve.dispose(),ye.dispose(),B.dispose(),pe.dispose(),re.dispose(),Pe.dispose(),de.dispose(),Se.dispose(),Oe.dispose(),Oe.removeEventListener("sessionstart",Pd),Oe.removeEventListener("sessionend",Id),Ss.stop()};function It(w){w.preventDefault(),ho("WebGLRenderer: Context Lost."),I=!0}function Mt(){ho("WebGLRenderer: Context Restored."),I=!1;let w=U.autoReset,z=qe.enabled,K=qe.autoUpdate,Z=qe.needsUpdate,J=qe.type;He(),U.autoReset=w,qe.enabled=z,qe.autoUpdate=K,qe.needsUpdate=Z,qe.type=J}function on(w){Ze("WebGLRenderer: A WebGL context could not be created. Reason: ",w.statusMessage)}function an(w){let z=w.target;z.removeEventListener("dispose",an),Xm(z)}function Xm(w){qm(w),B.remove(w)}function qm(w){let z=B.get(w).programs;z!==void 0&&(z.forEach(function(K){Se.releaseProgram(K)}),w.isShaderMaterial&&Se.releaseShaderCache(w))}this.renderBufferDirect=function(w,z,K,Z,J,Ce){z===null&&(z=ce);let Ue=J.isMesh&&J.matrixWorld.determinantAffine()<0,Re=Zm(w,z,K,Z,J);v.setMaterial(Z,Ue);let ze=K.index,Ge=1;if(Z.wireframe===!0){if(ze=te.getWireframeAttribute(K),ze===void 0)return;Ge=2}let nt=K.drawRange,lt=K.attributes.position,Ye=nt.start*Ge,_t=(nt.start+nt.count)*Ge;Ce!==null&&(Ye=Math.max(Ye,Ce.start*Ge),_t=Math.min(_t,(Ce.start+Ce.count)*Ge)),ze!==null?(Ye=Math.max(Ye,0),_t=Math.min(_t,ze.count)):lt!=null&&(Ye=Math.max(Ye,0),_t=Math.min(_t,lt.count));let Nt=_t-Ye;if(Nt<0||Nt===1/0)return;Pe.setup(J,Z,Re,K,ze);let Dt,vt=Ee;if(ze!==null&&(Dt=_e.get(ze),vt=oe,vt.setIndex(Dt)),J.isMesh)Z.wireframe===!0?(v.setLineWidth(Z.wireframeLinewidth*me()),vt.setMode(D.LINES)):vt.setMode(D.TRIANGLES);else if(J.isLine){let oi=Z.linewidth;oi===void 0&&(oi=1),v.setLineWidth(oi*me()),J.isLineSegments?vt.setMode(D.LINES):J.isLineLoop?vt.setMode(D.LINE_LOOP):vt.setMode(D.LINE_STRIP)}else J.isPoints?vt.setMode(D.POINTS):J.isSprite&&vt.setMode(D.TRIANGLES);if(J.isBatchedMesh)if(Ve.get("WEBGL_multi_draw"))vt.renderMultiDraw(J._multiDrawStarts,J._multiDrawCounts,J._multiDrawCount);else{let oi=J._multiDrawStarts,Le=J._multiDrawCounts,Ti=J._multiDrawCount,dt=ze?_e.get(ze).bytesPerElement:1,Fi=B.get(Z).currentProgram.getUniforms();for(let ln=0;ln<Ti;ln++)Fi.setValue(D,"_gl_DrawID",ln),vt.render(oi[ln]/dt,Le[ln])}else if(J.isInstancedMesh)vt.renderInstances(Ye,Nt,J.count);else if(K.isInstancedBufferGeometry){let oi=K._maxInstanceCount!==void 0?K._maxInstanceCount:1/0,Le=Math.min(K.instanceCount,oi);vt.renderInstances(Ye,Nt,Le)}else vt.render(Ye,Nt)};function Cd(w,z,K){w.transparent===!0&&w.side===Mi&&w.forceSinglePass===!1?(w.side=ti,w.needsUpdate=!0,wa(w,z,K),w.side=Zi,w.needsUpdate=!0,wa(w,z,K),w.side=Mi):wa(w,z,K)}this.compile=function(w,z,K=null){K===null&&(K=w),S=ye.get(K),S.init(z),_.push(S),K.traverseVisible(function(J){J.isLight&&J.layers.test(z.layers)&&(S.pushLight(J),J.castShadow&&S.pushShadow(J))}),w!==K&&w.traverseVisible(function(J){J.isLight&&J.layers.test(z.layers)&&(S.pushLight(J),J.castShadow&&S.pushShadow(J))}),S.setupLights();let Z=new Set;return w.traverse(function(J){if(!(J.isMesh||J.isPoints||J.isLine||J.isSprite))return;let Ce=J.material;if(Ce)if(Array.isArray(Ce))for(let Ue=0;Ue<Ce.length;Ue++){let Re=Ce[Ue];Cd(Re,K,J),Z.add(Re)}else Cd(Ce,K,J),Z.add(Ce)}),S=_.pop(),Z},this.compileAsync=function(w,z,K=null){let Z=this.compile(w,z,K);return new Promise(J=>{function Ce(){if(Z.forEach(function(Ue){B.get(Ue).currentProgram.isReady()&&Z.delete(Ue)}),Z.size===0){J(w);return}setTimeout(Ce,10)}Ve.get("KHR_parallel_shader_compile")!==null?Ce():setTimeout(Ce,10)})};let Sh=null;function Ym(w){Sh&&Sh(w)}function Pd(){Ss.stop()}function Id(){Ss.start()}let Ss=new Lp;Ss.setAnimationLoop(Ym),typeof self<"u"&&Ss.setContext(self),this.setAnimationLoop=function(w){Sh=w,Oe.setAnimationLoop(w),w===null?Ss.stop():Ss.start()},Oe.addEventListener("sessionstart",Pd),Oe.addEventListener("sessionend",Id),this.render=function(w,z){if(z!==void 0&&z.isCamera!==!0){Ze("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(I===!0)return;L!==null&&L.renderStart(w,z);let K=Oe.enabled===!0&&Oe.isPresenting===!0,Z=E!==null&&(ne===null||K)&&E.begin(C,ne);if(w.matrixWorldAutoUpdate===!0&&w.updateMatrixWorld(),z.parent===null&&z.matrixWorldAutoUpdate===!0&&z.updateMatrixWorld(),Oe.enabled===!0&&Oe.isPresenting===!0&&(E===null||E.isCompositing()===!1)&&(Oe.cameraAutoUpdate===!0&&Oe.updateCamera(z),z=Oe.getCamera()),w.isScene===!0&&w.onBeforeRender(C,w,z,ne),S=ye.get(w,_.length),S.init(z),S.state.textureUnits=k.getTextureUnits(),_.push(S),W.multiplyMatrices(z.projectionMatrix,z.matrixWorldInverse),O.setFromProjectionMatrix(W,$i,z.reversedDepth),Q=this.localClippingEnabled,H=Be.init(this.clippingPlanes,Q),T=ve.get(w,A.length),T.init(),A.push(T),Oe.enabled===!0&&Oe.isPresenting===!0){let Ue=C.xr.getDepthSensingMesh();Ue!==null&&bh(Ue,z,-1/0,C.sortObjects)}bh(w,z,0,C.sortObjects),T.finish(),C.sortObjects===!0&&T.sort(Ae,Fe,z.reversedDepth),fe=Oe.enabled===!1||Oe.isPresenting===!1||Oe.hasDepthSensing()===!1,fe&&Je.addToRenderList(T,w),this.info.render.frame++,this.info.autoReset===!0&&this.info.reset(),H===!0&&Be.beginShadows();let J=S.state.shadowsArray;if(qe.render(J,w,z),H===!0&&Be.endShadows(),(Z&&E.hasRenderPass())===!1){let Ue=T.opaque,Re=T.transmissive;if(S.setupLights(),z.isArrayCamera){let ze=z.cameras;if(Re.length>0)for(let Ge=0,nt=ze.length;Ge<nt;Ge++){let lt=ze[Ge];Ld(Ue,Re,w,lt)}fe&&Je.render(w);for(let Ge=0,nt=ze.length;Ge<nt;Ge++){let lt=ze[Ge];Dd(T,w,lt,lt.viewport)}}else Re.length>0&&Ld(Ue,Re,w,z),fe&&Je.render(w),Dd(T,w,z)}ne!==null&&X===0&&(k.updateMultisampleRenderTarget(ne),k.updateRenderTargetMipmap(ne)),Z&&E.end(C),w.isScene===!0&&w.onAfterRender(C,w,z),Pe.resetDefaultState(),ie=-1,ge=null,_.pop(),_.length>0?(S=_[_.length-1],k.setTextureUnits(S.state.textureUnits),H===!0&&Be.setGlobalState(C.clippingPlanes,S.state.camera)):S=null,A.pop(),A.length>0?T=A[A.length-1]:T=null,L!==null&&L.renderEnd()};function bh(w,z,K,Z){if(w.visible===!1)return;if(w.layers.test(z.layers)){if(w.isGroup)K=w.renderOrder;else if(w.isLOD)w.autoUpdate===!0&&w.update(z);else if(w.isLightProbeGrid)S.pushLightProbeGrid(w);else if(w.isLight)S.pushLight(w),w.castShadow&&S.pushShadow(w);else if(w.isSprite){if(!w.frustumCulled||O.intersectsSprite(w)){Z&&se.setFromMatrixPosition(w.matrixWorld).applyMatrix4(W);let Ue=re.update(w),Re=w.material;Re.visible&&T.push(w,Ue,Re,K,se.z,null)}}else if((w.isMesh||w.isLine||w.isPoints)&&(!w.frustumCulled||O.intersectsObject(w))){let Ue=re.update(w),Re=w.material;if(Z&&(w.boundingSphere!==void 0?(w.boundingSphere===null&&w.computeBoundingSphere(),se.copy(w.boundingSphere.center)):(Ue.boundingSphere===null&&Ue.computeBoundingSphere(),se.copy(Ue.boundingSphere.center)),se.applyMatrix4(w.matrixWorld).applyMatrix4(W)),Array.isArray(Re)){let ze=Ue.groups;for(let Ge=0,nt=ze.length;Ge<nt;Ge++){let lt=ze[Ge],Ye=Re[lt.materialIndex];Ye&&Ye.visible&&T.push(w,Ue,Ye,K,se.z,lt)}}else Re.visible&&T.push(w,Ue,Re,K,se.z,null)}}let Ce=w.children;for(let Ue=0,Re=Ce.length;Ue<Re;Ue++)bh(Ce[Ue],z,K,Z)}function Dd(w,z,K,Z){let{opaque:J,transmissive:Ce,transparent:Ue}=w;S.setupLightsView(K),H===!0&&Be.setGlobalState(C.clippingPlanes,K),Z&&v.viewport(ue.copy(Z)),J.length>0&&Ea(J,z,K),Ce.length>0&&Ea(Ce,z,K),Ue.length>0&&Ea(Ue,z,K),v.buffers.depth.setTest(!0),v.buffers.depth.setMask(!0),v.buffers.color.setMask(!0),v.setPolygonOffset(!1)}function Ld(w,z,K,Z){if((K.isScene===!0?K.overrideMaterial:null)!==null)return;if(S.state.transmissionRenderTarget[Z.id]===void 0){let Ye=Ve.has("EXT_color_buffer_half_float")||Ve.has("EXT_color_buffer_float");S.state.transmissionRenderTarget[Z.id]=new Ht(1,1,{generateMipmaps:!0,type:Ye?ii:ci,minFilter:fs,samples:Math.max(4,R.samples),stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:ht.workingColorSpace})}let Ce=S.state.transmissionRenderTarget[Z.id],Ue=Z.viewport||ue;Ce.setSize(Ue.z*C.transmissionResolutionScale,Ue.w*C.transmissionResolutionScale);let Re=C.getRenderTarget(),ze=C.getActiveCubeFace(),Ge=C.getActiveMipmapLevel();C.setRenderTarget(Ce),C.getClearColor(st),Xe=C.getClearAlpha(),Xe<1&&C.setClearColor(16777215,.5),C.clear(),fe&&Je.render(K);let nt=C.toneMapping;C.toneMapping=en;let lt=Z.viewport;if(Z.viewport!==void 0&&(Z.viewport=void 0),S.setupLightsView(Z),H===!0&&Be.setGlobalState(C.clippingPlanes,Z),Ea(w,K,Z),k.updateMultisampleRenderTarget(Ce),k.updateRenderTargetMipmap(Ce),Ve.has("WEBGL_multisampled_render_to_texture")===!1){let Ye=!1;for(let _t=0,Nt=z.length;_t<Nt;_t++){let Dt=z[_t],{object:vt,geometry:oi,material:Le,group:Ti}=Dt;if(Le.side===Mi&&vt.layers.test(Z.layers)){let dt=Le.side;Le.side=ti,Le.needsUpdate=!0,Ud(vt,K,Z,oi,Le,Ti),Le.side=dt,Le.needsUpdate=!0,Ye=!0}}Ye===!0&&(k.updateMultisampleRenderTarget(Ce),k.updateRenderTargetMipmap(Ce))}C.setRenderTarget(Re,ze,Ge),C.setClearColor(st,Xe),lt!==void 0&&(Z.viewport=lt),C.toneMapping=nt}function Ea(w,z,K){let Z=z.isScene===!0?z.overrideMaterial:null;for(let J=0,Ce=w.length;J<Ce;J++){let Ue=w[J],{object:Re,geometry:ze,group:Ge}=Ue,nt=Ue.material;nt.allowOverride===!0&&Z!==null&&(nt=Z),Re.layers.test(K.layers)&&Ud(Re,z,K,ze,nt,Ge)}}function Ud(w,z,K,Z,J,Ce){w.onBeforeRender(C,z,K,Z,J,Ce),w.modelViewMatrix.multiplyMatrices(K.matrixWorldInverse,w.matrixWorld),w.normalMatrix.getNormalMatrix(w.modelViewMatrix),J.onBeforeRender(C,z,K,Z,w,Ce),J.transparent===!0&&J.side===Mi&&J.forceSinglePass===!1?(J.side=ti,J.needsUpdate=!0,C.renderBufferDirect(K,z,Z,J,w,Ce),J.side=Zi,J.needsUpdate=!0,C.renderBufferDirect(K,z,Z,J,w,Ce),J.side=Mi):C.renderBufferDirect(K,z,Z,J,w,Ce),w.onAfterRender(C,z,K,Z,J,Ce)}function wa(w,z,K){z.isScene!==!0&&(z=ce);let Z=B.get(w),J=S.state.lights,Ce=S.state.shadowsArray,Ue=J.state.version,Re=Se.getParameters(w,J.state,Ce,z,K,S.state.lightProbeGridArray),ze=Se.getProgramCacheKey(Re),Ge=Z.programs;Z.environment=w.isMeshStandardMaterial||w.isMeshLambertMaterial||w.isMeshPhongMaterial?z.environment:null,Z.fog=z.fog;let nt=w.isMeshStandardMaterial||w.isMeshLambertMaterial&&!w.envMap||w.isMeshPhongMaterial&&!w.envMap;Z.envMap=pe.get(w.envMap||Z.environment,nt),Z.envMapRotation=Z.environment!==null&&w.envMap===null?z.environmentRotation:w.envMapRotation,Ge===void 0&&(w.addEventListener("dispose",an),Ge=new Map,Z.programs=Ge);let lt=Ge.get(ze);if(lt!==void 0){if(Z.currentProgram===lt&&Z.lightsStateVersion===Ue)return Fd(w,Re),lt}else Re.uniforms=Se.getUniforms(w),L!==null&&w.isNodeMaterial&&L.build(w,K,Re),w.onBeforeCompile(Re,C),lt=Se.acquireProgram(Re,ze),Ge.set(ze,lt),Z.uniforms=Re.uniforms;let Ye=Z.uniforms;return(!w.isShaderMaterial&&!w.isRawShaderMaterial||w.clipping===!0)&&(Ye.clippingPlanes=Be.uniform),Fd(w,Re),Z.needsLights=jm(w),Z.lightsStateVersion=Ue,Z.needsLights&&(Ye.ambientLightColor.value=J.state.ambient,Ye.lightProbe.value=J.state.probe,Ye.directionalLights.value=J.state.directional,Ye.directionalLightShadows.value=J.state.directionalShadow,Ye.spotLights.value=J.state.spot,Ye.spotLightShadows.value=J.state.spotShadow,Ye.rectAreaLights.value=J.state.rectArea,Ye.ltc_1.value=J.state.rectAreaLTC1,Ye.ltc_2.value=J.state.rectAreaLTC2,Ye.pointLights.value=J.state.point,Ye.pointLightShadows.value=J.state.pointShadow,Ye.hemisphereLights.value=J.state.hemi,Ye.directionalShadowMatrix.value=J.state.directionalShadowMatrix,Ye.spotLightMatrix.value=J.state.spotLightMatrix,Ye.spotLightMap.value=J.state.spotLightMap,Ye.pointShadowMatrix.value=J.state.pointShadowMatrix),Z.lightProbeGrid=S.state.lightProbeGridArray.length>0,Z.currentProgram=lt,Z.uniformsList=null,lt}function Nd(w){if(w.uniformsList===null){let z=w.currentProgram.getUniforms();w.uniformsList=Nr.seqWithValue(z.seq,w.uniforms)}return w.uniformsList}function Fd(w,z){let K=B.get(w);K.outputColorSpace=z.outputColorSpace,K.batching=z.batching,K.batchingColor=z.batchingColor,K.instancing=z.instancing,K.instancingColor=z.instancingColor,K.instancingMorph=z.instancingMorph,K.skinning=z.skinning,K.morphTargets=z.morphTargets,K.morphNormals=z.morphNormals,K.morphColors=z.morphColors,K.morphTargetsCount=z.morphTargetsCount,K.numClippingPlanes=z.numClippingPlanes,K.numIntersection=z.numClipIntersection,K.vertexAlphas=z.vertexAlphas,K.vertexTangents=z.vertexTangents,K.toneMapping=z.toneMapping}function $m(w,z){if(w.length===0)return null;if(w.length===1)return w[0].texture!==null?w[0]:null;y.setFromMatrixPosition(z.matrixWorld);for(let K=0,Z=w.length;K<Z;K++){let J=w[K];if(J.texture!==null&&J.boundingBox.containsPoint(y))return J}return null}function Zm(w,z,K,Z,J){z.isScene!==!0&&(z=ce),k.resetTextureUnits();let Ce=z.fog,Ue=Z.isMeshStandardMaterial||Z.isMeshLambertMaterial||Z.isMeshPhongMaterial?z.environment:null,Re=ne===null?C.outputColorSpace:ne.isXRRenderTarget===!0?ne.texture.colorSpace:ht.workingColorSpace,ze=Z.isMeshStandardMaterial||Z.isMeshLambertMaterial&&!Z.envMap||Z.isMeshPhongMaterial&&!Z.envMap,Ge=pe.get(Z.envMap||Ue,ze),nt=Z.vertexColors===!0&&!!K.attributes.color&&K.attributes.color.itemSize===4,lt=!!K.attributes.tangent&&(!!Z.normalMap||Z.anisotropy>0),Ye=!!K.morphAttributes.position,_t=!!K.morphAttributes.normal,Nt=!!K.morphAttributes.color,Dt=en;Z.toneMapped&&(ne===null||ne.isXRRenderTarget===!0)&&(Dt=C.toneMapping);let vt=K.morphAttributes.position||K.morphAttributes.normal||K.morphAttributes.color,oi=vt!==void 0?vt.length:0,Le=B.get(Z),Ti=S.state.lights;if(H===!0&&(Q===!0||w!==ge)){let St=w===ge&&Z.id===ie;Be.setState(Z,w,St)}let dt=!1;Z.version===Le.__version?(Le.needsLights&&Le.lightsStateVersion!==Ti.state.version||Le.outputColorSpace!==Re||J.isBatchedMesh&&Le.batching===!1||!J.isBatchedMesh&&Le.batching===!0||J.isBatchedMesh&&Le.batchingColor===!0&&J.colorTexture===null||J.isBatchedMesh&&Le.batchingColor===!1&&J.colorTexture!==null||J.isInstancedMesh&&Le.instancing===!1||!J.isInstancedMesh&&Le.instancing===!0||J.isSkinnedMesh&&Le.skinning===!1||!J.isSkinnedMesh&&Le.skinning===!0||J.isInstancedMesh&&Le.instancingColor===!0&&J.instanceColor===null||J.isInstancedMesh&&Le.instancingColor===!1&&J.instanceColor!==null||J.isInstancedMesh&&Le.instancingMorph===!0&&J.morphTexture===null||J.isInstancedMesh&&Le.instancingMorph===!1&&J.morphTexture!==null||Le.envMap!==Ge||Z.fog===!0&&Le.fog!==Ce||Le.numClippingPlanes!==void 0&&(Le.numClippingPlanes!==Be.numPlanes||Le.numIntersection!==Be.numIntersection)||Le.vertexAlphas!==nt||Le.vertexTangents!==lt||Le.morphTargets!==Ye||Le.morphNormals!==_t||Le.morphColors!==Nt||Le.toneMapping!==Dt||Le.morphTargetsCount!==oi||!!Le.lightProbeGrid!=S.state.lightProbeGridArray.length>0)&&(dt=!0):(dt=!0,Le.__version=Z.version);let Fi=Le.currentProgram;dt===!0&&(Fi=wa(Z,z,J),L&&Z.isNodeMaterial&&L.onUpdateProgram(Z,Fi,Le));let ln=!1,$n=!1,Ys=!1,yt=Fi.getUniforms(),Ft=Le.uniforms;if(v.useProgram(Fi.program)&&(ln=!0,$n=!0,Ys=!0),Z.id!==ie&&(ie=Z.id,$n=!0),Le.needsLights){let St=$m(S.state.lightProbeGridArray,J);Le.lightProbeGrid!==St&&(Le.lightProbeGrid=St,$n=!0)}if(ln||ge!==w){v.buffers.depth.getReversed()&&w.reversedDepth!==!0&&(w._reversedDepth=!0,w.updateProjectionMatrix()),yt.setValue(D,"projectionMatrix",w.projectionMatrix),yt.setValue(D,"viewMatrix",w.matrixWorldInverse);let Jn=yt.map.cameraPosition;Jn!==void 0&&Jn.setValue(D,G.setFromMatrixPosition(w.matrixWorld)),R.logarithmicDepthBuffer&&yt.setValue(D,"logDepthBufFC",2/(Math.log(w.far+1)/Math.LN2)),(Z.isMeshPhongMaterial||Z.isMeshToonMaterial||Z.isMeshLambertMaterial||Z.isMeshBasicMaterial||Z.isMeshStandardMaterial||Z.isShaderMaterial)&&yt.setValue(D,"isOrthographic",w.isOrthographicCamera===!0),ge!==w&&(ge=w,$n=!0,Ys=!0)}if(Le.needsLights&&(Ti.state.directionalShadowMap.length>0&&yt.setValue(D,"directionalShadowMap",Ti.state.directionalShadowMap,k),Ti.state.spotShadowMap.length>0&&yt.setValue(D,"spotShadowMap",Ti.state.spotShadowMap,k),Ti.state.pointShadowMap.length>0&&yt.setValue(D,"pointShadowMap",Ti.state.pointShadowMap,k)),J.isSkinnedMesh){yt.setOptional(D,J,"bindMatrix"),yt.setOptional(D,J,"bindMatrixInverse");let St=J.skeleton;St&&(St.boneTexture===null&&St.computeBoneTexture(),yt.setValue(D,"boneTexture",St.boneTexture,k))}J.isBatchedMesh&&(yt.setOptional(D,J,"batchingTexture"),yt.setValue(D,"batchingTexture",J._matricesTexture,k),yt.setOptional(D,J,"batchingIdTexture"),yt.setValue(D,"batchingIdTexture",J._indirectTexture,k),yt.setOptional(D,J,"batchingColorTexture"),J._colorsTexture!==null&&yt.setValue(D,"batchingColorTexture",J._colorsTexture,k));let Zn=K.morphAttributes;if((Zn.position!==void 0||Zn.normal!==void 0||Zn.color!==void 0)&&F.update(J,K,Fi),($n||Le.receiveShadow!==J.receiveShadow)&&(Le.receiveShadow=J.receiveShadow,yt.setValue(D,"receiveShadow",J.receiveShadow)),(Z.isMeshStandardMaterial||Z.isMeshLambertMaterial||Z.isMeshPhongMaterial)&&Z.envMap===null&&z.environment!==null&&(Ft.envMapIntensity.value=z.environmentIntensity),Ft.dfgLUT!==void 0&&(Ft.dfgLUT.value=cM()),$n){if(yt.setValue(D,"toneMappingExposure",C.toneMappingExposure),Le.needsLights&&Jm(Ft,Ys),Ce&&Z.fog===!0&&Ie.refreshFogUniforms(Ft,Ce),Ie.refreshMaterialUniforms(Ft,Z,le,he,S.state.transmissionRenderTarget[w.id]),Le.needsLights&&Le.lightProbeGrid){let St=Le.lightProbeGrid;Ft.probesSH.value=St.texture,Ft.probesMin.value.copy(St.boundingBox.min),Ft.probesMax.value.copy(St.boundingBox.max),Ft.probesResolution.value.copy(St.resolution)}Nr.upload(D,Nd(Le),Ft,k)}if(Z.isShaderMaterial&&Z.uniformsNeedUpdate===!0&&(Nr.upload(D,Nd(Le),Ft,k),Z.uniformsNeedUpdate=!1),Z.isSpriteMaterial&&yt.setValue(D,"center",J.center),yt.setValue(D,"modelViewMatrix",J.modelViewMatrix),yt.setValue(D,"normalMatrix",J.normalMatrix),yt.setValue(D,"modelMatrix",J.matrixWorld),Z.uniformsGroups!==void 0){let St=Z.uniformsGroups;for(let Jn=0,$s=St.length;Jn<$s;Jn++){let Od=St[Jn];de.update(Od,Fi),de.bind(Od,Fi)}}return Fi}function Jm(w,z){w.ambientLightColor.needsUpdate=z,w.lightProbe.needsUpdate=z,w.directionalLights.needsUpdate=z,w.directionalLightShadows.needsUpdate=z,w.pointLights.needsUpdate=z,w.pointLightShadows.needsUpdate=z,w.spotLights.needsUpdate=z,w.spotLightShadows.needsUpdate=z,w.rectAreaLights.needsUpdate=z,w.hemisphereLights.needsUpdate=z}function jm(w){return w.isMeshLambertMaterial||w.isMeshToonMaterial||w.isMeshPhongMaterial||w.isMeshStandardMaterial||w.isShadowMaterial||w.isShaderMaterial&&w.lights===!0}this.getActiveCubeFace=function(){return Y},this.getActiveMipmapLevel=function(){return X},this.getRenderTarget=function(){return ne},this.setRenderTargetTextures=function(w,z,K){let Z=B.get(w);Z.__autoAllocateDepthBuffer=w.resolveDepthBuffer===!1,Z.__autoAllocateDepthBuffer===!1&&(Z.__useRenderToTexture=!1),B.get(w.texture).__webglTexture=z,B.get(w.depthTexture).__webglTexture=Z.__autoAllocateDepthBuffer?void 0:K,Z.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(w,z){let K=B.get(w);K.__webglFramebuffer=z,K.__useDefaultFramebuffer=z===void 0},this.setRenderTarget=function(w,z=0,K=0){ne=w,Y=z,X=K;let Z=null,J=!1,Ce=!1;if(w){let Re=B.get(w);if(Re.__useDefaultFramebuffer!==void 0){v.bindFramebuffer(D.FRAMEBUFFER,Re.__webglFramebuffer),ue.copy(w.viewport),xe.copy(w.scissor),Ne=w.scissorTest,v.viewport(ue),v.scissor(xe),v.setScissorTest(Ne),ie=-1;return}else if(Re.__webglFramebuffer===void 0)k.setupRenderTarget(w);else if(Re.__hasExternalTextures)k.rebindTextures(w,B.get(w.texture).__webglTexture,B.get(w.depthTexture).__webglTexture);else if(w.depthBuffer){let nt=w.depthTexture;if(Re.__boundDepthTexture!==nt){if(nt!==null&&B.has(nt)&&(w.width!==nt.image.width||w.height!==nt.image.height))throw new Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");k.setupDepthRenderbuffer(w)}}let ze=w.texture;(ze.isData3DTexture||ze.isDataArrayTexture||ze.isCompressedArrayTexture)&&(Ce=!0);let Ge=B.get(w).__webglFramebuffer;w.isWebGLCubeRenderTarget?(Array.isArray(Ge[z])?Z=Ge[z][K]:Z=Ge[z],J=!0):w.samples>0&&k.useMultisampledRTT(w)===!1?Z=B.get(w).__webglMultisampledFramebuffer:Array.isArray(Ge)?Z=Ge[K]:Z=Ge,ue.copy(w.viewport),xe.copy(w.scissor),Ne=w.scissorTest}else ue.copy(ke).multiplyScalar(le).floor(),xe.copy(ae).multiplyScalar(le).floor(),Ne=ee;if(K!==0&&(Z=V),v.bindFramebuffer(D.FRAMEBUFFER,Z)&&v.drawBuffers(w,Z),v.viewport(ue),v.scissor(xe),v.setScissorTest(Ne),J){let Re=B.get(w.texture);D.framebufferTexture2D(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_CUBE_MAP_POSITIVE_X+z,Re.__webglTexture,K)}else if(Ce){let Re=z;for(let ze=0;ze<w.textures.length;ze++){let Ge=B.get(w.textures[ze]);D.framebufferTextureLayer(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0+ze,Ge.__webglTexture,K,Re)}}else if(w!==null&&K!==0){let Re=B.get(w.texture);D.framebufferTexture2D(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Re.__webglTexture,K)}ie=-1},this.readRenderTargetPixels=function(w,z,K,Z,J,Ce,Ue,Re=0){if(!(w&&w.isWebGLRenderTarget)){Ze("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ze=B.get(w).__webglFramebuffer;if(w.isWebGLCubeRenderTarget&&Ue!==void 0&&(ze=ze[Ue]),ze){v.bindFramebuffer(D.FRAMEBUFFER,ze);try{let Ge=w.textures[Re],nt=Ge.format,lt=Ge.type;if(w.textures.length>1&&D.readBuffer(D.COLOR_ATTACHMENT0+Re),!R.textureFormatReadable(nt)){Ze("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!R.textureTypeReadable(lt)){Ze("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}z>=0&&z<=w.width-Z&&K>=0&&K<=w.height-J&&D.readPixels(z,K,Z,J,we.convert(nt),we.convert(lt),Ce)}finally{let Ge=ne!==null?B.get(ne).__webglFramebuffer:null;v.bindFramebuffer(D.FRAMEBUFFER,Ge)}}},this.readRenderTargetPixelsAsync=async function(w,z,K,Z,J,Ce,Ue,Re=0){if(!(w&&w.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ze=B.get(w).__webglFramebuffer;if(w.isWebGLCubeRenderTarget&&Ue!==void 0&&(ze=ze[Ue]),ze)if(z>=0&&z<=w.width-Z&&K>=0&&K<=w.height-J){v.bindFramebuffer(D.FRAMEBUFFER,ze);let Ge=w.textures[Re],nt=Ge.format,lt=Ge.type;if(w.textures.length>1&&D.readBuffer(D.COLOR_ATTACHMENT0+Re),!R.textureFormatReadable(nt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!R.textureTypeReadable(lt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");let Ye=D.createBuffer();D.bindBuffer(D.PIXEL_PACK_BUFFER,Ye),D.bufferData(D.PIXEL_PACK_BUFFER,Ce.byteLength,D.STREAM_READ),D.readPixels(z,K,Z,J,we.convert(nt),we.convert(lt),0);let _t=ne!==null?B.get(ne).__webglFramebuffer:null;v.bindFramebuffer(D.FRAMEBUFFER,_t);let Nt=D.fenceSync(D.SYNC_GPU_COMMANDS_COMPLETE,0);return D.flush(),await ip(D,Nt,4),D.bindBuffer(D.PIXEL_PACK_BUFFER,Ye),D.getBufferSubData(D.PIXEL_PACK_BUFFER,0,Ce),D.deleteBuffer(Ye),D.deleteSync(Nt),Ce}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(w,z=null,K=0){let Z=Math.pow(2,-K),J=Math.floor(w.image.width*Z),Ce=Math.floor(w.image.height*Z),Ue=z!==null?z.x:0,Re=z!==null?z.y:0;k.setTexture2D(w,0),D.copyTexSubImage2D(D.TEXTURE_2D,K,0,0,Ue,Re,J,Ce),v.unbindTexture()},this.copyTextureToTexture=function(w,z,K=null,Z=null,J=0,Ce=0){let Ue,Re,ze,Ge,nt,lt,Ye,_t,Nt,Dt=w.isCompressedTexture?w.mipmaps[Ce]:w.image;if(K!==null)Ue=K.max.x-K.min.x,Re=K.max.y-K.min.y,ze=K.isBox3?K.max.z-K.min.z:1,Ge=K.min.x,nt=K.min.y,lt=K.isBox3?K.min.z:0;else{let Ft=Math.pow(2,-J);Ue=Math.floor(Dt.width*Ft),Re=Math.floor(Dt.height*Ft),w.isDataArrayTexture?ze=Dt.depth:w.isData3DTexture?ze=Math.floor(Dt.depth*Ft):ze=1,Ge=0,nt=0,lt=0}Z!==null?(Ye=Z.x,_t=Z.y,Nt=Z.z):(Ye=0,_t=0,Nt=0);let vt=we.convert(z.format),oi=we.convert(z.type),Le;z.isData3DTexture?(k.setTexture3D(z,0),Le=D.TEXTURE_3D):z.isDataArrayTexture||z.isCompressedArrayTexture?(k.setTexture2DArray(z,0),Le=D.TEXTURE_2D_ARRAY):(k.setTexture2D(z,0),Le=D.TEXTURE_2D),v.activeTexture(D.TEXTURE0),v.pixelStorei(D.UNPACK_FLIP_Y_WEBGL,z.flipY),v.pixelStorei(D.UNPACK_PREMULTIPLY_ALPHA_WEBGL,z.premultiplyAlpha),v.pixelStorei(D.UNPACK_ALIGNMENT,z.unpackAlignment);let Ti=v.getParameter(D.UNPACK_ROW_LENGTH),dt=v.getParameter(D.UNPACK_IMAGE_HEIGHT),Fi=v.getParameter(D.UNPACK_SKIP_PIXELS),ln=v.getParameter(D.UNPACK_SKIP_ROWS),$n=v.getParameter(D.UNPACK_SKIP_IMAGES);v.pixelStorei(D.UNPACK_ROW_LENGTH,Dt.width),v.pixelStorei(D.UNPACK_IMAGE_HEIGHT,Dt.height),v.pixelStorei(D.UNPACK_SKIP_PIXELS,Ge),v.pixelStorei(D.UNPACK_SKIP_ROWS,nt),v.pixelStorei(D.UNPACK_SKIP_IMAGES,lt);let Ys=w.isDataArrayTexture||w.isData3DTexture,yt=z.isDataArrayTexture||z.isData3DTexture;if(w.isDepthTexture){let Ft=B.get(w),Zn=B.get(z),St=B.get(Ft.__renderTarget),Jn=B.get(Zn.__renderTarget);v.bindFramebuffer(D.READ_FRAMEBUFFER,St.__webglFramebuffer),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,Jn.__webglFramebuffer);for(let $s=0;$s<ze;$s++)Ys&&(D.framebufferTextureLayer(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,B.get(w).__webglTexture,J,lt+$s),D.framebufferTextureLayer(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,B.get(z).__webglTexture,Ce,Nt+$s)),D.blitFramebuffer(Ge,nt,Ue,Re,Ye,_t,Ue,Re,D.DEPTH_BUFFER_BIT,D.NEAREST);v.bindFramebuffer(D.READ_FRAMEBUFFER,null),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,null)}else if(J!==0||w.isRenderTargetTexture||B.has(w)){let Ft=B.get(w),Zn=B.get(z);v.bindFramebuffer(D.READ_FRAMEBUFFER,q),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,N);for(let St=0;St<ze;St++)Ys?D.framebufferTextureLayer(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,Ft.__webglTexture,J,lt+St):D.framebufferTexture2D(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Ft.__webglTexture,J),yt?D.framebufferTextureLayer(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,Zn.__webglTexture,Ce,Nt+St):D.framebufferTexture2D(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Zn.__webglTexture,Ce),J!==0?D.blitFramebuffer(Ge,nt,Ue,Re,Ye,_t,Ue,Re,D.COLOR_BUFFER_BIT,D.NEAREST):yt?D.copyTexSubImage3D(Le,Ce,Ye,_t,Nt+St,Ge,nt,Ue,Re):D.copyTexSubImage2D(Le,Ce,Ye,_t,Ge,nt,Ue,Re);v.bindFramebuffer(D.READ_FRAMEBUFFER,null),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,null)}else yt?w.isDataTexture||w.isData3DTexture?D.texSubImage3D(Le,Ce,Ye,_t,Nt,Ue,Re,ze,vt,oi,Dt.data):z.isCompressedArrayTexture?D.compressedTexSubImage3D(Le,Ce,Ye,_t,Nt,Ue,Re,ze,vt,Dt.data):D.texSubImage3D(Le,Ce,Ye,_t,Nt,Ue,Re,ze,vt,oi,Dt):w.isDataTexture?D.texSubImage2D(D.TEXTURE_2D,Ce,Ye,_t,Ue,Re,vt,oi,Dt.data):w.isCompressedTexture?D.compressedTexSubImage2D(D.TEXTURE_2D,Ce,Ye,_t,Dt.width,Dt.height,vt,Dt.data):D.texSubImage2D(D.TEXTURE_2D,Ce,Ye,_t,Ue,Re,vt,oi,Dt);v.pixelStorei(D.UNPACK_ROW_LENGTH,Ti),v.pixelStorei(D.UNPACK_IMAGE_HEIGHT,dt),v.pixelStorei(D.UNPACK_SKIP_PIXELS,Fi),v.pixelStorei(D.UNPACK_SKIP_ROWS,ln),v.pixelStorei(D.UNPACK_SKIP_IMAGES,$n),Ce===0&&z.generateMipmaps&&D.generateMipmap(Le),v.unbindTexture()},this.initRenderTarget=function(w){B.get(w).__webglFramebuffer===void 0&&k.setupRenderTarget(w)},this.initTexture=function(w){w.isCubeTexture?k.setTextureCube(w,0):w.isData3DTexture?k.setTexture3D(w,0):w.isDataArrayTexture||w.isCompressedArrayTexture?k.setTexture2DArray(w,0):k.setTexture2D(w,0),v.unbindTexture()},this.resetState=function(){Y=0,X=0,ne=null,v.reset(),Pe.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return $i}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;let t=this.getContext();t.drawingBufferColorSpace=ht._getDrawingBufferColorSpace(e),t.unpackColorSpace=ht._getUnpackColorSpace()}};var kp={type:"change"},Qu={type:"start"},Vp={type:"end"},Xc=new Dn,Hp=new Bi,hM=Math.cos(70*Vt.DEG2RAD),qt=new P,bi=2*Math.PI,xt={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},Ku=1e-6,qc=class extends qo{constructor(e,t=null){super(e,t),this.state=xt.NONE,this.target=new P,this.cursor=new P,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:cs.ROTATE,MIDDLE:cs.DOLLY,RIGHT:cs.PAN},this.touches={ONE:hs.ROTATE,TWO:hs.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._cursorStyle="auto",this._domElementKeyEvents=null,this._lastPosition=new P,this._lastQuaternion=new Pi,this._lastTargetPosition=new P,this._quat=new Pi().setFromUnitVectors(e.up,new P(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new Pr,this._sphericalDelta=new Pr,this._scale=1,this._panOffset=new P,this._rotateStart=new $,this._rotateEnd=new $,this._rotateDelta=new $,this._panStart=new $,this._panEnd=new $,this._panDelta=new $,this._dollyStart=new $,this._dollyEnd=new $,this._dollyDelta=new $,this._dollyDirection=new P,this._mouse=new $,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=dM.bind(this),this._onPointerDown=uM.bind(this),this._onPointerUp=fM.bind(this),this._onContextMenu=yM.bind(this),this._onMouseWheel=gM.bind(this),this._onKeyDown=_M.bind(this),this._onTouchStart=xM.bind(this),this._onTouchMove=vM.bind(this),this._onMouseDown=pM.bind(this),this._onMouseMove=mM.bind(this),this._interceptControlDown=MM.bind(this),this._interceptControlUp=SM.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}set cursorStyle(e){this._cursorStyle=e,e==="grab"?this.domElement.style.cursor="grab":this.domElement.style.cursor="auto"}get cursorStyle(){return this._cursorStyle}connect(e){super.connect(e),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction=""}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(e){e.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=e}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(kp),this.update(),this.state=xt.NONE}pan(e,t){this._pan(e,t),this.update()}dollyIn(e){this._dollyIn(e),this.update()}dollyOut(e){this._dollyOut(e),this.update()}rotateLeft(e){this._rotateLeft(e),this.update()}rotateUp(e){this._rotateUp(e),this.update()}update(e=null){let t=this.object.position;qt.copy(t).sub(this.target),qt.applyQuaternion(this._quat),this._spherical.setFromVector3(qt),this.autoRotate&&this.state===xt.NONE&&this._rotateLeft(this._getAutoRotationAngle(e)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let i=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(i)&&isFinite(s)&&(i<-Math.PI?i+=bi:i>Math.PI&&(i-=bi),s<-Math.PI?s+=bi:s>Math.PI&&(s-=bi),i<=s?this._spherical.theta=Math.max(i,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(i+s)/2?Math.max(i,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{let o=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=o!=this._spherical.radius}if(qt.setFromSpherical(this._spherical),qt.applyQuaternion(this._quatInverse),t.copy(this.target).add(qt),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let o=null;if(this.object.isPerspectiveCamera){let a=qt.length();o=this._clampDistance(a*this._scale);let c=a-o;this.object.position.addScaledVector(this._dollyDirection,c),this.object.updateMatrixWorld(),r=!!c}else if(this.object.isOrthographicCamera){let a=new P(this._mouse.x,this._mouse.y,0);a.unproject(this.object);let c=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=c!==this.object.zoom;let l=new P(this._mouse.x,this._mouse.y,0);l.unproject(this.object),this.object.position.sub(l).add(a),this.object.updateMatrixWorld(),o=qt.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;o!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(o).add(this.object.position):(Xc.origin.copy(this.object.position),Xc.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Xc.direction))<hM?this.object.lookAt(this.target):(Hp.setFromNormalAndCoplanarPoint(this.object.up,this.target),Xc.intersectPlane(Hp,this.target))))}else if(this.object.isOrthographicCamera){let o=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),o!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>Ku||8*(1-this._lastQuaternion.dot(this.object.quaternion))>Ku||this._lastTargetPosition.distanceToSquared(this.target)>Ku?(this.dispatchEvent(kp),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(e){return e!==null?bi/60*this.autoRotateSpeed*e:bi/60/60*this.autoRotateSpeed}_getZoomScale(e){let t=Math.abs(e*.01);return Math.pow(.95,this.zoomSpeed*t)}_rotateLeft(e){this._sphericalDelta.theta-=e}_rotateUp(e){this._sphericalDelta.phi-=e}_panLeft(e,t){qt.setFromMatrixColumn(t,0),qt.multiplyScalar(-e),this._panOffset.add(qt)}_panUp(e,t){this.screenSpacePanning===!0?qt.setFromMatrixColumn(t,1):(qt.setFromMatrixColumn(t,0),qt.crossVectors(this.object.up,qt)),qt.multiplyScalar(e),this._panOffset.add(qt)}_pan(e,t){let i=this.domElement;if(this.object.isPerspectiveCamera){let s=this.object.position;qt.copy(s).sub(this.target);let r=qt.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*e*r/i.clientHeight,this.object.matrix),this._panUp(2*t*r/i.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(e*(this.object.right-this.object.left)/this.object.zoom/i.clientWidth,this.object.matrix),this._panUp(t*(this.object.top-this.object.bottom)/this.object.zoom/i.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(e,t){if(!this.zoomToCursor)return;this._performCursorZoom=!0;let i=this.domElement.getBoundingClientRect(),s=e-i.left,r=t-i.top,o=i.width,a=i.height;this._mouse.x=s/o*2-1,this._mouse.y=-(r/a)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(e){return Math.max(this.minDistance,Math.min(this.maxDistance,e))}_handleMouseDownRotate(e){this._rotateStart.set(e.clientX,e.clientY)}_handleMouseDownDolly(e){this._updateZoomParameters(e.clientX,e.clientX),this._dollyStart.set(e.clientX,e.clientY)}_handleMouseDownPan(e){this._panStart.set(e.clientX,e.clientY)}_handleMouseMoveRotate(e){this._rotateEnd.set(e.clientX,e.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let t=this.domElement;this._rotateLeft(bi*this._rotateDelta.x/t.clientHeight),this._rotateUp(bi*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(e){this._dollyEnd.set(e.clientX,e.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(e){this._panEnd.set(e.clientX,e.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(e){this._updateZoomParameters(e.clientX,e.clientY),e.deltaY<0?this._dollyIn(this._getZoomScale(e.deltaY)):e.deltaY>0&&this._dollyOut(this._getZoomScale(e.deltaY)),this.update()}_handleKeyDown(e){let t=!1;switch(e.code){case this.keys.UP:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(bi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),t=!0;break;case this.keys.BOTTOM:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(-bi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),t=!0;break;case this.keys.LEFT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(bi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),t=!0;break;case this.keys.RIGHT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(-bi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),this.update())}_handleTouchStartRotate(e){if(this._pointers.length===1)this._rotateStart.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._rotateStart.set(i,s)}}_handleTouchStartPan(e){if(this._pointers.length===1)this._panStart.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panStart.set(i,s)}}_handleTouchStartDolly(e){let t=this._getSecondPointerPosition(e),i=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(i*i+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enablePan&&this._handleTouchStartPan(e)}_handleTouchStartDollyRotate(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enableRotate&&this._handleTouchStartRotate(e)}_handleTouchMoveRotate(e){if(this._pointers.length==1)this._rotateEnd.set(e.pageX,e.pageY);else{let i=this._getSecondPointerPosition(e),s=.5*(e.pageX+i.x),r=.5*(e.pageY+i.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let t=this.domElement;this._rotateLeft(bi*this._rotateDelta.x/t.clientHeight),this._rotateUp(bi*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(e){if(this._pointers.length===1)this._panEnd.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panEnd.set(i,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(e){let t=this._getSecondPointerPosition(e),i=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(i*i+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);let o=(e.pageX+t.x)*.5,a=(e.pageY+t.y)*.5;this._updateZoomParameters(o,a)}_handleTouchMoveDollyPan(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enablePan&&this._handleTouchMovePan(e)}_handleTouchMoveDollyRotate(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enableRotate&&this._handleTouchMoveRotate(e)}_addPointer(e){this._pointers.push(e.pointerId)}_removePointer(e){delete this._pointerPositions[e.pointerId];for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId){this._pointers.splice(t,1);return}}_isTrackingPointer(e){for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId)return!0;return!1}_trackPointer(e){let t=this._pointerPositions[e.pointerId];t===void 0&&(t=new $,this._pointerPositions[e.pointerId]=t),t.set(e.pageX,e.pageY)}_getSecondPointerPosition(e){let t=e.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[t]}_customWheelEvent(e){let t=e.deltaMode,i={clientX:e.clientX,clientY:e.clientY,deltaY:e.deltaY};switch(t){case 1:i.deltaY*=16;break;case 2:i.deltaY*=100;break}return e.ctrlKey&&!this._controlActive&&(i.deltaY*=10),i}};function uM(n){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(n.pointerId),this.domElement.ownerDocument.addEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(n)&&(this._addPointer(n),n.pointerType==="touch"?this._onTouchStart(n):this._onMouseDown(n),this._cursorStyle==="grab"&&(this.domElement.style.cursor="grabbing")))}function dM(n){this.enabled!==!1&&(n.pointerType==="touch"?this._onTouchMove(n):this._onMouseMove(n))}function fM(n){switch(this._removePointer(n),this._pointers.length){case 0:this.domElement.releasePointerCapture(n.pointerId),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Vp),this.state=xt.NONE,this._cursorStyle==="grab"&&(this.domElement.style.cursor="grab");break;case 1:let e=this._pointers[0],t=this._pointerPositions[e];this._onTouchStart({pointerId:e,pageX:t.x,pageY:t.y});break}}function pM(n){let e;switch(n.button){case 0:e=this.mouseButtons.LEFT;break;case 1:e=this.mouseButtons.MIDDLE;break;case 2:e=this.mouseButtons.RIGHT;break;default:e=-1}switch(e){case cs.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(n),this.state=xt.DOLLY;break;case cs.ROTATE:if(n.ctrlKey||n.metaKey||n.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(n),this.state=xt.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(n),this.state=xt.ROTATE}break;case cs.PAN:if(n.ctrlKey||n.metaKey||n.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(n),this.state=xt.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(n),this.state=xt.PAN}break;default:this.state=xt.NONE}this.state!==xt.NONE&&this.dispatchEvent(Qu)}function mM(n){switch(this.state){case xt.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(n);break;case xt.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(n);break;case xt.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(n);break}}function gM(n){this.enabled===!1||this.enableZoom===!1||this.state!==xt.NONE||(n.preventDefault(),this.dispatchEvent(Qu),this._handleMouseWheel(this._customWheelEvent(n)),this.dispatchEvent(Vp))}function _M(n){this.enabled!==!1&&this._handleKeyDown(n)}function xM(n){switch(this._trackPointer(n),this._pointers.length){case 1:switch(this.touches.ONE){case hs.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(n),this.state=xt.TOUCH_ROTATE;break;case hs.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(n),this.state=xt.TOUCH_PAN;break;default:this.state=xt.NONE}break;case 2:switch(this.touches.TWO){case hs.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(n),this.state=xt.TOUCH_DOLLY_PAN;break;case hs.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(n),this.state=xt.TOUCH_DOLLY_ROTATE;break;default:this.state=xt.NONE}break;default:this.state=xt.NONE}this.state!==xt.NONE&&this.dispatchEvent(Qu)}function vM(n){switch(this._trackPointer(n),this.state){case xt.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(n),this.update();break;case xt.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(n),this.update();break;case xt.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(n),this.update();break;case xt.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(n),this.update();break;default:this.state=xt.NONE}}function yM(n){this.enabled!==!1&&n.preventDefault()}function MM(n){n.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function SM(n){n.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}var Yc=class extends Ds{constructor(){super(),this.name="RoomEnvironment",this.position.y=-3.5;let e=new Bt;e.deleteAttribute("uv");let t=new Qe({side:ti}),i=new Qe,s=new Ho(16777215,900,28,2);s.position.set(.418,16.199,.3),this.add(s);let r=new Ke(e,t);r.position.set(-.757,13.219,.717),r.scale.set(31.713,28.305,28.591),this.add(r);let o=new jt(e,i,6),a=new ft;a.position.set(-10.906,2.009,1.846),a.rotation.set(0,-.195,0),a.scale.set(2.328,7.905,4.651),a.updateMatrix(),o.setMatrixAt(0,a.matrix),a.position.set(-5.607,-.754,-.758),a.rotation.set(0,.994,0),a.scale.set(1.97,1.534,3.955),a.updateMatrix(),o.setMatrixAt(1,a.matrix),a.position.set(6.167,.857,7.803),a.rotation.set(0,.561,0),a.scale.set(3.927,6.285,3.687),a.updateMatrix(),o.setMatrixAt(2,a.matrix),a.position.set(-2.017,.018,6.124),a.rotation.set(0,.333,0),a.scale.set(2.002,4.566,2.064),a.updateMatrix(),o.setMatrixAt(3,a.matrix),a.position.set(2.291,-.756,-2.621),a.rotation.set(0,-.286,0),a.scale.set(1.546,1.552,1.496),a.updateMatrix(),o.setMatrixAt(4,a.matrix),a.position.set(-2.193,-.369,-5.547),a.rotation.set(0,.516,0),a.scale.set(3.875,3.487,2.986),a.updateMatrix(),o.setMatrixAt(5,a.matrix),this.add(o);let c=new Ke(e,Br(50));c.position.set(-16.116,14.37,8.208),c.scale.set(.1,2.428,2.739),this.add(c);let l=new Ke(e,Br(50));l.position.set(-16.109,18.021,-8.207),l.scale.set(.1,2.425,2.751),this.add(l);let h=new Ke(e,Br(17));h.position.set(14.904,12.198,-1.832),h.scale.set(.15,4.265,6.331),this.add(h);let u=new Ke(e,Br(43));u.position.set(-.462,8.89,14.52),u.scale.set(4.38,5.441,.088),this.add(u);let d=new Ke(e,Br(20));d.position.set(3.235,11.486,-12.541),d.scale.set(2.5,2,.1),this.add(d);let f=new Ke(e,Br(100));f.position.set(0,20,0),f.scale.set(1,.1,1),this.add(f)}dispose(){let e=new Set;this.traverse(t=>{t.isMesh&&(e.add(t.geometry),e.add(t.material))});for(let t of e)t.dispose()}};function Br(n){return new Oo({color:0,emissive:16777215,emissiveIntensity:n})}var Gp=new pi,$c=new P,ks=class extends Vo{constructor(){super(),this.isLineSegmentsGeometry=!0,this.type="LineSegmentsGeometry";let e=[-1,2,0,1,2,0,-1,1,0,1,1,0,-1,0,0,1,0,0,-1,-1,0,1,-1,0],t=[-1,2,1,2,-1,1,1,1,-1,-1,1,-1,-1,-2,1,-2],i=[0,2,1,2,3,1,2,4,3,4,5,3,4,6,5,6,7,5];this.setIndex(i),this.setAttribute("position",new it(e,3)),this.setAttribute("uv",new it(t,2))}applyMatrix4(e){let t=this.attributes.instanceStart,i=this.attributes.instanceEnd;return t!==void 0&&(t.applyMatrix4(e),i.applyMatrix4(e),t.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}setPositions(e){let t;e instanceof Float32Array?t=e:Array.isArray(e)&&(t=new Float32Array(e));let i=new ls(t,6,1);return this.setAttribute("instanceStart",new Di(i,3,0)),this.setAttribute("instanceEnd",new Di(i,3,3)),this.instanceCount=this.attributes.instanceStart.count,this.computeBoundingBox(),this.computeBoundingSphere(),this}setColors(e){let t;e instanceof Float32Array?t=e:Array.isArray(e)&&(t=new Float32Array(e));let i=new ls(t,6,1);return this.setAttribute("instanceColorStart",new Di(i,3,0)),this.setAttribute("instanceColorEnd",new Di(i,3,3)),this}fromWireframeGeometry(e){return this.setPositions(e.attributes.position.array),this}fromEdgesGeometry(e){return this.setPositions(e.attributes.position.array),this}fromMesh(e){return this.fromWireframeGeometry(new Lo(e.geometry)),this}fromLineSegments(e){let t=e.geometry;return this.setPositions(t.attributes.position.array),this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new pi);let e=this.attributes.instanceStart,t=this.attributes.instanceEnd;e!==void 0&&t!==void 0&&(this.boundingBox.setFromBufferAttribute(e),Gp.setFromBufferAttribute(t),this.boundingBox.union(Gp))}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new vi),this.boundingBox===null&&this.computeBoundingBox();let e=this.attributes.instanceStart,t=this.attributes.instanceEnd;if(e!==void 0&&t!==void 0){let i=this.boundingSphere.center;this.boundingBox.getCenter(i);let s=0;for(let r=0,o=e.count;r<o;r++)$c.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared($c)),$c.fromBufferAttribute(t,r),s=Math.max(s,i.distanceToSquared($c));this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&console.error("THREE.LineSegmentsGeometry.computeBoundingSphere(): Computed radius is NaN. The instanced position data is likely to have NaN values.",this)}}toJSON(){}};be.line={worldUnits:{value:1},linewidth:{value:1},resolution:{value:new $},dashOffset:{value:0},dashScale:{value:1},dashSize:{value:1},gapSize:{value:1}};gi.line={uniforms:mi.merge([be.common,be.fog,be.line]),vertexShader:`
		#include <common>
		#include <color_pars_vertex>
		#include <fog_pars_vertex>
		#include <logdepthbuf_pars_vertex>
		#include <clipping_planes_pars_vertex>

		uniform float linewidth;
		uniform vec2 resolution;

		attribute vec3 instanceStart;
		attribute vec3 instanceEnd;

		attribute vec3 instanceColorStart;
		attribute vec3 instanceColorEnd;

		#ifdef WORLD_UNITS

			varying vec4 worldPos;
			varying vec3 worldStart;
			varying vec3 worldEnd;

			#ifdef USE_DASH

				varying vec2 vUv;

			#endif

		#else

			varying vec2 vUv;

		#endif

		#ifdef USE_DASH

			uniform float dashScale;
			attribute float instanceDistanceStart;
			attribute float instanceDistanceEnd;
			varying float vLineDistance;

		#endif

		float trimSegmentAlpha( const in vec4 start, const in vec4 end ) {

			// compute the interpolation factor needed to trim the segment so it terminates
			// between the camera plane and the near plane

			// conservative estimate of the near plane
			float a = projectionMatrix[ 2 ][ 2 ]; // 3nd entry in 3th column
			float b = projectionMatrix[ 3 ][ 2 ]; // 3nd entry in 4th column

			// we need different nearEstimate formula for reversed and default depth buffer
			// a is positive with a reversed depth buffer so it can be used for controlling the code flow
			float nearEstimate = ( a > 0.0 ) ? ( - b / ( a + 1.0 ) ) : ( - 0.5 * b / a );

			return ( nearEstimate - start.z ) / ( end.z - start.z );

		}

		void main() {

			#ifdef USE_COLOR

				vColor.xyz = ( position.y < 0.5 ) ? instanceColorStart : instanceColorEnd;

			#endif

			float aspect = resolution.x / resolution.y;

			// camera space
			vec4 start = modelViewMatrix * vec4( instanceStart, 1.0 );
			vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );

			#ifdef USE_DASH

				float lineDistanceStart = dashScale * instanceDistanceStart;
				float lineDistanceEnd = dashScale * instanceDistanceEnd;

			#endif

			#ifdef WORLD_UNITS

				worldStart = start.xyz;
				worldEnd = end.xyz;

			#else

				vUv = uv;

			#endif

			// special case for perspective projection, and segments that terminate either in, or behind, the camera plane
			// clearly the gpu firmware has a way of addressing this issue when projecting into ndc space
			// but we need to perform ndc-space calculations in the shader, so we must address this issue directly
			// perhaps there is a more elegant solution -- WestLangley

			bool perspective = ( projectionMatrix[ 2 ][ 3 ] == - 1.0 ); // 4th entry in the 3rd column

			if ( perspective ) {

				if ( start.z < 0.0 && end.z >= 0.0 ) {

					float alpha = trimSegmentAlpha( start, end );
					end.xyz = mix( start.xyz, end.xyz, alpha );

					#ifdef USE_DASH

						lineDistanceEnd = mix( lineDistanceStart, lineDistanceEnd, alpha );

					#endif

				} else if ( end.z < 0.0 && start.z >= 0.0 ) {

					float alpha = trimSegmentAlpha( end, start );
					start.xyz = mix( end.xyz, start.xyz, alpha );

					#ifdef USE_DASH

						lineDistanceStart = mix( lineDistanceEnd, lineDistanceStart, alpha );

					#endif

				}

			}

			#ifdef USE_DASH

				vLineDistance = ( position.y < 0.5 ) ? lineDistanceStart : lineDistanceEnd;
				vUv = uv;

			#endif

			// clip space
			vec4 clipStart = projectionMatrix * start;
			vec4 clipEnd = projectionMatrix * end;

			// ndc space
			vec3 ndcStart = clipStart.xyz / clipStart.w;
			vec3 ndcEnd = clipEnd.xyz / clipEnd.w;

			// direction
			vec2 dir = ndcEnd.xy - ndcStart.xy;

			// account for clip-space aspect ratio
			dir.x *= aspect;
			dir = normalize( dir );

			#ifdef WORLD_UNITS

				vec3 worldDir = normalize( end.xyz - start.xyz );
				vec3 tmpFwd = normalize( mix( start.xyz, end.xyz, 0.5 ) );
				vec3 worldUp = normalize( cross( worldDir, tmpFwd ) );
				vec3 worldFwd = cross( worldDir, worldUp );
				worldPos = position.y < 0.5 ? start: end;

				// height offset
				float hw = linewidth * 0.5;
				worldPos.xyz += position.x < 0.0 ? hw * worldUp : - hw * worldUp;

				// don't extend the line if we're rendering dashes because we
				// won't be rendering the endcaps
				#ifndef USE_DASH

					// cap extension
					worldPos.xyz += position.y < 0.5 ? - hw * worldDir : hw * worldDir;

					// add width to the box
					worldPos.xyz += worldFwd * hw;

					// endcaps
					if ( position.y > 1.0 || position.y < 0.0 ) {

						worldPos.xyz -= worldFwd * 2.0 * hw;

					}

				#endif

				// project the worldpos
				vec4 clip = projectionMatrix * worldPos;

				// shift the depth of the projected points so the line
				// segments overlap neatly
				vec3 clipPose = ( position.y < 0.5 ) ? ndcStart : ndcEnd;
				clip.z = clipPose.z * clip.w;

			#else

				vec2 offset = vec2( dir.y, - dir.x );
				// undo aspect ratio adjustment
				dir.x /= aspect;
				offset.x /= aspect;

				// sign flip
				if ( position.x < 0.0 ) offset *= - 1.0;

				// endcaps
				if ( position.y < 0.0 ) {

					offset += - dir;

				} else if ( position.y > 1.0 ) {

					offset += dir;

				}

				// adjust for linewidth
				offset *= linewidth;

				// adjust for clip-space to screen-space conversion // maybe resolution should be based on viewport ...
				offset /= resolution.y;

				// select end
				vec4 clip = ( position.y < 0.5 ) ? clipStart : clipEnd;

				// back to clip space
				offset *= clip.w;

				clip.xy += offset;

			#endif

			gl_Position = clip;

			vec4 mvPosition = ( position.y < 0.5 ) ? start : end; // this is an approximation

			#include <logdepthbuf_vertex>
			#include <clipping_planes_vertex>
			#include <fog_vertex>

		}
		`,fragmentShader:`
		uniform vec3 diffuse;
		uniform float opacity;
		uniform float linewidth;

		#ifdef USE_DASH

			uniform float dashOffset;
			uniform float dashSize;
			uniform float gapSize;

		#endif

		varying float vLineDistance;

		#ifdef WORLD_UNITS

			varying vec4 worldPos;
			varying vec3 worldStart;
			varying vec3 worldEnd;

			#ifdef USE_DASH

				varying vec2 vUv;

			#endif

		#else

			varying vec2 vUv;

		#endif

		#include <common>
		#include <color_pars_fragment>
		#include <fog_pars_fragment>
		#include <logdepthbuf_pars_fragment>
		#include <clipping_planes_pars_fragment>

		vec2 closestLineToLine(vec3 p1, vec3 p2, vec3 p3, vec3 p4) {

			float mua;
			float mub;

			vec3 p13 = p1 - p3;
			vec3 p43 = p4 - p3;

			vec3 p21 = p2 - p1;

			float d1343 = dot( p13, p43 );
			float d4321 = dot( p43, p21 );
			float d1321 = dot( p13, p21 );
			float d4343 = dot( p43, p43 );
			float d2121 = dot( p21, p21 );

			float denom = d2121 * d4343 - d4321 * d4321;

			float numer = d1343 * d4321 - d1321 * d4343;

			mua = numer / denom;
			mua = clamp( mua, 0.0, 1.0 );
			mub = ( d1343 + d4321 * ( mua ) ) / d4343;
			mub = clamp( mub, 0.0, 1.0 );

			return vec2( mua, mub );

		}

		void main() {

			float alpha = opacity;
			vec4 diffuseColor = vec4( diffuse, alpha );

			#include <clipping_planes_fragment>

			#ifdef USE_DASH

				if ( vUv.y < - 1.0 || vUv.y > 1.0 ) discard; // discard endcaps

				if ( mod( vLineDistance + dashOffset, dashSize + gapSize ) > dashSize ) discard; // todo - FIX

			#endif

			#ifdef WORLD_UNITS

				// Find the closest points on the view ray and the line segment
				vec3 rayEnd = normalize( worldPos.xyz ) * 1e5;
				vec3 lineDir = worldEnd - worldStart;
				vec2 params = closestLineToLine( worldStart, worldEnd, vec3( 0.0, 0.0, 0.0 ), rayEnd );

				vec3 p1 = worldStart + lineDir * params.x;
				vec3 p2 = rayEnd * params.y;
				vec3 delta = p1 - p2;
				float len = length( delta );
				float norm = len / linewidth;

				#ifndef USE_DASH

					#ifdef USE_ALPHA_TO_COVERAGE

						float dnorm = fwidth( norm );
						alpha = 1.0 - smoothstep( 0.5 - dnorm, 0.5 + dnorm, norm );

					#else

						if ( norm > 0.5 ) {

							discard;

						}

					#endif

				#endif

			#else

				#ifdef USE_ALPHA_TO_COVERAGE

					// artifacts appear on some hardware if a derivative is taken within a conditional
					float a = vUv.x;
					float b = ( vUv.y > 0.0 ) ? vUv.y - 1.0 : vUv.y + 1.0;
					float len2 = a * a + b * b;
					float dlen = fwidth( len2 );

					if ( abs( vUv.y ) > 1.0 ) {

						alpha = 1.0 - smoothstep( 1.0 - dlen, 1.0 + dlen, len2 );

					}

				#else

					if ( abs( vUv.y ) > 1.0 ) {

						float a = vUv.x;
						float b = ( vUv.y > 0.0 ) ? vUv.y - 1.0 : vUv.y + 1.0;
						float len2 = a * a + b * b;

						if ( len2 > 1.0 ) discard;

					}

				#endif

			#endif

			#include <logdepthbuf_fragment>
			#include <color_fragment>

			gl_FragColor = vec4( diffuseColor.rgb, alpha );

			#include <tonemapping_fragment>
			#include <colorspace_fragment>
			#include <fog_fragment>
			#include <premultiplied_alpha_fragment>

		}
		`};var zr=class extends bt{constructor(e){super({type:"LineMaterial",uniforms:mi.clone(gi.line.uniforms),vertexShader:gi.line.vertexShader,fragmentShader:gi.line.fragmentShader,clipping:!0}),this.isLineMaterial=!0,this.setValues(e)}get color(){return this.uniforms.diffuse.value}set color(e){this.uniforms.diffuse.value=e}get worldUnits(){return"WORLD_UNITS"in this.defines}set worldUnits(e){e===!0!==this.worldUnits&&(this.needsUpdate=!0),e===!0?this.defines.WORLD_UNITS="":delete this.defines.WORLD_UNITS}get linewidth(){return this.uniforms.linewidth.value}set linewidth(e){this.uniforms.linewidth&&(this.uniforms.linewidth.value=e)}get dashed(){return"USE_DASH"in this.defines}set dashed(e){e===!0!==this.dashed&&(this.needsUpdate=!0),e===!0?this.defines.USE_DASH="":delete this.defines.USE_DASH}get dashScale(){return this.uniforms.dashScale.value}set dashScale(e){this.uniforms.dashScale.value=e}get dashSize(){return this.uniforms.dashSize.value}set dashSize(e){this.uniforms.dashSize.value=e}get dashOffset(){return this.uniforms.dashOffset.value}set dashOffset(e){this.uniforms.dashOffset.value=e}get gapSize(){return this.uniforms.gapSize.value}set gapSize(e){this.uniforms.gapSize.value=e}get opacity(){return this.uniforms.opacity.value}set opacity(e){this.uniforms&&(this.uniforms.opacity.value=e)}get resolution(){return this.uniforms.resolution.value}set resolution(e){this.uniforms.resolution.value.copy(e)}get alphaToCoverage(){return"USE_ALPHA_TO_COVERAGE"in this.defines}set alphaToCoverage(e){this.defines&&(e===!0!==this.alphaToCoverage&&(this.needsUpdate=!0),e===!0?this.defines.USE_ALPHA_TO_COVERAGE="":delete this.defines.USE_ALPHA_TO_COVERAGE)}};var ed=new mt,Wp=new P,Xp=new P,ni=new mt,si=new mt,yn=new mt,td=new P,id=new rt,ri=new Xo,qp=new P,Zc=new pi,Jc=new vi,Mn=new mt,Sn,Hs;function Yp(n,e,t){return Mn.set(0,0,-e,1).applyMatrix4(n.projectionMatrix),Mn.multiplyScalar(1/Mn.w),Mn.x=Hs/t.width,Mn.y=Hs/t.height,Mn.applyMatrix4(n.projectionMatrixInverse),Mn.multiplyScalar(1/Mn.w),Math.abs(Math.max(Mn.x,Mn.y))}function bM(n,e){let t=n.matrixWorld,i=n.geometry,s=i.attributes.instanceStart,r=i.attributes.instanceEnd,o=Math.min(i.instanceCount,s.count);for(let a=0,c=o;a<c;a++){ri.start.fromBufferAttribute(s,a),ri.end.fromBufferAttribute(r,a),ri.applyMatrix4(t);let l=new P,h=new P;Sn.distanceSqToSegment(ri.start,ri.end,h,l),h.distanceTo(l)<Hs*.5&&e.push({point:h,pointOnLine:l,distance:Sn.origin.distanceTo(h),object:n,face:null,faceIndex:a,uv:null,uv1:null})}}function EM(n,e,t){let i=e.projectionMatrix,r=n.material.resolution,o=n.matrixWorld,a=n.geometry,c=a.attributes.instanceStart,l=a.attributes.instanceEnd,h=Math.min(a.instanceCount,c.count),u=-e.near;Sn.at(1,yn),yn.w=1,yn.applyMatrix4(e.matrixWorldInverse),yn.applyMatrix4(i),yn.multiplyScalar(1/yn.w),yn.x*=r.x/2,yn.y*=r.y/2,yn.z=0,td.copy(yn),id.multiplyMatrices(e.matrixWorldInverse,o);for(let d=0,f=h;d<f;d++){if(ni.fromBufferAttribute(c,d),si.fromBufferAttribute(l,d),ni.w=1,si.w=1,ni.applyMatrix4(id),si.applyMatrix4(id),ni.z>u&&si.z>u)continue;if(ni.z>u){let b=ni.z-si.z,y=(ni.z-u)/b;ni.lerp(si,y)}else if(si.z>u){let b=si.z-ni.z,y=(si.z-u)/b;si.lerp(ni,y)}ni.applyMatrix4(i),si.applyMatrix4(i),ni.multiplyScalar(1/ni.w),si.multiplyScalar(1/si.w),ni.x*=r.x/2,ni.y*=r.y/2,si.x*=r.x/2,si.y*=r.y/2,ri.start.copy(ni),ri.start.z=0,ri.end.copy(si),ri.end.z=0;let x=ri.closestPointToPointParameter(td,!0);ri.at(x,qp);let p=Vt.lerp(ni.z,si.z,x),m=p>=-1&&p<=1,M=td.distanceTo(qp)<Hs*.5;if(m&&M){ri.start.fromBufferAttribute(c,d),ri.end.fromBufferAttribute(l,d),ri.start.applyMatrix4(o),ri.end.applyMatrix4(o);let b=new P,y=new P;Sn.distanceSqToSegment(ri.start,ri.end,y,b),t.push({point:y,pointOnLine:b,distance:Sn.origin.distanceTo(y),object:n,face:null,faceIndex:d,uv:null,uv1:null})}}}var jc=class extends Ke{constructor(e=new ks,t=new zr({color:Math.random()*16777215})){super(e,t),this.isLineSegments2=!0,this.type="LineSegments2"}computeLineDistances(){let e=this.geometry,t=e.attributes.instanceStart,i=e.attributes.instanceEnd,s=new Float32Array(2*t.count);for(let o=0,a=0,c=t.count;o<c;o++,a+=2)Wp.fromBufferAttribute(t,o),Xp.fromBufferAttribute(i,o),s[a]=a===0?0:s[a-1],s[a+1]=s[a]+Wp.distanceTo(Xp);let r=new ls(s,2,1);return e.setAttribute("instanceDistanceStart",new Di(r,1,0)),e.setAttribute("instanceDistanceEnd",new Di(r,1,1)),this}raycast(e,t){let i=this.material.worldUnits,s=e.camera;if(s===null&&!i&&console.error('LineSegments2: "Raycaster.camera" needs to be set in order to raycast against LineSegments2 while worldUnits is set to false.'),i===!1&&(this.material.resolution.x===0||this.material.resolution.y===0))return;let r=e.params.Line2!==void 0&&e.params.Line2.threshold||0;Sn=e.ray;let o=this.matrixWorld,a=this.geometry,c=this.material;Hs=c.linewidth+r,a.boundingSphere===null&&a.computeBoundingSphere(),Jc.copy(a.boundingSphere).applyMatrix4(o);let l;if(i)l=Hs*.5;else{let u=Math.max(s.near,Jc.distanceToPoint(Sn.origin));l=Yp(s,u,c.resolution)}if(Jc.radius+=l,Sn.intersectsSphere(Jc)===!1)return;a.boundingBox===null&&a.computeBoundingBox(),Zc.copy(a.boundingBox).applyMatrix4(o);let h;if(i)h=Hs*.5;else{let u=Math.max(s.near,Zc.distanceToPoint(Sn.origin));h=Yp(s,u,c.resolution)}Zc.expandByScalar(h),Sn.intersectsBox(Zc)!==!1&&(i?bM(this,t):EM(this,s,t))}onBeforeRender(e){let t=this.material.uniforms;t&&t.resolution&&(e.getViewport(ed),this.material.uniforms.resolution.value.set(ed.z,ed.w))}};var ua=new P;function Vi(n,e,t,i,s,r){let o=2*Math.PI*s/4,a=Math.max(r-2*s,0),c=Math.PI/4;ua.copy(e),ua[i]=0,ua.normalize();let l=.5*o/(o+a),h=1-ua.angleTo(n)/c;return Math.sign(ua[t])===1?h*l:a/(o+a)+l+l*(1-h)}var Kc=class n extends Bt{constructor(e=1,t=1,i=1,s=2,r=.1){let o=s*2+1;if(r=Math.min(e/2,t/2,i/2,r),super(1,1,1,o,o,o),this.type="RoundedBoxGeometry",this.parameters={width:e,height:t,depth:i,segments:s,radius:r},o===1)return;let a=this.toNonIndexed();this.index=null,this.attributes.position=a.attributes.position,this.attributes.normal=a.attributes.normal,this.attributes.uv=a.attributes.uv;let c=new P,l=new P,h=new P(e,t,i).divideScalar(2).subScalar(r),u=this.attributes.position.array,d=this.attributes.normal.array,f=this.attributes.uv.array,g=u.length/6,x=new P,p=.5/o;for(let m=0,M=0;m<u.length;m+=3,M+=2)switch(c.fromArray(u,m),l.copy(c),l.x-=Math.sign(l.x)*p,l.y-=Math.sign(l.y)*p,l.z-=Math.sign(l.z)*p,l.normalize(),u[m+0]=h.x*Math.sign(c.x)+l.x*r,u[m+1]=h.y*Math.sign(c.y)+l.y*r,u[m+2]=h.z*Math.sign(c.z)+l.z*r,d[m+0]=l.x,d[m+1]=l.y,d[m+2]=l.z,Math.floor(m/g)){case 0:x.set(1,0,0),f[M+0]=Vi(x,l,"z","y",r,i),f[M+1]=1-Vi(x,l,"y","z",r,t);break;case 1:x.set(-1,0,0),f[M+0]=1-Vi(x,l,"z","y",r,i),f[M+1]=1-Vi(x,l,"y","z",r,t);break;case 2:x.set(0,1,0),f[M+0]=1-Vi(x,l,"x","z",r,e),f[M+1]=Vi(x,l,"z","x",r,i);break;case 3:x.set(0,-1,0),f[M+0]=1-Vi(x,l,"x","z",r,e),f[M+1]=1-Vi(x,l,"z","x",r,i);break;case 4:x.set(0,0,1),f[M+0]=1-Vi(x,l,"x","y",r,e),f[M+1]=1-Vi(x,l,"y","x",r,t);break;case 5:x.set(0,0,-1),f[M+0]=Vi(x,l,"x","y",r,e),f[M+1]=1-Vi(x,l,"y","x",r,t);break}}static fromJSON(e){return new n(e.width,e.height,e.depth,e.segments,e.radius)}};var Qp=[16756767,3262128,16740193,10194175],nd=new Map;function Gt(n,e={}){let t=`${n}:${JSON.stringify(e)}`;return nd.has(t)||nd.set(t,new Qe({color:n,roughness:.52,metalness:0,...e})),nd.get(t)}var We={dark:Gt(3883079,{roughness:.7}),darker:Gt(2830133,{roughness:.8}),rubber:Gt(2763824,{roughness:.95}),metal:Gt(10989748,{roughness:.35,metalness:.55}),chrome:Gt(14936298,{roughness:.18,metalness:.85}),glass:Gt(8242390,{roughness:.08,metalness:.1,emissive:1454650,emissiveIntensity:.35}),seat:Gt(3093304,{roughness:.9}),soil:Gt(11039551,{roughness:1,flatShading:!0}),lamp:Gt(16773570,{emissive:16770720,emissiveIntensity:.9}),tail:Gt(16730682,{emissive:12590608,emissiveIntensity:.6}),white:Gt(16052712,{roughness:.6})},wM=[{body:16036379,accent:16765788,trim:3883079},{body:14964026,accent:16165179,trim:3883079,bed:15903035},{body:16022304,accent:16752717,trim:3093304}],$p=[14721067,4158630,5934942,9071536],TM=13194813,AM=14645804,sd=new Map;function RM(n){return sd.has(n)||sd.set(n,new No({color:n,roughness:.4,metalness:.05,clearcoat:.55,clearcoatRoughness:.28})),sd.get(n)}var wi={glass:Gt(3820888,{roughness:.07,metalness:.25}),steel:Gt(4935766,{roughness:.52,metalness:.3}),worn:Gt(10330790,{roughness:.32,metalness:.8}),lamp:Gt(16183256,{emissive:16771512,emissiveIntensity:.25}),soil:Gt(7295544,{roughness:1,flatShading:!0})};function CM(n,e){if(e!=="studio")return{...wM[n.type],paint:Gt};let t=n.type===0?$p[n.id%$p.length]:n.type===1?TM:AM;return{body:t,accent:t,trim:3027511,bed:t,studio:!0,paint:RM}}var ud=new Bt(1,1,1),Vs=new Ki(1,1,1,24),dd=new Ki(1,1,1,10);function Zt(n,e,t,i=0,s=0,r=0){let o=new Ke(e,t);return o.position.set(i,s,r),o.castShadow=!0,o.receiveShadow=!0,n.add(o),o}function ot(n,e,t,i,s,r,o,a){let c=Zt(n,ud,e,t,i,s);return c.scale.set(r,o,a),c}function Gi(n,e,t,i,s,r,o,a,c){return Zt(n,new Kc(r,o,a,3,Math.min(c,r/2,o/2,a/2)),e,t,i,s)}function xi(n,e,t,i,s,r,o,a=0,c=Vs){let l=Zt(n,c,e,t,i,s);return l.scale.set(r,o,r),l.rotation.x=a,l}function da(n,e,t,i,s){let r=new P(...t),o=new P(...i),a=o.clone().sub(r),c=Zt(n,Vs,e);return c.position.copy(r.add(o).multiplyScalar(.5)),c.scale.set(s,a.length(),s),c.quaternion.setFromUnitVectors(new P(0,1,0),a.normalize()),c}function rd(n,e,t,i,s,r=!1){let o=new gn;o.moveTo(0,-t*.4),r?(o.quadraticCurveTo(-t*.14,t*.15,e*.18,t*.5),o.quadraticCurveTo(e*.24,t*.6,e*.33,t*.48),o.lineTo(e*.89,t*.2),o.quadraticCurveTo(e+t*.2,t*.2,e+t*.15,-t*.08),o.quadraticCurveTo(e+t*.1,-t*.4,e*.9,-t*.32),o.lineTo(e*.26,-t*.25)):(o.lineTo(e*.22,t*.55),o.lineTo(e*.7,t*.3),o.lineTo(e,t*.12),o.lineTo(e,-t*.32),o.lineTo(e*.24,-t*.25)),o.closePath();let a=new Fn(o,{depth:i,bevelEnabled:!0,bevelSegments:r?3:1,curveSegments:10,steps:1,bevelSize:t*.085,bevelThickness:t*.085});return Zt(n,a,s,0,0,-i/2)}function ui(n,e,t,i,s){let r=new ft;return r.name=e,r.position.set(t,i,s),n.add(r),r}function _i(n,e,t,i,s,r,o){return xi(n,e,t,i,s,r,o,Math.PI/2)}function th(n,e,t,i){let s=t.clone().sub(e);n.position.copy(e).add(t).multiplyScalar(.5),n.scale.set(i,s.length(),i),n.quaternion.setFromUnitVectors(new P(0,1,0),s.normalize())}function od(n,e,t,i,s){let r=Zt(n,Vs,We.dark),o=Zt(n,Vs,We.chrome);return r.name=`${s}-barrel`,o.name=`${s}-piston`,{start:e,end:t,barrel:r,piston:o,update(){let a=n.worldToLocal(e.getWorldPosition(new P)),c=n.worldToLocal(t.getWorldPosition(new P));th(r,a,a.clone().lerp(c,.58),i),th(o,a.clone().lerp(c,.43),c,i*.56)}}}function PM(n,e,t,i){let s=e.x-n.x,r=e.y-n.y,o=Math.hypot(s,r);if(o<=Math.abs(t-i)||o>=t+i)throw new Error("Bucket linkage pose is outside its mechanical range.");let a=(t**2-i**2+o**2)/(2*o),c=Math.sqrt(Math.max(0,t**2-a**2));return new P(n.x+(a*s-c*r)/o,n.y+(a*r+c*s)/o,0)}function fd(n,e,t){if(typeof document>"u")return null;let i=document.createElement("canvas");i.width=n,i.height=e,t(i.getContext("2d"));let s=new pn(i);return s.colorSpace=Lt,s}function IM(){return fd(128,32,n=>{n.fillStyle="#f4b21b",n.fillRect(0,0,128,32),n.fillStyle="#2b2f35";for(let e=-32;e<160;e+=24)n.beginPath(),n.moveTo(e,32),n.lineTo(e+12,32),n.lineTo(e+44,0),n.lineTo(e+32,0),n.fill()})}var ad;function DM(n,e,t,i,s,r){let o=new et;o.position.set(e,i,t),n.add(o);let a=new et;o.add(a),xi(a,We.rubber,0,0,0,i*.9,s,Math.PI/2);let c=Math.sign(t)||1;xi(a,r,0,0,c*s*.47,i*.56,s*.12,Math.PI/2),xi(a,We.metal,0,0,c*s*.54,i*.2,s*.1,Math.PI/2,dd);for(let l=0;l<6;l++){let h=l*Math.PI/3;ot(a,We.darker,Math.cos(h)*i*.36,Math.sin(h)*i*.36,c*s*.53,i*.09,i*.09,s*.05)}for(let l=0;l<14;l++){let h=l*Math.PI*2/14,u=ot(a,We.rubber,Math.sin(h)*i*.93,Math.cos(h)*i*.93,(l%2?.18:-.18)*s,i*.26,i*.16,s*.6);u.rotation.z=-h}return{steer:o,spin:a,radius:i}}function LM(n,e,t,i,s,r){let o=new et;o.position.z=s*t*.35,n.add(o);let a=i*.13,c=e*.34,l=i*.02,h=a+l,u=t*.2,d=4*c+2*Math.PI*a,f=new gn;f.moveTo(-c,l+a*.35),f.lineTo(c,l+a*.35),f.absarc(c,h,a*.65,-Math.PI/2,Math.PI/2,!1),f.lineTo(-c,h+a*.65),f.absarc(-c,h,a*.65,Math.PI/2,Math.PI*1.5,!1);let g=new Fn(f,{depth:u*.7,bevelEnabled:!0,bevelSegments:2,bevelSize:i*.012,bevelThickness:i*.012,steps:1});Zt(o,g,r,0,0,-u*.35);let x=[];for(let _=0;_<5;_++)x.push(xi(o,We.metal,e*(-.26+_*.13),l+a*.42,s*u*.38,i*.045,u*.12,Math.PI/2));for(let _ of[-c,c]){let E=new et;E.position.set(_,h,0),o.add(E),x.push(E),xi(E,We.dark,0,0,0,a*.82,u*.86,Math.PI/2),xi(E,We.metal,0,0,s*u*.44,a*.38,u*.06,Math.PI/2,dd);for(let C=0;C<8;C++){let I=C*Math.PI/4;ot(E,We.darker,Math.cos(I)*a*.6,Math.sin(I)*a*.6,s*u*.44,a*.14,a*.14,u*.04)}}let p=Math.max(24,Math.round(d/(i*.055))),m=d/p,M=new jt(ud,We.rubber,p);M.castShadow=!0,M.receiveShadow=!0,o.add(M);let b=new ft,y=i*.034,T=(_,E)=>{if(_=(_%d+d)%d,_<2*c){E.set(-c+_,l,-Math.PI/2);return}if(_-=2*c,_<Math.PI*a){let I=-Math.PI/2+_/a;E.set(c+Math.cos(I)*a,h+Math.sin(I)*a,I);return}if(_-=Math.PI*a,_<2*c){E.set(c-_,h+a,Math.PI/2);return}_-=2*c;let C=Math.PI/2+_/a;E.set(-c+Math.cos(C)*a,h+Math.sin(C)*a,C)},S=new P,A=_=>{for(let E=0;E<p;E++){T(E*m+_,S);let C=S.z;b.position.set(S.x+Math.cos(C)*y*.4,S.y+Math.sin(C)*y*.4,0),b.rotation.set(0,0,C-Math.PI/2),b.scale.set(m*.82,y,u),b.updateMatrix(),M.setMatrixAt(E,b.matrix)}M.instanceMatrix.needsUpdate=!0;for(let E of x)E.isGroup&&(E.rotation.z=-_/(a*.82))};return A(0),{update:A,shoes:M}}function UM(n,e,t,i,s,r,o){let a=[],c=[],l=[];if(ot(n,We.dark,0,i*.22,0,e*.74,i*.17,t*.6),s){let h=i*(o?.23:.19),u=t*(o?.22:.18);for(let d of[-.3,.3])for(let f of[-.4,.4]){let g=DM(n,d*e,f*t,h,u,Gt(r.accent));c.push(g),d>0&&a.push(g.steer)}for(let d of[-.3,.3])da(n,We.darker,[d*e,h,-.4*t],[d*e,h,.4*t],i*.05)}else for(let h of[-1,1])l.push({side:h,...LM(n,e,t,i,h,Gt(r.trim,{roughness:.7}))});return{steering:a,spinning:c,tracks:l}}function ld(n,e,t,i,s,r,o,a="x"){let c=Gt(16777215,{transparent:!0,opacity:.45,emissive:16777215,emissiveIntensity:.4,depthWrite:!1});for(let[l,h]of[[-.18,.16],[.1,.07]]){let u=ot(n,c,e,t,i,s,r,o);u.castShadow=!1,a==="x"?(u.scale.set(s,r*1.2,o*h),u.position.z+=o*l*2.2,u.rotation.x=.5):(u.scale.set(s*h,r*1.2,o),u.position.x+=s*l*2.2,u.rotation.z=-.5),u.userData.skipAO=!0,u.name="glass-highlight"}}function Zp(n,e,t,i,s,r,o){let a=o.paint(o.body),c=new et;c.position.set(s,0,r),n.add(c);let l=.3*e,h=.39*t,u=.5*i,d=o.studio?wi.glass:We.glass;Gi(c,a,0,.12*i,0,l,.2*i,h,i*.04),Gi(c,a,0,.36*i,0,l*.96,u*.72,h*.96,i*.05).name="cab-shell";let f=.39*i,g=u*.56;ot(c,d,l*.485,f,0,i*.012,g,h*.84),ld(c,l*.492,f,0,i*.01,g*.8,h*.84,"x");for(let x of[-1,1])ot(c,d,-l*.04,f,x*h*.485,l*.76,g,i*.012),ld(c,-l*.04,f,x*h*.492,l*.76,g*.8,i*.01,"z");ot(c,d,-l*.485,f+g*.1,0,i*.012,g*.6,h*.7),ot(c,We.seat,-l*.12,.3*i,0,l*.3,.16*i,h*.5),Gi(c,o.paint(o.studio?o.body:o.trim),0,.62*i,0,l*(o.studio?.98:1.06),i*.05,h*(o.studio?.98:1.06),i*.02);for(let x of[-1,1])ot(c,o.studio?wi.lamp:We.lamp,l*.5,.6*i,x*h*.3,i*.02,i*.035,h*.12);return c}function Hr(n,e,t,i=12){let s=new Fn(n,{depth:e,bevelEnabled:t>0,bevelSegments:2,bevelSize:t,bevelThickness:t,curveSegments:i,steps:1});return s.translate(0,0,-e/2),s}function Vr(n){let e=new gn;return e.setFromPoints(n),e}function em(n,e,t){let i=n.map((s,r)=>{let o=n[Math.max(0,r-1)],a=n[Math.min(n.length-1,r+1)],c=new $(o.y-a.y,a.x-o.x).normalize();return c.dot(t.clone().sub(s))<0&&c.negate(),s.clone().addScaledVector(c,e)});return[...n,...i.reverse()]}function NM(n){let e=[...n].sort((r,o)=>r.x-o.x||r.y-o.y),t=(r,o,a)=>(o.x-r.x)*(a.y-r.y)-(o.y-r.y)*(a.x-r.x),i=[],s=[];for(let r of e){for(;i.length>1&&t(i.at(-2),i.at(-1),r)<=0;)i.pop();i.push(r)}for(let r of e.reverse()){for(;s.length>1&&t(s.at(-2),s.at(-1),r)<=0;)s.pop();s.push(r)}return[...i.slice(0,-1),...s.slice(0,-1)]}function FM(n,e,t,i){let s=new et;n.add(s),s.name="loader-bucket";let r=e,o=i.studio?wi.steel:We.dark,a=i.studio?wi.steel:i.paint(i.body),c=i.studio?wi.worn:We.metal,l=new mn;l.moveTo(.29*r,-.155*r),l.lineTo(-.05*r,-.135*r),l.quadraticCurveTo(-.16*r,-.13*r,-.165*r,0*r),l.quadraticCurveTo(-.17*r,.13*r,-.12*r,.185*r);let h=l.getPoints(10),u=new $(.05*r,.02*r);Zt(s,Hr(Vr(em(h,.022*r,u)),t*.97,.004*r),o).name="loader-bucket-shell";let d=Vr([...h,new $(-.04*r,.2*r),new $(.07*r,.2*r)]),f=Hr(d,t*.045,.006*r);for(let p of[-1,1])Zt(s,f,a,0,0,p*t*.49);ot(s,o,0*r,.195*r,0,.17*r,.02*r,t*.99).rotation.z=.08;let g=ot(s,c,.31*r,-.157*r,0,.07*r,.022*r,t*1);g.rotation.z=-.06,ot(s,o,-.19*r,.03*r,0,.03*r,.26*r,t*.62);for(let p of[-1,1])ot(s,o,-.21*r,.03*r,p*t*.2,.05*r,.24*r,.04*r);ui(s,"loader-edge",.34*r,-.15*r,0),ui(s,"loader-lip",.16*r,.05*r,0);let x=Zt(s,pd(3),We.soil,.06*r,-.06*r,0);return x.name="bucket-soil",x.scale.set(r*.2,r*.14,t*.42),x.visible=!1,{root:s,soil:x}}var eh=new Map;function pd(n){if(eh.has(n))return eh.get(n);let e=new Qi(1,1),t=e.attributes.position,i=n*9301+49297,s=()=>(i=i*16807%2147483647)/2147483647,r=new Map;for(let o=0;o<t.count;o++){let a=`${t.getX(o).toFixed(3)},${t.getY(o).toFixed(3)},${t.getZ(o).toFixed(3)}`;r.has(a)||r.set(a,.82+s()*.3);let c=r.get(a),l=t.getY(o);t.setXYZ(o,t.getX(o)*c,(l<0?l*.25:l)*c,t.getZ(o)*c)}return e.computeVertexNormals(),eh.set(n,e),e}function OM(n,e,t,i,s){let r=new et;r.name="bucket-curl",n.add(r);let o=new et;o.name="bucket-orientation",o.rotation.y=Math.PI,r.add(o);let a=e,c=s.studio?wi.steel:We.dark,l=s.studio?wi.steel:s.paint(s.body),h=s.studio?wi.worn:We.metal,u=s.studio?wi.steel:s.paint(s.accent),d=new $(-.205*a,-.07*a),f=new $(.22*a,-.53*a),g=new $(.43*a,-.41*a),x=new $(.125*a,-.05*a),p=new mn;p.moveTo(d.x,d.y),p.bezierCurveTo(-.33*a,-.15*a,-.345*a,-.36*a,-.245*a,-.47*a),p.bezierCurveTo(-.14*a,-.585*a,.07*a,-.61*a,f.x,f.y),p.lineTo(g.x,g.y);let m=p.getPoints(14),M=new $(.07*a,-.3*a);Zt(o,Hr(Vr(em(m,.03*a,M)),t*.96,.005*a,16),c).name="bucket-shell";let b=Hr(Vr([...m,x,d]),t*.05,.006*a,16);for(let j of[-1,1])Zt(o,b,l,0,0,j*t*.475).name=`bucket-side-${j}`;let y=ot(o,c,(d.x+x.x)/2,(d.y+x.y)/2-.012*a,0,x.distanceTo(d)+.02*a,.028*a,t*.97);y.rotation.z=Math.atan2(x.y-d.y,x.x-d.x);for(let j of[-.32,.32]){let he=ot(o,h,-.06*a,-.585*a,j*t,.26*a,.018*a,t*.07);he.rotation.z=-.12}let T=g.clone().sub(f).normalize(),S=Math.atan2(T.y,T.x),A=ot(o,h,g.x-T.x*.02*a,g.y-T.y*.02*a-.006*a,0,.11*a,.032*a,t*1);A.rotation.z=S,A.name="bucket-cutting-edge";let _=Vr([[0,.026],[.07,.021],[.13,.006],[.145,-.001],[.075,-.014],[0,-.02]].map(([j,he])=>new $(j*a,he*a))),E=Math.min(.055*a,t*.13),C=Hr(_,E,.005*a,4),I=t>.3*a?5:4;for(let j=0;j<I;j++){let he=(j/(I-1)-.5)*t*.84,le=g.clone().addScaledVector(T,.03*a),Ae=ot(o,c,g.x,g.y+.004*a,he,.075*a,.045*a,E*1.35);Ae.rotation.z=S;let Fe=Zt(o,C,h,le.x,le.y,he);Fe.rotation.z=S,Fe.name="bucket-tooth"}let L=g.clone().addScaledVector(T,.175*a),V=t*.075,q=a*.008,N=i+a*.02,Y=(N+V)/2+q,X=Math.max(t*.72,N+2*V+4*q+a*.014),ne=new $(-.08*a,.12*a),ie=(j,he)=>Array.from({length:20},(le,Ae)=>j.clone().add(new $(Math.cos(Ae/20*Math.PI*2)*he,Math.sin(Ae/20*Math.PI*2)*he))),ge=NM([...ie(new $(0,0),.075*a),...ie(ne,.06*a),new $(.09*a,-.06*a),new $(-.175*a,-.075*a)]),ue=Hr(Vr(ge),V,q,10);for(let j of[-1,1])Zt(o,ue,u,0,0,j*Y).name=`bucket-ear-${j}`;_i(o,h,0,0,0,a*.045,X).name="bucket-main-pin";let xe=ui(o,"bucket-link-pin",ne.x,ne.y,0);_i(xe,h,0,0,0,a*.032,X),ui(o,"bucket-teeth",L.x,L.y,0);let Ne=new $((g.x+x.x)/2-.04*a,(g.y+x.y)/2),st=new $(x.y-g.y,g.x-x.x).normalize();o.userData.opening=new P(st.x,st.y,0),ui(o,"bucket-lip",Ne.x,Ne.y,0);let Xe=Zt(o,pd(1),We.soil,.09*a,-.28*a,0);return Xe.name="bucket-soil",Xe.scale.set(a*.25,a*.2,t*.4),Xe.visible=!1,{curl:r,orientation:o,soil:Xe,linkPin:xe,toothTip:L,lipPoint:Ne}}function BM(n,e){let t=`#${Qp[n.id%4].toString(16).padStart(6,"0")}`,i=String(n.id+1).padStart(2,"0"),s=fd(160,96,o=>{if(o.textAlign="center",o.textBaseline="middle",e==="paper"){o.font='600 44px Inter, "Helvetica Neue", Arial, sans-serif',o.lineJoin="round",o.lineWidth=10,o.strokeStyle="#ffffffee",o.strokeText(i,80,38),o.fillStyle="#1f2426",o.fillText(i,80,38),o.fillStyle="#ffffffee",o.beginPath(),o.roundRect(52,66,56,14,7),o.fill(),o.fillStyle=t,o.beginPath(),o.roundRect(56,69,48,8,4),o.fill();return}o.fillStyle="#00000033",o.beginPath(),o.roundRect(22,12,116,58,29),o.fill(),o.fillStyle=t,o.beginPath(),o.roundRect(20,8,120,58,29),o.fill(),o.beginPath(),o.moveTo(68,62),o.lineTo(92,62),o.lineTo(80,80),o.closePath(),o.fill(),o.lineWidth=5,o.strokeStyle="#ffffffcc",o.beginPath(),o.roundRect(22.5,10.5,115,53,26.5),o.stroke(),o.font='800 38px ui-rounded, "SF Pro Rounded", system-ui, sans-serif',o.fillStyle="#1f2a2c",o.fillText(i,80,39)}),r=new _o(new Sr({map:s,depthTest:!1,transparent:!0,sizeAttenuation:!1}));return r.center.set(.5,0),r.renderOrder=30,r.scale.set(.05,.03,1),r}function zM(n,e=!0){let t=`#${n.toString(16).padStart(6,"0")}`,i=fd(256,256,r=>{if(e){let o=r.createRadialGradient(128,128,60,128,128,126);o.addColorStop(0,`${t}00`),o.addColorStop(.82,`${t}38`),o.addColorStop(1,`${t}00`),r.fillStyle=o,r.fillRect(0,0,256,256)}r.strokeStyle=t,r.lineWidth=9,r.lineCap="round";for(let o=0;o<16;o++)r.beginPath(),r.arc(128,128,112,o*Math.PI/8+.06,(o+.62)*Math.PI/8),r.stroke()}),s=new Ke(new ki(1,1),new Ln({map:i,transparent:!0,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-4}));return s.rotation.x=-Math.PI/2,s.renderOrder=16,s.userData.skipAO=!0,s}var $t=n=>n*n*(3-2*n),ih=n=>Math.min(1,Math.max(0,n)),cd=n=>1+(1.9+1)*(n-1)**3+1.9*(n-1)**2,Yt=(n,e,t)=>ih((n-e)/(t-e));function kM(n,e,t){for(let i=1;i<e.length;i++){let[s,r,o=$t]=e[i],[a,c]=e[i-1];if(t<=s||i===e.length-1){let l=o(Yt(t,a,s)),h=n[c],u=n[r];return Object.fromEntries(Object.keys(h).map(d=>[d,h[d]+(u[d]-h[d])*l]))}}return n[e[0][1]]}var kr={carry:{boom:.86,stick:-1.92,pitch:.04},reach:{boom:.3,stick:-1.05,pitch:-.42},scoop:{boom:.2,stick:-1.3,pitch:.62},raise:{boom:.8,stick:-1.02,pitch:.1},pour:{boom:.74,stick:-.98,pitch:.94}},Jp={dig:[[0,"carry"],[.3,"reach"],[.56,"scoop"],[1,"carry",cd]],dump:[[0,"carry"],[.34,"raise"],[.62,"pour"],[1,"carry",cd]]},_s={empty:-.12,loaded:-.5},Ei={ground:{arm:-.33,curl:-.06},raised:{arm:.45},tipped:{curl:-.95}},nn=n=>n<.5?4*n*n*n:1-(-2*n+2)**3/2,HM=n=>n*n,jp=n=>1-(1-n)*(1-n),Qc=n=>.75*n+.25*$t(n),fa=(n,e,t)=>Math.min(t,Math.max(e,n));function Kp(n,e){let t=1;for(;t<n.length-1&&e>n[t].t;)t++;let i=n[t-1],s=n[t],r=(s.ease??$t)(ih((e-i.t)/Math.max(1e-6,s.t-i.t))),o={};for(let a of Object.keys(s))a==="t"||a==="ease"||(o[a]=s[a]?.isVector2?i[a].clone().lerp(s[a],r):i[a]+(s[a]-i[a])*r);return o}var hd=(n,e)=>new $(n.x*Math.cos(e)-n.y*Math.sin(e),n.x*Math.sin(e)+n.y*Math.cos(e));function VM(n,e,t,i,s){let r=new $(n.position.x,n.position.y),o=u=>new $(-u.x,u.y),a=o(i.toothTip),c=o(i.lipPoint),l=(u,d)=>new $(e*Math.cos(u)+t*Math.cos(u+d),e*Math.sin(u)+t*Math.sin(u+d));function h(u){let d=u.clone(),f=(e+t)*.995,g=Math.abs(e-t)*1.05+1e-6,x=d.length();x>f?d.multiplyScalar(f/x):x<g&&d.multiplyScalar(g/Math.max(x,1e-6));let p=d.length(),m=-Math.acos(fa((p*p-e*e-t*t)/(2*e*t),-1,1));return{boom:Math.atan2(d.y,d.x)-Math.atan2(t*Math.sin(m),e+t*Math.cos(m)),stick:m}}return{pivot:r,size:s,hinge:l,solveHinge:h,teeth:a,lip:c,tip:(u,d,f)=>l(u,d).add(hd(a,f)),solveTip:(u,d)=>h(u.clone().sub(hd(a,d)))}}function tm(n,e,{labels:t=!0,style:i="diorama"}={}){let s=new et,r=n.height*e,o=n.width*e,a=Math.min(r,o),c=CM(n,i),l=n.action_type===1||n.type===1,h=i==="studio";s.name=`machine-${n.id}`;let u=UM(s,r,o,a,l,c,n.type===1),d=new et;d.name="suspension",s.add(d);let f=new et;f.position.y=.32*a,d.add(f);let g=c.paint(c.body),x=c.paint(c.accent),p=c.paint(c.trim),m=h?wi.soil:We.soil,M=h?wi.steel:g,b=h?wi.steel:x,y=new Qe({color:16753183,roughness:.3,emissive:16742912,emissiveIntensity:.2,transparent:!0,opacity:.92}),T,S,A,_,E,C,I,L,V,q=null,N=null,Y=null,X=[];if(ad||(ad=new Qe({map:IM(),color:typeof document>"u"?16036379:16777215,roughness:.6})),n.type===0){xi(f,We.dark,0,.045*a,0,a*.31,a*.1),Gi(f,g,-.1*r,.16*a,0,r*.65,a*.23,o*.66,a*.065).name="excavator-upper-body",Gi(f,h?p:We.dark,-.33*r,.255*a,0,r*.2,a*.2,o*.65,a*.068).name="excavator-counterweight",ot(f,h?p:ad,-.434*r,.255*a,0,r*.012,a*.09,o*.56);for(let v of[-1,1])ot(f,We.tail,-.434*r,.3*a,v*o*.29,r*.012,a*.03,o*.05);Gi(f,x,-.22*r,.29*a,.16*o,r*.26,a*.05,o*.3,a*.02);for(let v=0;v<5;v++)ot(f,We.darker,(-.3+v*.04)*r,.318*a,.16*o,r*.018,a*.012,o*.22);Zp(f,r,o,a,-.02*r,-.18*o,c),V=xi(f,y,-.1*r,.7*a,-.18*o,a*.032,a*.06),xi(f,We.dark,-.1*r,.665*a,-.18*o,a*.04,a*.02),xi(f,We.dark,-.28*r,.45*a,.26*o,a*.026,a*.32),xi(f,We.darker,-.28*r,.62*a,.26*o,a*.034,a*.03),L=ui(f,"exhaust",-.28*r,.66*a,.26*o),da(f,We.metal,[-.33*r,.4*a,.32*o],[-.12*r,.4*a,.32*o],a*.012);for(let v of[-.33,-.12])da(f,We.metal,[v*r,.27*a,.32*o],[v*r,.4*a,.32*o],a*.012);let ae=Math.max(r*.63,n.reach[1]*e*.4),ee=Math.max(r*.48,n.reach[1]*e*.32);T=new et,T.name="boom-pivot",T.position.set(.16*r,.23*a,.09*o),f.add(T),rd(T,ae,a*.21,o*.12,x,!0),_i(T,We.dark,0,0,0,a*.095,o*.19),_i(T,We.metal,0,0,0,a*.05,o*.205);for(let v of[-1,1])ot(T,h?wi.lamp:We.lamp,ae*.3,a*.1,v*o*.065,a*.04,a*.03,a*.012);for(let v of[-1,1]){let U=ui(f,`boom-cylinder-${v}-start`,.2*r,.12*a,(.09+v*.12)*o),B=ui(T,`boom-cylinder-${v}-end`,ae*.48,-.055*a,v*o*.12);_i(U,M,0,0,0,a*.047,o*.055),_i(B,M,0,0,0,a*.047,o*.055),X.push(od(f,U,B,a*.036,`boom-cylinder-${v}`))}S=new et,S.name="stick-pivot",S.position.x=ae,T.add(S),rd(S,ee,a*.17,o*.09,g,!0),Gi(S,g,-.07*ee,.045*a,0,.22*ee,a*.105,o*.09,a*.035),_i(S,We.dark,0,0,0,a*.078,o*.16),_i(S,We.metal,0,0,0,a*.04,o*.175);let O=ui(T,"stick-cylinder-start",ae*.4,a*.145,0),H=ui(S,"stick-cylinder-end",-.09*ee,a*.08,0);_i(O,b,0,0,0,a*.048,o*.11),_i(H,M,0,0,0,a*.043,o*.115),X.push(od(f,O,H,a*.04,"stick-cylinder"));let Q=c.studio?.56:.65,W=a*Q,G=o*.39*Q,se=o*.145,ce=OM(S,W,G,se,c);A=ce.curl,A.position.x=ee,_=ce.soil,_.material=m,N=ce.orientation.getObjectByName("bucket-teeth"),Y=ce.orientation.getObjectByName("bucket-lip"),ui(S,"bucket-hinge",ee,0,0),_i(S,g,ee,0,0,W*.068,se).name="bucket-hinge-housing";let fe=ui(S,"bucket-rocker-pivot",ee-W*.24,W*.1,0);Gi(S,g,fe.position.x,.04*W,0,W*.115,W*.17,o*.105,W*.025),_i(fe,We.metal,0,0,0,W*.036,G*.72);let me=ui(S,"bucket-rocker-joint",0,0,0);_i(me,We.metal,0,0,0,W*.036,G*.72);let D=W*.22,Me=W*.25,Ve=[];for(let v of[-1,1]){let U=Zt(S,Vs,g),B=Zt(S,Vs,We.dark);U.name=`bucket-rocker-${v}`,B.name=`bucket-link-${v}`,Ve.push({first:U,second:B,z:v*G*.3})}let R=ui(S,"bucket-cylinder-start",ee*.24,a*.12,0);_i(R,M,0,0,0,W*.038,o*.115),X.push(od(S,R,me,W*.03,"bucket-cylinder")),I={origin:fe,joint:me,destination:ce.linkPin,firstLength:D,secondLength:Me,links:Ve,update(){let v=S.worldToLocal(ce.linkPin.getWorldPosition(new P)),U=PM(fe.position,v,D,Me);me.position.copy(U);for(let B of Ve){let k=fe.position.clone(),pe=v.clone(),_e=U.clone();k.z=B.z,_e.z=B.z,pe.z=B.z,th(B.first,k,_e,W*.029),th(B.second,_e,pe,W*.025)}}},q=VM(T,ae,ee,ce,W)}else if(n.type===1){ot(f,We.dark,0,.02*a,0,r*.92,.09*a,o*.62);for(let H of[-1,1])ot(f,p,.05*r,.08*a,H*o*.44,r*.7,.05*a,o*.1);Gi(f,g,.36*r,.16*a,0,.22*r,.26*a,o*.82,a*.05),ot(f,We.darker,.475*r,.15*a,0,r*.02,.15*a,o*.5);for(let H=0;H<4;H++)ot(f,We.metal,.486*r,(.1+H*.035)*a,0,r*.01,a*.012,o*.44);for(let H of[-1,1])ot(f,We.lamp,.478*r,.24*a,H*o*.32,r*.02,.05*a,o*.1),ot(f,We.chrome,.44*r,.06*a,H*o*.37,r*.08,.04*a,o*.12);Zp(f,r*.92,o*1.62,a*.95,.3*r,-.14*o,c),L=ui(f,"exhaust",.2*r,.78*a,.3*o),xi(f,We.chrome,.2*r,.5*a,.3*o,a*.03,a*.52);let ae=new et;ae.name="truck-bed",ae.position.set(-.43*r,.12*a,0),f.add(ae),E=ae;let ee=c.paint(c.bed);ot(ae,ee,.3*r,0,0,r*.64,a*.08,o*.86);for(let H of[-1,1]){let Q=ot(ae,ee,.3*r,.2*a,H*o*.41,.66*r,a*.38,o*.05);Q.rotation.x=H*.08;for(let W=0;W<4;W++)ot(ae,x,(.06+W*.16)*r,.22*a,H*o*.44,r*.025,a*.34,o*.02);ot(ae,x,.3*r,.4*a,H*o*.43,.66*r,a*.04,o*.07)}ot(ae,ee,.62*r,.28*a,0,.04*r,a*.52,o*.86);let O=ot(ae,ee,.72*r,.52*a,0,.22*r,a*.04,o*.86);O.rotation.z=-.06,ot(ae,We.dark,-.02*r,.22*a,0,.03*r,a*.3,o*.78),_=Zt(ae,pd(2),m,r*.3,a*.2,0),_.scale.set(r*.27,a*.2,o*.33);for(let H of[-1,1])ot(f,We.tail,-.47*r,.05*a,H*.32*o,.02*r,.05*a,.1*o)}else{Gi(f,g,-.06*r,.12*a,0,.72*r,.26*a,.64*o,a*.05),Gi(f,We.dark,-.34*r,.2*a,0,.14*r,.22*a,.6*o,a*.04);for(let W=0;W<4;W++)ot(f,We.darker,-.412*r,(.12+W*.045)*a,0,r*.01,a*.018,o*.46);let ae=new et;ae.position.set(-.06*r,.25*a,0),f.add(ae);let ee=.34*r,O=.4*o,H=.46*a;for(let W of[-1,1])for(let G of[-1,1])ot(ae,We.darker,W*ee*.47,H/2,G*O*.47,a*.035,H,a*.035);Gi(ae,g,0,H,0,ee*1.06,a*.05,O*1.08,a*.02),ot(ae,h?wi.glass:We.glass,ee*.47,H*.52,0,a*.01,H*.78,O*.86),ld(ae,ee*.478,H*.52,0,a*.01,H*.6,O*.86,"x");for(let W of[-1,1])for(let G=0;G<4;G++)ot(ae,We.darker,(-.3+G*.2)*ee,H*.55,W*O*.47,a*.012,H*.8,a*.012);ot(ae,We.seat,-ee*.1,H*.25,0,ee*.35,H*.3,O*.5),V=xi(ae,y,-ee*.3,H+a*.05,0,a*.03,a*.05),L=ui(f,"exhaust",-.36*r,.42*a,.2*o),xi(f,We.dark,-.36*r,.36*a,.2*o,a*.025,a*.14),C=new et,C.name="loader-arm",C.position.set(-.18*r,.22*a,0),f.add(C);for(let W of[-1,1]){let G=new et;G.position.z=W*o*.36,C.add(G),rd(G,r*.81,a*.13,o*.075,x),da(G,We.chrome,[r*.1,-.08*a,0],[r*.5,-.06*a,0],a*.022),_i(G,We.metal,0,0,0,a*.05,o*.09)}da(C,p,[r*.66,-.04*a,-o*.36],[r*.66,-.04*a,o*.36],a*.04);let Q=FM(C,a*1.22,o*.92,c);A=Q.root,A.position.set(.8*r,-.1*a,0),_=Q.soil,_.material=m,N=A.getObjectByName("loader-edge"),Y=A.getObjectByName("loader-lip")}V&&(V.name="beacon");let ne=Qp[n.id%4],ie=zM(ne,i==="diorama"),ge=i==="paper";i!=="diorama"&&s.traverse(ae=>{ae.name==="glass-highlight"&&(ae.visible=!1)}),ie.scale.set(r*1.34,o*1.34+(r-o)*.35,1),ie.position.y=e*.03,s.add(ie);let ue=null;t&&(ue=BM(n,i==="diorama"?"diorama":"paper"),ue.position.set(-.05*r,a*1.12,0),s.add(ue));let xe=new P,Ne={last:null,treads:[0,0],spin:0,active:!1,kind:"",phase:1,lift:0,tags:!0},st=u.spinning[0]?.radius??a*.2,Xe=_?_.scale.clone():null;function j(ae,ee){return s.updateWorldMatrix(!0,!0),ae.cells.map(O=>{let H=ee.worldToLocal(new P(O.x,O.before,O.z)),Q=ee.worldToLocal(new P(O.x,O.after,O.z));return{key:O.key,x:H.x,z:H.z,before:H.y,after:Q.y,weight:Math.max(1,Math.abs(O.delta??1))}})}let he=(ae,ee)=>ae.reduce((O,H)=>O+ee(H)*H.weight,0)/ae.reduce((O,H)=>O+H.weight,0),le=(ae,ee)=>O=>he(ae,H=>$t(Yt(O,...ee.get(H.key))));function Ae(ae){let ee=j(ae,f);if(!ee.length)return null;let O=he(ee,k=>k.x),H=he(ee,k=>k.z),Q=Math.hypot(O,H),W=T.position.z,G=Q>Math.abs(W)*1.5?fa(Math.asin(fa(W/Q,-1,1))-Math.atan2(H,O),-.6,.6):0,se=Math.cos(G),ce=Math.sin(G),fe=q.size,me=new Map;for(let k of ee)k.s=se*k.x-ce*k.z-q.pivot.x,k.before-=q.pivot.y,k.after-=q.pivot.y;let D=k=>q.tip(kr.carry.boom,kr.carry.stick,k?_s.loaded:_s.empty);if(ae.kind==="dig"){let k=Math.max(...ee.map(Ie=>Ie.s))+e*.3,pe=Math.min(...ee.map(Ie=>Ie.s))-e*.35,_e=Math.min(...ee.map(Ie=>Ie.after))-e*.05,te=Math.max(...ee.map(Ie=>Ie.before),_e+e*.2),re=[{t:0,point:D(!1),pitch:_s.empty},{t:.22,point:new $(k+fe*.12,te+fe*.45),pitch:1.25,ease:nn},{t:.34,point:new $(k,_e+e*.03),pitch:1.05,ease:HM},{t:.68,point:new $(pe,_e),pitch:.4,ease:Qc},{t:.8,point:new $(pe-fe*.08,te+fe*.3),pitch:-.55,ease:jp},{t:1,point:D(!0),pitch:_s.loaded,ease:nn}],Se=Math.max(k-pe,1e-6);for(let Ie of ee){let ve=.34+.34*ih((k-Ie.s)/Se);me.set(Ie.key,[ve-.05,ve+.07])}return{kind:"dig",space:"tip",keys:re,yaw:G,yawWindow:[.24,.84],timing:me,fill:le(ee,me),events:{bite:.34,drag:[.34,.68],breakout:.76}}}let Me=1.8,Ve=he(ee,k=>k.s),R=Math.max(...ee.map(k=>Math.max(k.before,k.after))),v=new $(Ve-hd(q.lip,Me).x,R+fe*.78),U=q.hinge(kr.carry.boom,kr.carry.stick),B=[{t:0,point:U,pitch:_s.loaded},{t:.3,point:v,pitch:-.4,ease:nn},{t:.6,point:v.clone().add(new $(0,fe*.04)),pitch:Me,ease:nn},{t:.72,point:v.clone().add(new $(0,fe*.07)),pitch:Me+.12,ease:Qc},{t:1,point:U,pitch:_s.empty,ease:nn}];for(let k of ee)me.set(k.key,[.46,.8]);return{kind:"dump",space:"hinge",keys:B,yaw:G,yawWindow:[.3,.78],timing:me,fill:k=>1-$t(Yt(k,.38,.66)),events:{pour:[.38,.7]}}}function Fe(ae){let ee=j(ae,s);if(!ee.length)return null;let O=[C.rotation.z,A.rotation.z],H=(R,v,U)=>(C.rotation.z=R,A.rotation.z=v,s.updateWorldMatrix(!0,!0),s.worldToLocal(U.getWorldPosition(new P)).x),Q=H(Ei.ground.arm,Ei.ground.curl,N),W=H(Ei.raised.arm,Ei.tipped.curl,Y);C.rotation.z=O[0],A.rotation.z=O[1];let G=Math.min(...ee.map(R=>R.x)),se=Math.max(...ee.map(R=>R.x)),ce=new Map,fe=R=>({arm:R.shovel_lifted?.35:-.2,curl:R.loaded>0?.22:0,lunge:0}),me=fe(ae.from),D=fe(ae.to);if(ae.kind==="dig"){let R=fa(G-Q+e*.35,0,3.5),v=[{t:0,...me},{t:.2,arm:Ei.ground.arm,curl:Ei.ground.curl,lunge:R*.2,ease:nn},{t:.5,arm:Ei.ground.arm,curl:Ei.ground.curl,lunge:R,ease:Qc},{t:.62,arm:Ei.ground.arm+.03,curl:.5,lunge:R,ease:jp},{t:.8,arm:-.1,curl:.38,lunge:R*.5,ease:nn},{t:1,...D,ease:nn}],U=Math.max(se-G,1e-6);for(let B of ee){let k=.3+.2*ih((B.x-G)/U);ce.set(B.key,[k-.05,k+.07])}return{kind:"dig",space:"loader",keys:v,timing:ce,fill:le(ee,ce),events:{bite:.3,drag:[.3,.55]}}}let Me=fa(he(ee,R=>R.x)-W,0,3.5),Ve=[{t:0,...me},{t:.32,arm:Ei.raised.arm,curl:.3,lunge:Me,ease:nn},{t:.55,arm:Ei.raised.arm,curl:Ei.tipped.curl,lunge:Me,ease:nn},{t:.68,arm:Ei.raised.arm-.03,curl:Ei.tipped.curl,lunge:Me,ease:Qc},{t:1,...D,ease:nn}];for(let R of ee)ce.set(R.key,[.46,.8]);return{kind:"dump",space:"loader",keys:Ve,timing:ce,fill:R=>1-$t(Yt(R,.36,.6)),events:{pour:[.36,.66]}}}function ke(ae,ee,O=0,H="",Q=null){f.rotation.y=ae.cabin_yaw;for(let G of u.steering)G.rotation.y=Math.max(-.6,Math.min(.6,ae.wheel_angle*Math.PI/9));Ne.active=ee,Ne.kind=H,Ne.phase=O,ie.visible=ee&&Ne.tags&&!h;let W=ae.loaded>0;if(Q&&Xe){let G=Q.fill(O),se=.35+.65*G;_.visible=G>.03,_.scale.set(Xe.x*(.65+.35*G),Xe.y*se,Xe.z*(.75+.25*G))}else if(_.visible=W||H==="dump"&&O<.52||H==="transfer"&&O<.52||H==="dig"&&O>.5,H==="receive"&&(_.visible=O>.62),Xe&&n.type!==0){let G=H==="dump"?Math.max(1,ae.previous_loaded??ae.loaded):ae.loaded,se=.55+.45*(1-Math.exp(-Math.max(G,1)/18));H==="dump"&&(se*=1-$t(Yt(O,.22,.5)),_.visible=O<.5),_.scale.set(Xe.x,Xe.y*Math.max(se,.02),Xe.z*(H==="dump"?.7+.3*se:1))}if(T){if(Q){let G=Kp(Q.keys,O),[se,ce]=Q.yawWindow;f.rotation.y=ae.cabin_yaw+Q.yaw*$t(Yt(O,0,se))*(1-$t(Yt(O,ce,1)));let fe=Q.space==="hinge"?q.solveHinge(G.point):q.solveTip(G.point,G.pitch);T.rotation.z=fe.boom,S.rotation.z=fe.stick,A.rotation.z=G.pitch-fe.boom-fe.stick}else{let G=H==="dig"?Jp.dig:H==="dump"||H==="transfer"?Jp.dump:null,se={...kr,carry:{...kr.carry,pitch:W?_s.loaded:_s.empty}},ce=G?kM(se,G,O):se.carry;T.rotation.z=ce.boom,S.rotation.z=ce.stick,A.rotation.z=ce.pitch-T.rotation.z-S.rotation.z}s.updateWorldMatrix(!0,!0),I.update();for(let G of X)G.update()}if(E){let G=H==="dump"?O<.45?$t(Yt(O,0,.45)):O<.7?1:1-$t(Yt(O,.7,1)):0;E.rotation.z=.62*G}if(C){if(Q){let me=Kp(Q.keys,O);C.rotation.z=me.arm,A.rotation.z=me.curl,me.lunge&&s.translateX(me.lunge);return}let G=ae.shovel_lifted?.35:-.2,se=W?.22:0,ce=G,fe=se;H==="dig"?(ce=O<.35?Vt.lerp(-.2,-.3,$t(Yt(O,0,.35))):O<.6?-.3:Vt.lerp(-.3,G,cd(Yt(O,.6,1))),fe=O<.35?-.15*$t(Yt(O,0,.35)):O<.6?Vt.lerp(-.15,.3,$t(Yt(O,.35,.6))):Vt.lerp(.3,se,$t(Yt(O,.6,1)))):H==="dump"&&(ce=O<.6?Vt.lerp(.35,.45,$t(Yt(O,0,.35))):Vt.lerp(.45,G,$t(Yt(O,.6,1))),fe=O<.3?.22*(1-Yt(O,0,.3)):O<.65?-.8*$t(Yt(O,.3,.5)):Vt.lerp(-.8,se,$t(Yt(O,.65,1)))),C.rotation.z=ce,A.rotation.z=fe}}return{root:s,agent:n,bucketRig:I,hydraulics:X,suspension:d,ringColor:ne,arm:q,setPose:ke,plan(ae){return q?Ae(ae):C?Fe(ae):null},setTags(ae){Ne.tags=ae,ue&&(ue.visible=ae),ie.visible=ae&&Ne.active&&!h},drive(ae,ee){let O=Ne.last;if(Ne.last={x:ae.x,z:ae.z,yaw:ee},!O)return;let H=ae.x-O.x,Q=ae.z-O.z,W=Math.atan2(Math.sin(ee-O.yaw),Math.cos(ee-O.yaw));if(Math.hypot(H,Q)>r*1.5||Math.abs(W)>1.2)return;let G=H*Math.cos(ee)-Q*Math.sin(ee);Ne.speed=G;for(let[se,ce]of u.tracks.entries())Ne.treads[se]-=G+ce.side*o*.35*W,ce.update(Ne.treads[se]);Ne.spin-=G/st;for(let se of u.spinning)se.spin.rotation.z=Ne.spin},tick(ae,{move:ee=null,direction:O=1,reducedMotion:H=!1}={}){let Q=Ne.active;if(ge||h){y.emissiveIntensity=h?.08:.15,ie.material.opacity=Q&&ge?.9:0,ie.rotation.z=0,ue&&(ue.material.opacity=1,ue.position.y=a*1.12),d.position.y=0,d.rotation.z=h&&ee!==null&&!H?-Math.sin(ee*Math.PI*2)*.014*O:0;return}if(y.emissiveIntensity=Q?.5+.9*Math.max(0,Math.sin(ae*7))**3:.15,ie.material.opacity=Q?.75+.25*Math.sin(ae*3.2):0,ie.rotation.z=ae*.25,ue&&(ue.material.opacity=Q?1:.72,ue.position.y=a*1.12+(Q&&!H?Math.sin(ae*3)*a*.03:0)),H){d.position.y=0,d.rotation.z=0;return}let W=ee===null?0:-Math.sin(ee*Math.PI*2)*.035*O;d.rotation.z=W,d.position.y=Q?Math.sin(ae*41)*a*.0015:0,ee!==null&&(d.position.y+=Math.abs(Math.sin(ee*Math.PI*3))*a*.008)},tip(){return _.getWorldPosition(xe),xe.clone()},teeth(){return(N??_).getWorldPosition(new P)},lip(){return(Y??_).getWorldPosition(new P)},exhaust(){return L?L.getWorldPosition(new P):s.position.clone()},bedLip(){return E?E.localToWorld(new P(-.02*r,.05*a,0)):this.tip()},dispose(){let ae=new Set;s.traverse(ee=>{ee.geometry&&ee.geometry!==ud&&ee.geometry!==Vs&&ee.geometry!==dd&&![...eh.values()].includes(ee.geometry)&&ae.add(ee.geometry),ee.isSprite&&(ee.material.map.dispose(),ee.material.dispose())});for(let ee of ae)ee.dispose();ie.material.map?.dispose(),ie.material.dispose(),y.dispose();for(let ee of u.tracks)ee.shoes.dispose()}}}var im="terra.viewer3d.v1",md=["Excavator","Truck","Skid steer"],GM=["Forward","Backward","Turn clockwise","Turn anticlockwise","Cabin clockwise","Cabin anticlockwise","Work","Wait"],WM=["action","target","padding","dumpability"],nm=["dumpability_static","interaction","traversability"],xs=n=>typeof n=="number"&&Number.isFinite(n),zn=n=>Number.isSafeInteger(n);function At(n,e){if(!n)throw new Error(e)}function sm(n){At(n&&typeof n=="object"&&n.schema===im,`Expected a ${im} replay.`),At(n.metadata&&typeof n.metadata.title=="string"&&typeof n.metadata.source=="string","Replay metadata must include title and source strings."),At(Array.isArray(n.frames)&&n.frames.length>0,"The replay contains no frames."),At(n.frames.length<=1e5,"This viewer supports at most 100,000 frames.");for(let[e,t]of n.frames.entries())gd(t,`Frame ${e}`);return n}function gd(n,e="Frame"){At(n&&typeof n=="object",`${e}: expected an object.`);let{grid:t,maps:i,agents:s}=n;At(t&&zn(t.rows)&&zn(t.cols)&&t.rows>0&&t.cols>0&&t.rows<=128&&t.cols<=128,`${e}: grid must be between 1 and 128 cells on each side.`),At(xs(t.tile_size_m)&&t.tile_size_m>0,`${e}: invalid tile size.`),At(i&&typeof i=="object",`${e}: missing maps.`);for(let o of[...WM,...nm]){let a=i[o];if(a==null&&nm.includes(o))continue;At(Array.isArray(a)&&a.length===t.rows,`${e}: ${o} has the wrong row count.`);let c=o==="action"||o==="target",l=o==="traversability"?[-1,0,1,!1,!0]:[0,1,!1,!0];for(let h of a)At(Array.isArray(h)&&h.length===t.cols&&h.every(u=>c?zn(u):l.includes(u)),`${e}: ${o} has invalid cells or columns.`)}At(zn(n.step)&&n.step>=0&&xs(n.reward),`${e}: invalid step or reward.`),At(n.action===null||zn(n.action)&&n.action>=0&&n.action<=7,`${e}: invalid action.`),At(typeof n.done=="boolean"&&typeof n.task_done=="boolean",`${e}: invalid episode outcome.`),At(!n.task_done||n.done,`${e}: task_done requires done.`),At(Array.isArray(s)&&s.length>0&&s.length<=4,`${e}: expected 1\u20134 active agents.`);let r=new Set;for(let o of s)At(o&&zn(o.id)&&o.id>=0&&o.id<=3&&!r.has(o.id),`${e}: agent IDs must be unique original slots from 0 to 3.`),r.add(o.id),At(zn(o.type)&&o.type>=0&&o.type<=2&&(o.action_type===0||o.action_type===1),`${e}: unknown machine type.`),At(Array.isArray(o.position)&&o.position.length===2&&o.position.every(xs)&&o.position[0]>=0&&o.position[0]<t.rows&&o.position[1]>=0&&o.position[1]<t.cols,`${e}: agent position is outside the map.`),At(xs(o.base_yaw)&&xs(o.cabin_yaw)&&zn(o.wheel_angle),`${e}: invalid machine angle.`),At(xs(o.width)&&xs(o.height)&&o.width>0&&o.height>0,`${e}: invalid machine footprint.`),At(zn(o.loaded)&&o.loaded>=0&&(o.shovel_lifted===0||o.shovel_lifted===1),`${e}: invalid machine load or shovel state.`),At(Array.isArray(o.reach)&&o.reach.length===2&&o.reach.every(xs)&&o.reach[0]>=0&&o.reach[1]>=o.reach[0],`${e}: invalid machine reach.`);return At(r.has(n.current_agent),`${e}: the active agent does not exist.`),At(n.actor_id===null||r.has(n.actor_id),`${e}: the preceding actor does not exist.`),n}function _d(n,e){if(n.action===null)return"Initial state";let t=(e||n).agents.find(i=>i.id===n.actor_id)||n.agents[0];return t.action_type===1&&(n.action===2||n.action===3)?n.action===2?"Steer left":"Steer right":n.action===6?t.type===2?"Shovel action":t.loaded>0?"Dump / transfer":t.type===1?"Dump":"Dig":GM[n.action]}function pa(n,e){if(!n||e.grid.rows!==n.grid.rows||e.grid.cols!==n.grid.cols)return{kind:"snapshot",changed:[],removed:0,placed:0,message:"Initial state"};let t=[],i=0,s=0;for(let u=0;u<e.grid.rows;u++)for(let d=0;d<e.grid.cols;d++){let f=e.maps.action[u][d]-n.maps.action[u][d];f&&(t.push({row:u,col:d,delta:f}),f<0?i-=f:s+=f)}let r=e.agents.find(u=>u.id===e.actor_id),o=n.agents.find(u=>u.id===e.actor_id),a=r&&o?r.loaded-o.loaded:0,c=e.agents.find(u=>u.id!==e.actor_id&&u.loaded>(n.agents.find(d=>d.id===u.id)?.loaded??u.loaded)),l="unchanged",h="No visible state change";return i>0&&a>0?(l="dig",h=`Picked up ${a} soil units \xB7 ${t.length} cells changed`):s>0&&a<0?(l="dump",h=`Placed ${-a} soil units \xB7 ${t.length} cells changed`):a<0&&c?(l="transfer",h=`Transferred soil to machine ${c.id+1}`):t.length?(l="terrain",h=`${t.length} terrain cells changed`):r&&o&&r.position.some((u,d)=>u!==o.position[d])?(l="move",h="Machine moved"):r&&o&&(r.base_yaw!==o.base_yaw||r.cabin_yaw!==o.cabin_yaw||r.wheel_angle!==o.wheel_angle||r.shovel_lifted!==o.shovel_lifted)&&(l="turn",h="Machine configuration changed"),{kind:l,changed:t,removed:i,placed:s,loadDelta:a,recipient:c,message:h}}function rm(n){let e=0,t=0,i=0;for(let s=0;s<n.grid.rows;s++)for(let r=0;r<n.grid.cols;r++){let o=n.maps.action[s][r];e+=Math.max(0,-o),t+=Math.max(0,o),i+=Math.max(0,-n.maps.target[s][r])}return{cut:e,fill:t,target:i,carried:n.agents.reduce((s,r)=>s+r.loaded,0)}}function xd(n,e,t){let i=Math.atan2(Math.sin(e-n),Math.cos(e-n));return n+i*t}function om(n){if(n===0)return"0.00";let e=Math.abs(n)<.01?n.toPrecision(3):n.toFixed(2);return`${n>0?"+":""}${e}`}var kn={name:"diorama",sand:14069366,dug:[12749400,11366984,9855037,8343860],loose:11037754,strata:[13013084,11433289,13673068,10250821],rock:9340541,grass:[9224530,7646533],grassEdge:6263612,sky:[9225962,13624815,16246732],clods:[11039039,12157001,9723951,12881752]},ma={diorama:kn,paper:{...kn,name:"paper",sand:14208441,dug:[12889485,11441525,9928288,8415310],loose:11242084,strata:[13153428,11705468,13878182,10718574],rock:10196622,clods:[11242084,10255448,12098164,9400400]},studio:{...kn,name:"studio",sand:10849641,dug:[8939851,7953730,6967609,6047281],loose:10124118,strata:[9992538,8677193,10585447,7822659],rock:7301731,clods:[7098165,8084800,6112302,9071694]}},sn={uTime:{value:0},uUnit:{value:.27},uTile:{value:.57},uFloor:{value:-1},uMotion:{value:1}},am=`
varying vec3 vTWorld;
varying vec3 vTNormal;
float tHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float tNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
  return mix(mix(tHash(i), tHash(i + vec2(1, 0)), f.x), mix(tHash(i + vec2(0, 1)), tHash(i + vec2(1, 1)), f.x), f.y);
}
float tFbm(vec2 p) { return tNoise(p) * .55 + tNoise(p * 2.13 + 7.1) * .3 + tNoise(p * 4.7 + 3.3) * .15; }
`,lm=`
vec4 tWorld = vec4(transformed, 1.);
vec3 tNormal = objectNormal;
#ifdef USE_INSTANCING
tWorld = instanceMatrix * tWorld; tNormal = mat3(instanceMatrix) * tNormal;
#endif
tWorld = modelMatrix * tWorld; vTWorld = tWorld.xyz; vTNormal = normalize(mat3(modelMatrix) * tNormal);
`,nh=n=>new Te(n),sh=n=>`vec3(${n.r.toFixed(4)}, ${n.g.toFixed(4)}, ${n.b.toFixed(4)})`;function XM(n,e){let[t,i,s,r]=e.strata.map(o=>sh(nh(o)));return`
  {
    float depth = -vTWorld.y / uUnit;
    float along = vTWorld.x * .83 + vTWorld.z * 1.17;
    float wobble = (tNoise(vec2(along * 1.4, depth * .35)) - .5) * .32;
    float band = depth + wobble, index = mod(floor(band), 4.), phase = fract(band);
    vec3 stratum = index < 1. ? ${t} : index < 2. ? ${i} : index < 3. ? ${s} : ${r};
    stratum *= .93 + tNoise(vec2(along * 5.1, vTWorld.y * 9.)) * .12;
    stratum *= mix(.8, 1., smoothstep(0., .1, phase));
    stratum *= 1. - clamp(depth * .025, 0., .28);
    if (vTWorld.y < uFloor) stratum = ${sh(nh(e.rock))} * (.86 + tNoise(vec2(along * 2.3, vTWorld.y * 2.7)) * .2);
    ${n?`if (vTWorld.y > -uTile * .2) stratum = ${sh(nh(e.grassEdge))} * (.92 + tNoise(vec2(along * 3., 1.)) * .14);`:""}
    diffuseColor.rgb = stratum;
  }`}function Hn(n,e={},t=kn){let i=new Qe({roughness:1,metalness:0,...e}),[s,r]=t.grass.map(a=>sh(nh(a))),o=n==="island"?`float g = smoothstep(.3, .72, tFbm(vTWorld.xz * .28)); diffuseColor.rgb = mix(${s}, ${r}, g) * (.94 + tNoise(vTWorld.xz * 3.1) * .1);`:n==="pile"?`diffuseColor.rgb *= (.84 + .2 * smoothstep(0., 5., vTWorld.y / uUnit)) * (.92 + tFbm(vTWorld.xz * 2.6) * .16)${t.name==="studio"?" * (.94 + tNoise(vTWorld.xz * 11.) * .12)":""};`:t.name==="studio"?"diffuseColor.rgb *= (.88 + tFbm(vTWorld.xz * 1.15) * .2) * (.95 + tNoise(vTWorld.xz * 9.) * .1);":"diffuseColor.rgb *= .9 + tFbm(vTWorld.xz * 1.15) * .2;";return i.onBeforeCompile=a=>{Object.assign(a.uniforms,sn),a.vertexShader=`varying vec3 vTWorld;
varying vec3 vTNormal;
${a.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
${lm}`),a.fragmentShader=`uniform float uUnit;
uniform float uTile;
uniform float uFloor;
${am}
${a.fragmentShader}`.replace("#include <color_fragment>",`#include <color_fragment>
      {
        vec3 tn = normalize(vTNormal);
        if (tn.y > .5) { ${o} }
        else if (tn.y > -.5) { ${n==="pile"?o:XM(n==="island",t)} }
      }`)},i.customProgramCacheKey=()=>`terra-earth-${n}-${t.name}`,i}var qM={hatch:0,dots:1,solid:2,cross:3,stripes:4};function rh({color:n,opacity:e,pattern:t="solid",...i}){let s=new Ln({color:n,transparent:!0,opacity:e,depthWrite:!1,...i}),r=qM[t];return s.onBeforeCompile=o=>{Object.assign(o.uniforms,sn),o.vertexShader=`varying vec3 vTWorld;
varying vec3 vTNormal;
${o.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
vec3 objectNormal = vec3(0., 1., 0.);
${lm}`),o.fragmentShader=`uniform float uTile;
uniform float uTime;
uniform float uMotion;
${am}
${o.fragmentShader}`.replace("#include <color_fragment>",`#include <color_fragment>
      {
        vec2 p = vTWorld.xz / uTile;
        float a = 1.;
        ${r===0?"a = mix(.42, 1., step(.5, fract((p.x + p.y) * .7 - uTime * .12 * uMotion)));":""}
        ${r===1?"vec2 q = fract(p * 1.5) - .5; a = mix(.5, 1., 1. - smoothstep(.2, .26, length(q)));":""}
        ${r===3?"a = mix(.35, 1., max(step(.72, fract((p.x + p.y) * .7)), step(.72, fract((p.x - p.y) * .7))));":""}
        ${r===4?"a = mix(.3, 1., step(.62, fract((p.x - p.y) * .55)));":""}
        diffuseColor.a *= a;
      }`)},s.customProgramCacheKey=()=>`terra-zone-${r}`,s}function cm(){let n=document.createElement("canvas");n.width=512,n.height=288;let e=n.getContext("2d"),t=e.createRadialGradient(256,170,20,256,150,330);t.addColorStop(0,"#f2f1ec"),t.addColorStop(.55,"#d9dcdc"),t.addColorStop(1,"#9fa9b0"),e.fillStyle=t,e.fillRect(0,0,512,288);let i=new pn(n);return i.colorSpace=Lt,i}function hm(){let n=document.createElement("canvas");n.width=4,n.height=256;let e=n.getContext("2d"),t=e.createLinearGradient(0,0,0,256),[i,s,r]=kn.sky.map(a=>`#${a.toString(16).padStart(6,"0")}`);t.addColorStop(0,i),t.addColorStop(.58,s),t.addColorStop(1,r),e.fillStyle=t,e.fillRect(0,0,4,256);let o=new pn(n);return o.colorSpace=Lt,o}var YM=[[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[0,2],[0,1]],$M=1.6,ZM=(n,e,t)=>typeof n=="function"?n(e,t):n;function oh(n,e,t,i,s){if(e<0||t<0||e>=n.grid.rows||t>=n.grid.cols||n.maps.padding[e][t])return 0;let r=n.maps.action[e][t],o=i?i.maps.action[e][t]:r;return Math.max(0,o+(r-o)*ZM(s,e,t))}function JM(n,e,t,i){if(typeof i!="function")return i;let s=e%2?[(e-1)/2]:[e/2-1,e/2],r=t%2?[(t-1)/2]:[t/2-1,t/2],o=0,a=0;for(let c of s)for(let l of r)c>=0&&l>=0&&c<n.grid.rows&&l<n.grid.cols&&(o+=i(c,l),a++);return a?o/a:1}function jM(n,e,t,i,s){let r=e%2?[(e-1)/2]:[e/2-1,e/2],o=t%2?[(t-1)/2]:[t/2-1,t/2],a=1/0;for(let c of r)for(let l of o)a=Math.min(a,oh(n,c,l,i,s));return a}function Gr(n,e=null,t=1,i=null){let s=n.grid.rows*2+1,r=n.grid.cols*2+1,o=new Float64Array(s*r),a=$M/2;for(let c=0;c<s;c++)for(let l=0;l<r;l++)o[c*r+l]=jM(n,c,l,e,t);for(let c=0;c<s;c++){let l=c*r;for(let h=1;h<r;h++)o[l+h]=Math.min(o[l+h],o[l+h-1]+a);for(let h=r-2;h>=0;h--)o[l+h]=Math.min(o[l+h],o[l+h+1]+a)}for(let c=1;c<s;c++)for(let l=0;l<r;l++){let h=c*r+l;o[h]=Math.min(o[h],o[h-r]+a)}for(let c=s-2;c>=0;c--)for(let l=0;l<r;l++){let h=c*r+l;o[h]=Math.min(o[h],o[h+r]+a)}if(e&&(typeof t=="function"||t>0&&t<1)){let c=i?.start??Gr(n,e,0),l=i?.end??Gr(n);for(let h=0;h<s;h++)for(let u=0;u<r;u++){let d=h*r+u,f=JM(n,h,u,t);o[d]=Math.min(o[d],c.heights[d]+(l.heights[d]-c.heights[d])*f)}}return{rows:s,cols:r,heights:o}}function KM(n,e=null){let t=[],i=[],s=new Map,r=n.grid.cols*2+1,o=(a,c)=>{let l=a*r+c;return s.has(l)||(s.set(l,t.length),t.push([a,c])),s.get(l)};for(let a=0;a<n.grid.rows;a++)for(let c=0;c<n.grid.cols;c++){if(oh(n,a,c,e,0)<=0&&oh(n,a,c,e,1)<=0)continue;let l=o(a*2+1,c*2+1),h=YM.map(([d,f])=>o(a*2+d,c*2+f)),u=h.flatMap((d,f)=>[l,d,h[(f+1)%h.length]]);i.push({row:a,col:c,center:l,ring:h,triangles:u})}return{nodes:t,cells:i}}var ah=class extends et{constructor(e,{previous:t=null,unitHeight:i=e.grid.tile_size_m*.48,layerSettings:s={},visibility:r={},palette:o=kn,roughness:a=0}={}){super(),this.frame=e,this.previous=t,this.unitHeight=i,this.endHeights=Gr(e),this.startHeights=t?Gr(e,t,0):this.endHeights,this.endpoints={start:this.startHeights,end:this.endHeights},this.progress=1,this.topology=KM(e,t),this.activeCells=[];let{rows:c,cols:l,tile_size_m:h}=e.grid,u=new Float32Array(this.topology.nodes.length*3),d=new Float32Array(u.length),f=new Te;for(let p=0;p<this.topology.nodes.length;p++){let[m,M]=this.topology.nodes[p],b=(T,S)=>S?0:((Math.sin(m*12.9898+M*78.233+T)*43758.5453%1+1)%1-.5)*2*a*h;u[p*3]=(M/2-l/2)*h+b(1.7,M===0||M===l*2),u[p*3+2]=(m/2-c/2)*h+b(5.3,m===0||m===c*2);let y=(m*37+M*61+m*M*7)%29/29;f.set(o.loose).multiplyScalar(.95+y*.1),f.toArray(d,p*3)}this.positions=new Ut(u,3).setUsage(xn);let g=new ut;g.setAttribute("position",this.positions),g.setAttribute("color",new Ut(d,3)),this.surface=new Ke(g,Hn("pile",{vertexColors:!0,flatShading:!0,polygonOffset:!0,polygonOffsetFactor:-1,polygonOffsetUnits:-2},o)),this.surface.name="connected-soil-piles",this.surface.castShadow=!0,this.surface.receiveShadow=!0,this.surface.userData.soilPiles=this,this.add(this.surface),this.layerSettings=s,this.overlays={};for(let[p,m]of Object.entries(s)){let M=new ut;M.setAttribute("position",this.positions);let b=rh({color:m.color,opacity:m.opacity*.7,pattern:m.pattern,polygonOffset:!0,polygonOffsetFactor:-2}),y=new Ke(M,b);y.position.y=h*(.008+Object.keys(this.overlays).length*.003),y.renderOrder=3+Object.keys(this.overlays).length,y.visible=!!r[p],y.frustumCulled=!1,y.userData.skipAO=!0,this.overlays[p]=y,this.add(y)}let x=new ut;x.setAttribute("position",this.positions),this.gridLines=new ns(x,new Nn({color:7426351,transparent:!0,opacity:.25,depthWrite:!1})),this.gridLines.position.y=h*.022,this.gridLines.renderOrder=12,this.gridLines.visible=!!r.grid,this.gridLines.frustumCulled=!1,this.add(this.gridLines),this.update(1)}nodeHeight(e,t){return(this.heights.heights[e*this.heights.cols+t]??0)*this.unitHeight}endpointHeight(e,t,i=!1){let s=i?this.startHeights:this.endHeights;return(s.heights[(e*2+1)*s.cols+t*2+1]??0)*this.unitHeight}update(e=1){this.progress=e,this.heights=typeof e=="function"?Gr(this.frame,this.previous,e,this.endpoints):e<=0?this.startHeights:e>=1?this.endHeights:Gr(this.frame,this.previous,e,this.endpoints);let t=this.positions.array;for(let r=0;r<this.topology.nodes.length;r++){let[o,a]=this.topology.nodes[r];t[r*3+1]=this.nodeHeight(o,a)}this.positions.needsUpdate=!0;let i=this.topology.cells.filter(r=>oh(this.frame,r.row,r.col,this.previous,e)>0),s=i.length!==this.activeCells.length||i.some((r,o)=>r!==this.activeCells[o]);if(this.activeCells=i,this.surface.visible=i.length>0,s||!this.surface.geometry.index){this.surface.geometry.setIndex(i.flatMap(r=>r.triangles));for(let[r,o]of Object.entries(this.overlays)){let a=this.layerSettings[r],c=this.frame.maps[a.map];o.geometry.setIndex(i.filter(l=>c!=null&&a.test(c[l.row][l.col])).flatMap(l=>l.triangles))}this.gridLines.geometry.setIndex(i.flatMap(r=>r.ring.flatMap((o,a)=>[o,r.ring[(a+1)%r.ring.length]])))}this.surface.geometry.computeVertexNormals(),this.surface.geometry.computeBoundingSphere()}setLayer(e,t){e==="grid"?this.gridLines.visible=t:this.overlays[e]&&(this.overlays[e].visible=t&&this.frame.maps[this.layerSettings[e].map]!=null)}cellForHit(e){let t=this.activeCells[Math.floor(e.faceIndex/8)];return t?{row:t.row,col:t.col}:null}dispose(){this.traverse(e=>{e.geometry?.dispose(),e.material&&e.material.dispose()}),this.clear()}};function vd(n,e=0){let t=(n^Math.imul(e+1,2654435761))>>>0;return t=Math.imul(t^t>>>16,2246822507),t=Math.imul(t^t>>>13,3266489909),(t^t>>>16)>>>0}var Gs=(n,e)=>vd(n,e)/4294967295,um=(n,e,t)=>Math.max(e,Math.min(t,n));function dm(n,e=128){if(!Array.isArray(n)||!n.length||!Array.isArray(n[0])||!n[0].length)throw new Error("Obstacle padding must be a nonempty rectangular array.");let t=n.length,i=n[0].length;if(t>e||i>e||n.some(s=>!Array.isArray(s)||s.length!==i||s.some(r=>![0,1,!1,!0].includes(r))))throw new Error(`Obstacle padding must contain aligned 0/1 cells, at most ${e} \xD7 ${e}.`);return{rows:t,cols:i}}function QM(n,e,t){let i=new Uint8Array(e*t),s=[];for(let r=0;r<e;r++)for(let o=0;o<t;o++){let a=r*t+o;if(!n[r][o]||i[a])continue;let c=[[r,o]];i[a]=1;let l=r,h=r,u=o,d=o;for(let g=0;g<c.length;g++){let[x,p]=c[g];l=Math.min(l,x),h=Math.max(h,x),u=Math.min(u,p),d=Math.max(d,p);for(let[m,M]of[[x-1,p],[x,p-1],[x,p+1],[x+1,p]]){if(m<0||M<0||m>=e||M>=t)continue;let b=m*t+M;n[m][M]&&!i[b]&&(i[b]=1,c.push([m,M]))}}let f=2166136261;for(let[g,x]of[...c].sort((p,m)=>p[0]-m[0]||p[1]-m[1]))f=Math.imul(f^g*131+x,16777619)>>>0;s.push({cells:c,minRow:l,maxRow:h,minCol:u,maxCol:d,seed:f})}return s}function eS(n,e,t){let i=new Uint16Array(t),s=null,r=0;for(let o=0;o<e;o++){let a=[];for(let c=0;c<t;c++)i[c]=n[o*t+c]?i[c]+1:0;for(let c=0;c<=t;c++){let l=c<t?i[c]:0,h=c;for(;a.length&&a[a.length-1].height>l;){let u=a.pop(),d=u.height*(c-u.start),f={row:o-u.height+1,col:u.start,rows:u.height,cols:c-u.start};(d>r||d===r&&(f.row<s.row||f.row===s.row&&f.col<s.col))&&(s=f,r=d),h=u.start}l&&(!a.length||a[a.length-1].height<l)&&a.push({start:h,height:l})}}return s}function tS(n,e,t,i){let s={...n};function r(o){if(o.row<0||o.col<0||o.row+o.rows>t||o.col+o.cols>i)return!1;for(let a=o.row;a<o.row+o.rows;a++)for(let c=o.col;c<o.col+o.cols;c++)if(!e[a*i+c])return!1;return!0}for(;;){let o=s,a=[{...o,row:o.row-1,rows:o.rows+1},{...o,col:o.col-1,cols:o.cols+1},{...o,rows:o.rows+1},{...o,cols:o.cols+1}].filter(r).sort((c,l)=>l.rows*l.cols-c.rows*c.cols||c.row-l.row||c.col-l.col);if(!a.length)return s;s=a[0]}}function iS(n,e=128){let{rows:t,cols:i}=dm(n,e),s=[],r=QM(n,t,i);for(let[o,a]of r.entries()){let c=a.maxRow-a.minRow+1,l=a.maxCol-a.minCol+1,h=new Uint8Array(c*l);for(let[g,x]of a.cells)h[(g-a.minRow)*l+x-a.minCol]=1;let u=h.slice(),d=a.cells.length,f=a.cells.length/(c*l);for(;d;){let g=tS(eS(u,c,l),h,c,l);for(let T=g.row;T<g.row+g.rows;T++)for(let S=g.col;S<g.col+g.cols;S++){let A=T*l+S;u[A]&&(u[A]=0,d--)}let x={row:g.row+a.minRow,col:g.col+a.minCol,rows:g.rows,cols:g.cols},p=vd(a.seed,x.row*131+x.col),m=Math.min(g.rows,g.cols),M=Math.max(g.rows,g.cols),b=(a.minRow+a.minCol+c+l)%3;if(f>=.9&&m>=3&&M>=6&&g.rows*g.cols>=24&&b!==0){let T=g.cols>=g.rows,S=Math.min(3,Math.floor(m/3),Math.max(1,Math.round(m/(M*.37))));for(let A=0;A<S;A++){let _=Math.floor(m*A/S),E=Math.floor(m*(A+1)/S),C={...x};T?(C.row+=_,C.rows=E-_):(C.col+=_,C.cols=E-_),s.push({...C,kind:"container",component:o,seed:vd(p,A)})}}else s.push({...x,kind:"boulder",component:o,seed:p})}}return s}function nS(n,e,t,i,s=!0){let o=[],a=[],c=[],l=new Te().setHex([10131340,10721928,9278606][i%3]),h=new Te(8824919),u=new P,d=new P,f=new P;for(let p=0;p<3;p++){let m=[];for(let M=0;M<9;M++){let b=(M+Gs(i,M)*.13)*Math.PI*2/9,y=p===2?.44+Gs(i,M+20)*.22:.85+Gs(i,M+p*9+40)*.14;m.push(new P(Math.cos(b)*n/2*y,p===0?0:t*(p===1?.38+Gs(i,M+70)*.1:.76+Gs(i,M+90)*.18),Math.sin(b)*e/2*y))}o.push(m)}function g(p,m,M){f.crossVectors(u.subVectors(m,p),d.subVectors(M,p)).normalize();let b=Math.abs(f.y),y=l.clone().multiplyScalar(.84+Gs(i,a.length)*.2+b*.12);s&&b>.72&&Gs(i,a.length+7)<.45&&y.lerp(h,.55);for(let T of[p,m,M])a.push(T.x,T.y,T.z),c.push(y.r,y.g,y.b)}for(let p=0;p<9;p++){let m=(p+1)%9;for(let M=0;M<2;M++)g(o[M][p],o[M+1][p],o[M+1][m]),g(o[M][p],o[M+1][m],o[M][m]);g(new P(0,0,0),o[0][p],o[0][m]),g(o[2][p],new P(0,t,0),o[2][m])}let x=new ut;return x.setAttribute("position",new it(a,3)),x.setAttribute("color",new it(c,3)),x.computeVertexNormals(),x.computeBoundingBox(),x}function Wr(n,e,t,i,s,r,o,a,c){let l=new Ke(e,t);return l.position.set(i,s,r),l.scale.set(o,a,c),l.castShadow=!0,l.receiveShadow=!0,n.add(l),l}function sS(n,e,t,i,s){let r=Math.max(e.rows,e.cols)*t,o=Math.min(e.rows,e.cols)*t,a=r*.94,c=Math.min(o*.88,a*.42),l=um(c*.94,t*.7,t*3.6),h=new et;h.rotation.y=e.rows>e.cols?Math.PI/2:0,n.add(h);let u=i+t*.12;s.box||(s.box=new Bt(1,1,1)),s.metal||(s.metal=new Qe({color:7831675,roughness:.64,metalness:.25})),s.foundation||(s.foundation=new Qe({color:9343364,roughness:1}));let d=new Qe({color:(s.paper?[9277839,8357252,10001045]:[10772291,5340795,6455185])[e.seed%3],roughness:.77,metalness:.15}),f=d.clone();f.color.multiplyScalar(1.12),Wr(h,s.box,s.foundation,0,u/2,0,a+t*.06,u,c+t*.09),Wr(h,s.box,d,0,u+l/2,0,a,l,c),Wr(h,s.box,f,0,u+l+t*.025,0,a,t*.05,c);let g=Math.max(4,Math.round(a/(t*.45))),x=new jt(s.box,f,g*2),p=new ft;x.castShadow=!0,x.receiveShadow=!0;for(let m=0;m<2;m++)for(let M=0;M<g;M++)p.position.set(a*(-.46+.92*M/(g-1)),u+l/2,(m?1:-1)*(c/2+t*.013)),p.scale.set(t*.065,l*.94,t*.033),p.updateMatrix(),x.setMatrixAt(m*g+M,p.matrix);h.add(x);for(let m of[-1,1]){Wr(h,s.box,f,a/2+t*.02,u+l*.5,m*c*.237,t*.04,l*.88,c*.45),Wr(h,s.box,s.metal,a/2+t*.047,u+l*.5,m*c*.17,t*.028,l*.79,t*.038);for(let M of[-1,1])Wr(h,s.box,s.metal,M*(a/2-t*.045),u+l/2,m*(c/2-t*.036),t*.09,l,t*.075)}}function fm(n,e={}){typeof e=="number"&&(e={tile:e});let t=e.tile??n.grid.tile_size_m,i=e.unitHeight??t*.48;if(!Number.isFinite(t)||t<=0||!Number.isFinite(i)||i<=0)throw new Error("Obstacle display scale must be finite and positive.");let s=n.metric?1048576:128,{rows:r,cols:o}=dm(n.maps.padding,s);if(r!==n.grid.rows||o!==n.grid.cols||!Array.isArray(n.maps.action)||n.maps.action.length!==r||n.maps.action.some(h=>!Array.isArray(h)||h.length!==o||h.some(u=>!Number.isFinite(u))))throw new Error("Obstacle terrain must match the frame grid and contain finite heights.");let a=iS(n.maps.padding,s),c=new et,l={paper:e.style!=="diorama"};c.name="Terra obstacle props",c.userData.footprints=a;for(let h of a){let u=new et;u.name=`${h.kind}-${h.row}-${h.col}`,u.userData.footprint={...h};let d=1/0,f=-1/0;for(let g=h.row;g<h.row+h.rows;g++)for(let x=h.col;x<h.col+h.cols;x++){let p=n.maps.action[g][x]*i;d=Math.min(d,p),f=Math.max(f,p)}if(u.position.set((h.col+h.cols/2-o/2)*t,d+t*.008,(h.row+h.rows/2-r/2)*t),h.kind==="container")sS(u,h,t,f-d,l);else{l.stone||(l.stone=new Qe({vertexColors:!0,roughness:1,flatShading:!0}));let g=um(Math.min(h.rows,h.cols)*t*.62,t*.52,t*3.4)+f-d,x=new Ke(nS(h.cols*t*.96,h.rows*t*.96,g,h.seed,!l.paper),l.stone);x.castShadow=!0,x.receiveShadow=!0,u.add(x)}c.add(u)}return c}function gm(n){let e=n>>>0||1;return()=>(e=Math.imul(e^e>>>15,739982445)+1831565813>>>0,e^=e>>>13,(e>>>0)/4294967295)}function _m(n,e,t,i,s=!1){let r=Math.min(i,e*.98,t*.98),o=[],a=[[e-r,t-r,0],[-e+r,t-r,Math.PI/2],[-e+r,-t+r,Math.PI],[e-r,-t+r,Math.PI*1.5]];for(let[c,l,h]of a)for(let u=0;u<=6;u++){let d=h+u/6*Math.PI/2;o.push(new $(c+Math.cos(d)*r,l+Math.sin(d)*r))}return s&&o.reverse(),n?(n.setFromPoints(o),n):o}function rS(n,e,t,i,s){let r=gm(s),o=_m(null,n,e,t),a=[o],c=[[.9,.34],[.66,.72],[.26,1]];for(let[m,M]of c)a.push(o.map(b=>{let y=.9+r()*.2;return new P(b.x*m*y,-i*M*(.85+r()*.3),b.y*m*y)}));a[0]=o.map(m=>new P(m.x,0,m.y));let l=[],h=[],u=new Te,d=[9206374,8219740,9864302,7299410],f=(m,M,b)=>{u.setHex(d[Math.floor(r()*d.length)]);for(let y of[m,M,b])l.push(y.x,y.y,y.z),h.push(u.r,u.g,u.b)};for(let m=0;m<a.length-1;m++)for(let M=0;M<o.length;M++){let b=(M+1)%o.length,y=a[m][M],T=a[m][b],S=a[m+1][M],A=a[m+1][b];f(y,S,T),f(T,S,A)}let g=new P(0,-i*1.25,0),x=a[a.length-1];for(let m=0;m<x.length;m++)f(x[m],g,x[(m+1)%x.length]);let p=new ut;return p.setAttribute("position",new it(l,3)),p.setAttribute("color",new it(h,3)),p.computeVertexNormals(),p}function oS(n,e=8){if(typeof document>"u")return null;let t=document.createElement("canvas");t.width=64,t.height=8;let i=t.getContext("2d");for(let r=0;r<e;r++){i.fillStyle=n[r%n.length],i.beginPath();let o=64/e;i.moveTo(r*o,0),i.lineTo(r*o+o,0),i.lineTo(r*o+o-4,8),i.lineTo(r*o-4,8),i.fill()}let s=new pn(t);return s.colorSpace=Lt,s.wrapS=zi,s.anisotropy=4,s}function pm(n){let e=new Qe({roughness:.9,flatShading:!0,...n});return e.onBeforeCompile=t=>{t.uniforms.uTime=sn.uTime,t.vertexShader=`uniform float uTime;
${t.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
      #ifdef USE_INSTANCING
      float swayPhase = instanceMatrix[3].x * .37 + instanceMatrix[3].z * .23;
      float swayHeight = max(0., position.y + .5);
      transformed.x += sin(uTime * 1.3 + swayPhase) * .05 * swayHeight;
      transformed.z += cos(uTime * 1.1 + swayPhase) * .035 * swayHeight;
      #endif`)},e.customProgramCacheKey=()=>"terra-sway",e}var Vn=class{constructor(e,t,i,s){this.mesh=new jt(t,i,s),this.mesh.count=0,this.mesh.castShadow=!0,this.mesh.receiveShadow=!0,e.add(this.mesh),this.dummy=new ft,this.color=new Te}add(e,t,i,s,r,o,a,c=0,l=0){if(this.mesh.count>=this.mesh.instanceMatrix.count)return;let h=this.dummy;h.position.set(e,t,i),h.rotation.set(l,c,l*.6),h.scale.set(s,r,o),h.updateMatrix(),this.mesh.setMatrixAt(this.mesh.count,h.matrix),this.mesh.setColorAt(this.mesh.count,this.color.setHex(a)),this.mesh.count++}finish(){this.mesh.instanceMatrix.needsUpdate=!0,this.mesh.instanceColor&&(this.mesh.instanceColor.needsUpdate=!0),this.mesh.computeBoundingSphere()}};function Kt(n,e,t,i,s,r,o,a,c){let l=new Ke(c,e);return l.position.set(t,i,s),l.scale.set(r,o,a),l.castShadow=!0,l.receiveShadow=!0,n.add(l),l}function aS(n,e,t,i,s){let r=new et;r.position.set(e,0,t),r.rotation.y=i,n.add(r);let{cube:o}=s,a=s.materials;Kt(r,a.concrete,0,.08,0,4.2,.16,2.5,o),Kt(r,a.office,0,1.4,0,4,2.5,2.3,o),Kt(r,a.trim,0,2.7,0,4.15,.14,2.45,o),Kt(r,a.trim,0,.22,0,4.1,.14,2.4,o);for(let l of[-1.2,.15])Kt(r,a.window,l,1.6,1.16,1,.75,.04,o),Kt(r,a.trim,l,1.18,1.19,1.1,.07,.08,o);Kt(r,a.door,1.35,1.15,1.16,.8,1.9,.05,o),Kt(r,a.concrete,1.35,.15,1.55,1.05,.3,.6,o),Kt(r,a.window,-2.005,1.6,0,.04,.7,1,o),Kt(r,a.metal,-1.2,2.95,-.4,.8,.36,.6,o),Kt(r,a.sign,.15,3.08,1.05,1.7,.46,.06,o);let c=new et;return c.position.set(1.3,0,-1.95),r.add(c),Kt(c,a.loo,0,1.12,0,1.05,2.24,1.05,o),Kt(c,a.looRoof,0,2.3,0,1.12,.12,1.12,o),Kt(c,a.trim,0,1.12,.53,.7,1.8,.03,o),r}function lS(n,e,t,i,s){let r=new et;r.position.set(e,0,t),r.rotation.y=i,n.add(r);let o=s.pipe;for(let[a,c]of[[-.55,.32],[0,.32],[.55,.32],[-.27,.8],[.27,.8],[0,1.27]]){let l=new Ke(o,s.materials.pipe);l.rotation.x=Math.PI/2,l.position.set(a,c,0),l.scale.set(.3,3.2,.3),l.castShadow=!0,l.receiveShadow=!0,r.add(l);let h=new Ke(s.ring,s.materials.pipeEnd);h.position.set(a,c,1.61),h.scale.setScalar(.3),r.add(h)}return Kt(r,s.materials.wood,0,.03,-1.1,1.8,.06,.2,s.cube),Kt(r,s.materials.wood,0,.03,1.1,1.8,.06,.2,s.cube),r}function cS(n,e,t,i,s,r){let o=new et;o.position.set(e,0,t),o.rotation.y=i,n.add(o),Kt(o,s.materials.wood,0,.07,0,1.2,.14,1,s.cube);let a=2+Math.floor(r()*3);for(let c=0;c<a;c++){let l=Kt(o,s.materials.bag,(c%2-.5)*.52,.28+Math.floor(c/2)*.26,0,.5,.24,.86,s.bagGeometry);l.rotation.y=(r()-.5)*.2}return o}function mm(n,e=ma.paper,t=!1){let{rows:i,cols:s,tile_size_m:r}=n.grid,o=Math.max(i,s)*r,a=new et;a.name="Terra plinth";let c=new Bt(s*r,1,i*r);c.translate(0,-.5,0);let l=new Ke(c,Hn("soil",{polygonOffset:!0,polygonOffsetFactor:1,polygonOffsetUnits:2},e));l.name="plinth",l.receiveShadow=!0,l.castShadow=t,a.add(l);let h=null;return t&&(h=new Ke(new ki(o*12,o*12),new Uo({color:2893344,opacity:.28})),h.rotation.x=-Math.PI/2,h.receiveShadow=!0,h.name="studio-floor",a.add(h)),a.userData.extent={hx:s*r/2,hz:i*r/2},a.setFloor=u=>{let d=Math.min(-Math.max(o*(t?.085:.06),1.8),u-r*.8);l.position.y=u,l.scale.y=u-d,sn.uFloor.value=u,h&&(h.position.y=d-.002)},a.update=()=>{},a.dispose=()=>{c.dispose(),l.material.dispose(),h&&(h.geometry.dispose(),h.material.dispose()),a.removeFromParent()},a}function xm(n,{style:e="diorama"}={}){if(e==="paper")return mm(n);if(e==="studio")return mm(n,ma.studio,!0);let{rows:t,cols:i,tile_size_m:s}=n.grid,r=i*s/2,o=t*s/2,a=Math.max(t,i)*s,c=Vt.clamp(a*.2,6,22),l=new et;l.name="Terra surroundings";let h=gm(t*7919+i*104729+Math.round(s*1e3)),u=r+c,d=o+c,f=c*1.1,g={cube:new Bt(1,1,1),pipe:new Ki(1,1,1,10,1,!0),ring:new Do(.72,1,10),bagGeometry:new Bt(1,1,1),materials:{concrete:new Qe({color:12170925,roughness:1}),office:new Qe({color:15986918,roughness:.8}),trim:new Qe({color:4157338,roughness:.7}),window:new Qe({color:10475238,roughness:.15,metalness:.1,emissive:1915460,emissiveIntensity:.3}),door:new Qe({color:3102072,roughness:.7}),metal:new Qe({color:13225680,roughness:.5}),sign:new Qe({color:15905329,roughness:.6}),loo:new Qe({color:3842264,roughness:.6}),looRoof:new Qe({color:15397621,roughness:.6}),pipe:new Qe({color:15040058,roughness:.7,side:Mi}),pipeEnd:new Qe({color:12083499,roughness:.8,side:Mi}),wood:new Qe({color:12159573,roughness:1}),bag:new Qe({color:15327433,roughness:1})}},x=_m(new gn,u,d,f),p=new mn;p.moveTo(-r,-o),p.lineTo(-r,o),p.lineTo(r,o),p.lineTo(r,-o),p.closePath(),x.holes.push(p);let m=new Fn(x,{depth:1,bevelEnabled:!1,curveSegments:6});m.rotateX(Math.PI/2);let M=new Ke(m,Hn("island"));M.receiveShadow=!0,M.castShadow=!1,M.name="island-turf",l.add(M);let b=new Ke(rS(u,d,f,Math.max(a*.16,4),t*31+i),new Qe({vertexColors:!0,flatShading:!0,roughness:1,side:Mi}));b.name="island-underside",l.add(b);let y=Math.min(4.2,o*.5),T=new Ke(new ki(1,1),Hn("soil",{color:13482134}));T.rotation.x=-Math.PI/2,T.scale.set(c,y,1),T.position.set(r+c/2+.01,.012,0),T.name="site-road",T.receiveShadow=!0,l.add(T);let S=y/2+.3,A=new Vn(l,g.cube,new Qe({color:15657696,roughness:.7}),400),_=new jt(g.cube,new Qe({map:oS(["#e8573a","#f7f2e8"]),color:typeof document>"u"?15226682:16777215,roughness:.7}),800);_.count=0,_.castShadow=!0,l.add(_);let E=new ft,C=.18,I=[[[-r-C,-o-C],[r+C,-o-C]],[[-r-C,o+C],[r+C,o+C]],[[-r-C,-o-C],[-r-C,o+C]],[[r+C,-o-C],[r+C,-S]],[[r+C,S],[r+C,o+C]]];for(let[[W,G],[se,ce]]of I){let fe=Math.hypot(se-W,ce-G),me=Math.max(1,Math.round(fe/2.4)),D=Math.atan2(-(ce-G),se-W);for(let Me=0;Me<=me;Me++){let Ve=Me/me;A.add(W+(se-W)*Ve,.45,G+(ce-G)*Ve,.1,.9,.1,Me%2?15657696:15226682)}for(let Me=0;Me<me;Me++)for(let Ve of[.42,.78]){let R=(Me+.5)/me;E.position.set(W+(se-W)*R,Ve,G+(ce-G)*R),E.rotation.set(0,D,0),E.scale.set(fe/me,.09,.02),E.updateMatrix(),_.setMatrixAt(_.count++,E.matrix)}}A.finish(),_.instanceMatrix.needsUpdate=!0,_.computeBoundingSphere();let L=new Er(.2,.6,8);L.translate(0,.3,0);let V=new Vn(l,L,new Qe({color:16777215,roughness:.6,flatShading:!0}),40),q=new Vn(l,new Ki(.115,.145,.1,8),new Qe({color:16777215,roughness:.4}),40);for(let W=0;W<Math.floor(c/2.2);W++)for(let G of[-1,1]){let se=r+1+W*2.2,ce=G*(y/2+.35);V.add(se,0,ce,1,1,1,15953706),q.add(se,.33,ce,1,1,1,16250090)}V.finish(),q.finish();let N=[],Y=(W,G,se)=>{if(Math.abs(W)<r+1.2+se&&Math.abs(G)<o+1.2+se)return!1;let ce=Math.max(0,Math.abs(W)-(u-f)),fe=Math.max(0,Math.abs(G)-(d-f));return Math.hypot(ce,fe)>f-se-.5||W>r&&Math.abs(G)<y/2+se+.6?!1:N.every(([me,D,Me])=>Math.hypot(W-me,G-D)>se+Me)},X=-(y/2+3.2),ne=r+Math.min(c*.55,6);c>=6&&Y(ne,X,2.3)&&Y(ne+1.3,X-1.95,.8)&&(aS(l,ne,X,0,g),N.push([ne,X,2.8],[ne+1.3,X-1.95,.9]));let ie=r+Math.min(c*.6,6.5),ge=y/2+3;c>=6&&Y(ie,ge,1.7)&&(lS(l,ie,ge,Math.PI/2+.1,g),N.push([ie,ge,2]));for(let W=0;W<3;W++){let G=-r-c*(.35+h()*.3),se=(h()-.5)*o*1.4;Y(G,se,.9)&&(cS(l,G,se,h()*Math.PI,g,h),N.push([G,se,.9]))}let ue=new Ki(.5,.7,1,6);ue.translate(0,.5,0);let xe=new Er(1,1,7);xe.translate(0,.5,0);let Ne=new Qi(1,0),st=new Vn(l,ue,new Qe({color:16777215,roughness:1,flatShading:!0}),400),Xe=new Vn(l,xe,pm({color:16777215}),900),j=new Vn(l,Ne,pm({color:16777215}),700),he=new Vn(l,new bo(1,0),new Qe({color:16777215,roughness:1,flatShading:!0}),200),le=[5214042,6069343,4620114],Ae=[8238678,9224541,6989903,10930522],Fe=[15905628,15306091],ke=4*(u*d-r*o),ae=Math.min(2600,Math.round(ke/2.2));for(let W=0;W<ae;W++){let G=h(),se=Math.floor(h()*4),ce=Math.pow(h(),.8),fe,me;se<2?(fe=(G*2-1)*u,me=(se?1:-1)*(o+1.8+ce*(c-1.8))):(me=(G*2-1)*d,fe=(se===3?1:-1)*(r+1.8+ce*(c-1.8)));let D=h(),Me=.62+h()*.45,Ve=D<.45?1.1*Me:D<.8?1.3*Me:.7*Me,R=Math.min(Math.abs(Math.abs(fe)-r),Math.abs(Math.abs(me)-o));if(h()>.25+Math.min(1,R/c)*.9||!Y(fe,me,Ve))continue;N.push([fe,me,Ve]);let v=h()*Math.PI*2,U=(h()-.5)*.08;if(D<.45){let B=(3.4+h()*1.8)*Me;st.add(fe,0,me,.18*Me,B*.3,.18*Me,9067835,v);for(let k=0;k<3;k++)Xe.add(fe,B*(.22+k*.22),me,(1.25-k*.3)*Me,B*.42,(1.25-k*.3)*Me,le[(W+k)%3],v+k,U)}else if(D<.8){let B=(2.2+h()*1.4)*Me;st.add(fe,0,me,.16*Me,B*.55,.16*Me,9725247,v),j.add(fe,B*.75,me,1.25*Me,1.05*Me,1.2*Me,h()<.08?Fe[W%2]:Ae[W%4],v,U),h()<.6&&j.add(fe+.45*Me,B*.98,me-.2*Me,.8*Me,.7*Me,.8*Me,Ae[(W+1)%4],v+1)}else D<.93?j.add(fe,.35*Me,me,.7*Me,.5*Me,.7*Me,Ae[(W+2)%4],v):he.add(fe,.12*Me,me,.55*Me,.38*Me,.5*Me,[10130828,9078399,10985879][W%3],v,U*3)}for(let W of[st,Xe,j,he])W.finish();let ee=new et,O=new Qe({color:16777215,roughness:1,flatShading:!0,emissive:16777215,emissiveIntensity:.25}),H=new Qi(1,1),Q=Math.hypot(u,d);for(let W=0;W<6;W++){let G=new et,se=W/6*Math.PI*2+h()*.6,ce=Q*(1.25+h()*.45),fe=a*(.035+h()*.025);G.position.set(Math.cos(se)*ce,a*(.12+h()*.22),Math.sin(se)*ce);for(let me=0;me<5;me++){let D=new Ke(H,O);D.position.set((me-2)*fe*.9,(1-Math.abs(me-2)*.45)*fe*.35,(h()-.5)*fe*.7),D.scale.setScalar(fe*(1.1-Math.abs(me-2)*.22)),G.add(D)}G.userData={angle:se,distance:ce,height:G.position.y,speed:.006+h()*.006},ee.add(G)}return l.add(ee),l.userData.extent={hx:u,hz:d},l.setFloor=W=>{let G=Math.min(-Math.max(a*.07,2.2),W-s*.8);M.scale.y=-G,b.position.y=G,sn.uFloor.value=W},l.update=W=>{for(let G of ee.children){let{angle:se,distance:ce,height:fe,speed:me}=G.userData,D=se+W*me;G.position.set(Math.cos(D)*ce,fe+Math.sin(W*.3+se*3)*a*.006,Math.sin(D)*ce)}},l.dispose=()=>{let W=new Set,G=new Set;l.traverse(se=>{se.geometry&&W.add(se.geometry),se.material&&G.add(se.material)});for(let se of W)se.dispose();for(let se of G)se.map?.dispose(),se.dispose();l.removeFromParent()},l}function vm(n,e,t,i,s){let{rows:r,cols:o,tile_size_m:a}=n.grid,c=[],l=[],h=new Te(s.sand),u=new Te(s.loose),d=s.dug.map(S=>new Te(S)),f=new Te(16777215),g=(S,A)=>n.metric.loose[S][A]>1e-6?1:e[S][A]<0?2+Math.max(0,Math.min(d.length-1,Math.floor(-e[S][A])-1)):0,x=[h,u,...d],p=(S,A,_,E,C,I=!1)=>{for(let L of I?[S,E,A,A,E,_]:[S,A,E,A,_,E])c.push(...L),l.push(C.r,C.g,C.b)},m=S=>{let A=(S.col-o/2)*a,_=(S.end-o/2)*a,E=(r/2-S.bottom)*a,C=(r/2-S.row)*a,I=S.height*t;p([A,I,E],[A,I,C],[_,I,C],[_,I,E],x[S.kind])},M=new Map;for(let S=0;S<r;S++){let A=new Map;for(let _=0;_<o;){let E=e[S][_],C=g(S,_),I=_++;for(;_<o&&e[S][_]===E&&g(S,_)===C;)_++;let L=`${I}:${_}:${E}:${C}`,V=M.get(L)??{row:S,col:I,end:_,height:E,kind:C};V.bottom=S+1,A.set(L,V),M.delete(L)}for(let _ of M.values())m(_);M=A}for(let S of M.values())m(S);let b=c.length/3,y=i/t;for(let S=0;S<=r;S++)for(let A=0;A<o;){let _=S>0?e[S-1][A]:y,E=S<r?e[S][A]:y,C=A++;if(_===E)continue;for(;A<o&&(S>0?e[S-1][A]:y)===_&&(S<r?e[S][A]:y)===E;)A++;let I=(C-o/2)*a,L=(A-o/2)*a,V=(r/2-S)*a,q=Math.min(_,E)*t,N=Math.max(_,E)*t;p([I,q,V],[L,q,V],[L,N,V],[I,N,V],f,_>E)}for(let S=0;S<=o;S++)for(let A=0;A<r;){let _=S>0?e[A][S-1]:y,E=S<o?e[A][S]:y,C=A++;if(_===E)continue;for(;A<r&&(S>0?e[A][S-1]:y)===_&&(S<o?e[A][S]:y)===E;)A++;let I=(r/2-A)*a,L=(r/2-C)*a,V=(S-o/2)*a,q=Math.min(_,E)*t,N=Math.max(_,E)*t;p([V,q,I],[V,q,L],[V,N,L],[V,N,I],f,_>E)}let T=new ut;return T.setAttribute("position",new it(c,3)),T.setAttribute("color",new it(l,3)),T.addGroup(0,b,0),T.addGroup(b,c.length/3-b,1),T.computeVertexNormals(),T.computeBoundingSphere(),T}var ym=9.81,hS=[11039039,12157001,9723951,12881752],yd=900,lh=240,Mm=new P(0,1,0),uS={vertexShader:`
    attribute float size; attribute float alpha; varying float vAlpha;
    uniform float scale;
    void main() {
      vec4 view = modelViewMatrix * vec4(position, 1.);
      gl_PointSize = size * scale / -view.z; vAlpha = alpha;
      gl_Position = projectionMatrix * view;
    }`,fragmentShader:`
    uniform vec3 color; varying float vAlpha;
    void main() {
      float r = length(gl_PointCoord - .5) * 2.;
      float a = vAlpha * smoothstep(1., .15, r);
      if (a < .003) discard;
      gl_FragColor = vec4(color, a);
      #include <colorspace_fragment>
    }`},ch=class extends et{constructor({groundHeight:e=()=>0}={}){super(),this.name="Terra effects",this.groundHeight=e,this.dummy=new ft,this.color=new Te;let t=new Qi(1,0);this.clodMesh=new jt(t,new Qe({roughness:1,flatShading:!0}),yd),this.clodMesh.castShadow=!0,this.clodMesh.frustumCulled=!1,this.clodMesh.count=0,this.add(this.clodMesh),this.puffMesh=new jt(new Qi(1,1),new Qe({roughness:1,flatShading:!0}),260),this.puffMesh.frustumCulled=!1,this.puffMesh.count=0,this.add(this.puffMesh);for(let s of[this.clodMesh,this.puffMesh])s.setColorAt(0,this.color.setHex(16777215)),s.userData.skipAO=!0;let i=new ut;i.setAttribute("position",new Ut(new Float32Array(lh*3),3).setUsage(xn)),i.setAttribute("size",new Ut(new Float32Array(lh),1).setUsage(xn)),i.setAttribute("alpha",new Ut(new Float32Array(lh),1).setUsage(xn)),this.dustMaterial=new bt({...uS,uniforms:{color:{value:new Te(13219740)},scale:{value:600}},transparent:!0,depthWrite:!1}),this.dustPoints=new vo(i,this.dustMaterial),this.dustPoints.frustumCulled=!1,this.dustPoints.renderOrder=40,this.dustPoints.userData.skipAO=!0,this.add(this.dustPoints),this.clods=[],this.puffs=[],this.dusts=[],this.puffsEnabled=!0,this.dustEnabled=!1,this.palette=hS}setViewport(e,t){this.dustMaterial.uniforms.scale.value=e/(2*Math.tan(Vt.degToRad(t)/2))}dust(e,{count:t=6,size:i=.6,spread:s=.3,rise:r=.35,life:o=1.6,opacity:a=.22,drift:c=null}={}){if(this.dustEnabled)for(let l=0;l<t&&this.dusts.length<lh;l++){let h=new P((Math.random()-.5)*s*2,Math.random()*s*.4,(Math.random()-.5)*s*2),u=h.clone().multiplyScalar(.8).add(Mm.clone().multiplyScalar(r*(.5+Math.random()*.7)));c&&u.add(c),this.dusts.push({position:e.clone().add(h),velocity:u,size:i*(.6+Math.random()*.7),age:-Math.random()*.15,life:o*(.75+Math.random()*.5),opacity:a})}}throwClods(e,t,{count:i=10,flight:s=.45,spread:r=.3,size:o=.09,settle:a=!0,jitter:c=.08}={}){for(let l=0;l<i&&this.clods.length<yd;l++){let h=e.clone().add(new P((Math.random()-.5)*c*2,(Math.random()-.5)*c,(Math.random()-.5)*c*2)),u=t.clone().add(new P((Math.random()-.5)*r*2,0,(Math.random()-.5)*r*2)),d=s*(.8+Math.random()*.4),f=Math.random()*s*.6,g=u.sub(h).multiplyScalar(1/d);g.y+=.5*ym*d,this.clods.push({position:h,velocity:g,spin:new P(Math.random()*9,Math.random()*9,Math.random()*9),rotation:new Ii(Math.random()*6,Math.random()*6,0),size:o*(.6+Math.random()*.8),age:-f,life:d+(a?1.4:0),arrive:d,settle:a,bounced:!1,color:this.palette[Math.floor(Math.random()*this.palette.length)]})}}burst(e,{count:t=12,speed:i=2.2,size:s=.07}={}){for(let r=0;r<t&&this.clods.length<yd;r++){let o=Math.random()*Math.PI*2,a=.55+Math.random()*.5,c=new P(Math.cos(o)*(1-a),a*1.4,Math.sin(o)*(1-a)).multiplyScalar(i*(.6+Math.random()*.6));this.clods.push({position:e.clone(),velocity:c,spin:new P(Math.random()*12,Math.random()*12,0),rotation:new Ii,size:s*(.6+Math.random()*.8),age:-Math.random()*.08,life:2.2,arrive:1/0,settle:!0,bounced:!1,color:this.palette[Math.floor(Math.random()*this.palette.length)]})}}puff(e,{count:t=6,size:i=.25,spread:s=.35,rise:r=.6,life:o=.9,color:a=15326402,drift:c=null}={}){if(this.puffsEnabled)for(let l=0;l<t&&this.puffs.length<260;l++){let h=new P((Math.random()-.5)*s*2,Math.random()*s*.5,(Math.random()-.5)*s*2),u=h.clone().multiplyScalar(1.4).add(Mm.clone().multiplyScalar(r*(.6+Math.random()*.6)));c&&u.add(c),this.puffs.push({position:e.clone().add(h),velocity:u,size:i*(.6+Math.random()*.7),age:-Math.random()*.12,life:o*(.75+Math.random()*.5),color:a})}}update(e){e=Math.min(Math.max(e,0),.25);for(let t=e;t>1e-6;t-=.025)this.step(Math.min(.025,t));this.draw()}step(e){this.clods=this.clods.filter(t=>{if(t.age+=e,t.age<0)return!0;if(t.age>t.life)return!1;if((t.age<t.arrive||t.settle)&&!t.resting){t.velocity.y-=ym*e,t.position.addScaledVector(t.velocity,e),t.rotation.x+=t.spin.x*e,t.rotation.y+=t.spin.y*e,t.rotation.z+=t.spin.z*e;let i=this.groundHeight(t.position.x,t.position.z)+t.size*.5;t.position.y<i&&t.velocity.y<0&&(t.position.y=i,!t.bounced&&t.velocity.y<-1.2?(t.velocity.y*=-.28,t.velocity.x*=.45,t.velocity.z*=.45,t.bounced=!0):(t.resting=!0,t.restAge=t.age))}return!(t.age>=t.arrive&&!t.settle)}),this.puffs=this.puffs.filter(t=>(t.age+=e,t.age>=0&&(t.position.addScaledVector(t.velocity,e),t.velocity.multiplyScalar(Math.exp(-e*2.2))),t.age<t.life)),this.dusts=this.dusts.filter(t=>(t.age+=e,t.age>=0&&(t.position.addScaledVector(t.velocity,e),t.velocity.multiplyScalar(Math.exp(-e*1.6))),t.age<t.life))}draw(){let e=this.dummy,t=this.clods.length;for(let o=0;o<t;o++){let a=this.clods[o],c=a.age>=0,l=a.resting?Math.max(0,1-(a.age-a.restAge)/.9):1;e.position.copy(a.position),e.rotation.copy(a.rotation),e.scale.setScalar(c?a.size*l:0),e.updateMatrix(),this.clodMesh.setMatrixAt(o,e.matrix),this.clodMesh.setColorAt(o,this.color.setHex(a.color))}this.clodMesh.count=t,this.clodMesh.instanceMatrix.needsUpdate=!0,this.clodMesh.instanceColor&&(this.clodMesh.instanceColor.needsUpdate=!0);for(let o=0;o<this.puffs.length;o++){let a=this.puffs[o],c=Math.max(0,a.age)/a.life,l=a.age<0?0:Math.sin(Math.min(1,c*1.25)*Math.PI)*(1+c*.6);e.position.copy(a.position),e.rotation.set(a.age*.7,a.age,0),e.scale.setScalar(a.size*l),e.updateMatrix(),this.puffMesh.setMatrixAt(o,e.matrix),this.puffMesh.setColorAt(o,this.color.setHex(a.color))}this.puffMesh.count=this.puffs.length,this.puffMesh.instanceMatrix.needsUpdate=!0,this.puffMesh.instanceColor&&(this.puffMesh.instanceColor.needsUpdate=!0);let i=this.dustPoints.geometry.attributes.position,s=this.dustPoints.geometry.attributes.size,r=this.dustPoints.geometry.attributes.alpha;for(let o=0;o<this.dusts.length;o++){let a=this.dusts[o],c=Math.max(0,a.age)/a.life;i.setXYZ(o,a.position.x,a.position.y,a.position.z),s.setX(o,a.age<0?0:a.size*(.55+c*1.1)),r.setX(o,a.age<0?0:a.opacity*Math.min(1,c*6)*(1-c)**1.5)}this.dustPoints.geometry.setDrawRange(0,this.dusts.length);for(let o of[i,s,r])o.needsUpdate=!0}clear(){this.clods=[],this.puffs=[],this.dusts=[],this.clodMesh.count=0,this.puffMesh.count=0,this.dustPoints.geometry.setDrawRange(0,0)}dispose(){this.clear();for(let e of[this.clodMesh,this.puffMesh])e.geometry.dispose(),e.material.dispose(),e.dispose();this.dustPoints.geometry.dispose(),this.dustMaterial.dispose(),this.removeFromParent()}};var Xr={name:"CopyShader",uniforms:{tDiffuse:{value:null},opacity:{value:1}},vertexShader:`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`

		uniform float opacity;

		uniform sampler2D tDiffuse;

		varying vec2 vUv;

		void main() {

			vec4 texel = texture2D( tDiffuse, vUv );
			gl_FragColor = opacity * texel;


		}`};var Ni=class{constructor(){this.isPass=!0,this.enabled=!0,this.needsSwap=!0,this.clear=!1,this.renderToScreen=!1}setSize(){}render(){console.error("THREE.Pass: .render() must be implemented in derived pass.")}dispose(){}},dS=new as(-1,1,1,-1,0,1),Md=class extends ut{constructor(){super(),this.setAttribute("position",new it([-1,3,0,-1,-1,0,3,-1,0],3)),this.setAttribute("uv",new it([0,2,0,0,2,0],2))}},fS=new Md,vs=class{constructor(e){this._mesh=new Ke(fS,e)}dispose(){this._mesh.geometry.dispose()}render(e){e.render(this._mesh,dS)}get material(){return this._mesh.material}set material(e){this._mesh.material=e}};var ys=class extends Ni{constructor(e,t="tDiffuse"){super(),this.textureID=t,this.uniforms=null,this.material=null,e instanceof bt?(this.uniforms=e.uniforms,this.material=e):e&&(this.uniforms=mi.clone(e.uniforms),this.material=new bt({name:e.name!==void 0?e.name:"unspecified",defines:Object.assign({},e.defines),uniforms:this.uniforms,vertexShader:e.vertexShader,fragmentShader:e.fragmentShader})),this._fsQuad=new vs(this.material)}render(e,t,i){this.uniforms[this.textureID]&&(this.uniforms[this.textureID].value=i.texture),this._fsQuad.material=this.material,this.renderToScreen?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(t),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this._fsQuad.render(e))}dispose(){this.material.dispose(),this._fsQuad.dispose()}};var ga=class extends Ni{constructor(e,t){super(),this.scene=e,this.camera=t,this.clear=!0,this.needsSwap=!1,this.inverse=!1}render(e,t,i){let s=e.getContext(),r=e.state;r.buffers.color.setMask(!1),r.buffers.depth.setMask(!1),r.buffers.color.setLocked(!0),r.buffers.depth.setLocked(!0);let o,a;this.inverse?(o=0,a=1):(o=1,a=0),r.buffers.stencil.setTest(!0),r.buffers.stencil.setOp(s.REPLACE,s.REPLACE,s.REPLACE),r.buffers.stencil.setFunc(s.ALWAYS,o,4294967295),r.buffers.stencil.setClear(a),r.buffers.stencil.setLocked(!0),e.setRenderTarget(i),this.clear&&e.clear(),e.render(this.scene,this.camera),e.setRenderTarget(t),this.clear&&e.clear(),e.render(this.scene,this.camera),r.buffers.color.setLocked(!1),r.buffers.depth.setLocked(!1),r.buffers.color.setMask(!0),r.buffers.depth.setMask(!0),r.buffers.stencil.setLocked(!1),r.buffers.stencil.setFunc(s.EQUAL,1,4294967295),r.buffers.stencil.setOp(s.KEEP,s.KEEP,s.KEEP),r.buffers.stencil.setLocked(!0)}},hh=class extends Ni{constructor(){super(),this.needsSwap=!1}render(e){e.state.buffers.stencil.setLocked(!1),e.state.buffers.stencil.setTest(!1)}};var uh=class{constructor(e,t){if(this.renderer=e,this._pixelRatio=e.getPixelRatio(),t===void 0){let i=e.getSize(new $);this._width=i.width,this._height=i.height,t=new Ht(this._width*this._pixelRatio,this._height*this._pixelRatio,{type:ii}),t.texture.name="EffectComposer.rt1"}else this._width=t.width,this._height=t.height;this.renderTarget1=t,this.renderTarget2=t.clone(),this.renderTarget2.texture.name="EffectComposer.rt2",this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2,this.renderToScreen=!0,this.passes=[],this.copyPass=new ys(Xr),this.copyPass.material.blending=zt,this.timer=new Go}swapBuffers(){let e=this.readBuffer;this.readBuffer=this.writeBuffer,this.writeBuffer=e}addPass(e){this.passes.push(e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}insertPass(e,t){this.passes.splice(t,0,e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}removePass(e){let t=this.passes.indexOf(e);t!==-1&&this.passes.splice(t,1)}isLastEnabledPass(e){for(let t=e+1;t<this.passes.length;t++)if(this.passes[t].enabled)return!1;return!0}render(e){this.timer.update(),e===void 0&&(e=this.timer.getDelta());let t=this.renderer.getRenderTarget(),i=!1;for(let s=0,r=this.passes.length;s<r;s++){let o=this.passes[s];if(o.enabled!==!1){if(o.renderToScreen=this.renderToScreen&&this.isLastEnabledPass(s),o.render(this.renderer,this.writeBuffer,this.readBuffer,e,i),o.needsSwap){if(i){let a=this.renderer.getContext(),c=this.renderer.state.buffers.stencil;c.setFunc(a.NOTEQUAL,1,4294967295),this.copyPass.render(this.renderer,this.writeBuffer,this.readBuffer,e),c.setFunc(a.EQUAL,1,4294967295)}this.swapBuffers()}ga!==void 0&&(o instanceof ga?i=!0:o instanceof hh&&(i=!1))}}this.renderer.setRenderTarget(t)}reset(e){if(e===void 0){let t=this.renderer.getSize(new $);this._pixelRatio=this.renderer.getPixelRatio(),this._width=t.width,this._height=t.height,e=this.renderTarget1.clone(),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.renderTarget1=e,this.renderTarget2=e.clone(),this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2}setSize(e,t){this._width=e,this._height=t;let i=this._width*this._pixelRatio,s=this._height*this._pixelRatio;this.renderTarget1.setSize(i,s),this.renderTarget2.setSize(i,s);for(let r=0;r<this.passes.length;r++)this.passes[r].setSize(i,s)}setPixelRatio(e){this._pixelRatio=e,this.setSize(this._width,this._height)}dispose(){this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.copyPass.dispose()}};var dh=class extends Ni{constructor(e,t,i=null,s=null,r=null){super(),this.scene=e,this.camera=t,this.overrideMaterial=i,this.clearColor=s,this.clearAlpha=r,this.clear=!0,this.clearDepth=!1,this.needsSwap=!1,this.isRenderPass=!0,this._oldClearColor=new Te}render(e,t,i){let s=e.autoClear;e.autoClear=!1;let r,o;this.overrideMaterial!==null&&(o=this.scene.overrideMaterial,this.scene.overrideMaterial=this.overrideMaterial),this.clearColor!==null&&(e.getClearColor(this._oldClearColor),e.setClearColor(this.clearColor,e.getClearAlpha())),this.clearAlpha!==null&&(r=e.getClearAlpha(),e.setClearAlpha(this.clearAlpha)),this.clearDepth==!0&&e.clearDepth(),e.setRenderTarget(this.renderToScreen?null:i),this.clear===!0&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),e.render(this.scene,this.camera),this.clearColor!==null&&e.setClearColor(this._oldClearColor),this.clearAlpha!==null&&e.setClearAlpha(r),this.overrideMaterial!==null&&(this.scene.overrideMaterial=o),e.autoClear=s}};var _a={name:"GTAOShader",defines:{PERSPECTIVE_CAMERA:1,SAMPLES:16,NORMAL_VECTOR_TYPE:1,DEPTH_SWIZZLING:"x",SCREEN_SPACE_RADIUS:0,SCREEN_SPACE_RADIUS_SCALE:100,SCENE_CLIP_BOX:0},uniforms:{tNormal:{value:null},tDepth:{value:null},tNoise:{value:null},resolution:{value:new $},cameraNear:{value:null},cameraFar:{value:null},cameraProjectionMatrix:{value:new rt},cameraProjectionMatrixInverse:{value:new rt},cameraWorldMatrix:{value:new rt},radius:{value:.25},distanceExponent:{value:1},thickness:{value:1},distanceFallOff:{value:1},scale:{value:1},sceneBoxMin:{value:new P(-1,-1,-1)},sceneBoxMax:{value:new P(1,1,1)}},vertexShader:`

		varying vec2 vUv;

		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
		}`,fragmentShader:`
		varying vec2 vUv;
		uniform highp sampler2D tNormal;
		uniform highp sampler2D tDepth;
		uniform sampler2D tNoise;
		uniform vec2 resolution;
		uniform float cameraNear;
		uniform float cameraFar;
		uniform mat4 cameraProjectionMatrix;
		uniform mat4 cameraProjectionMatrixInverse;
		uniform mat4 cameraWorldMatrix;
		uniform float radius;
		uniform float distanceExponent;
		uniform float thickness;
		uniform float distanceFallOff;
		uniform float scale;
		#if SCENE_CLIP_BOX == 1
			uniform vec3 sceneBoxMin;
			uniform vec3 sceneBoxMax;
		#endif

		#include <common>
		#include <packing>

		#ifndef FRAGMENT_OUTPUT
		#define FRAGMENT_OUTPUT vec4(vec3(ao), 1.)
		#endif

		vec3 getViewPosition( const in vec2 screenPosition, const in float depth ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				vec4 clipSpacePosition = vec4( vec2( screenPosition ) * 2.0 - 1.0, depth, 1.0 );
			#else
				vec4 clipSpacePosition = vec4( vec3( screenPosition, depth ) * 2.0 - 1.0, 1.0 );
			#endif
			vec4 viewSpacePosition = cameraProjectionMatrixInverse * clipSpacePosition;
			return viewSpacePosition.xyz / viewSpacePosition.w;
		}

		float getDepth(const vec2 uv) {
			return textureLod(tDepth, uv.xy, 0.0).DEPTH_SWIZZLING;
		}

		float fetchDepth(const ivec2 uv) {
			return texelFetch(tDepth, uv.xy, 0).DEPTH_SWIZZLING;
		}

		float getViewZ(const in float depth) {
			#if PERSPECTIVE_CAMERA == 1
				return perspectiveDepthToViewZ(depth, cameraNear, cameraFar);
			#else
				return orthographicDepthToViewZ(depth, cameraNear, cameraFar);
			#endif
		}

		vec3 computeNormalFromDepth(const vec2 uv) {
			vec2 size = vec2(textureSize(tDepth, 0));
			ivec2 p = ivec2(uv * size);
			float c0 = fetchDepth(p);
			float l2 = fetchDepth(p - ivec2(2, 0));
			float l1 = fetchDepth(p - ivec2(1, 0));
			float r1 = fetchDepth(p + ivec2(1, 0));
			float r2 = fetchDepth(p + ivec2(2, 0));
			float b2 = fetchDepth(p - ivec2(0, 2));
			float b1 = fetchDepth(p - ivec2(0, 1));
			float t1 = fetchDepth(p + ivec2(0, 1));
			float t2 = fetchDepth(p + ivec2(0, 2));
			float dl = abs((2.0 * l1 - l2) - c0);
			float dr = abs((2.0 * r1 - r2) - c0);
			float db = abs((2.0 * b1 - b2) - c0);
			float dt = abs((2.0 * t1 - t2) - c0);
			vec3 ce = getViewPosition(uv, c0).xyz;
			vec3 dpdx = (dl < dr) ? ce - getViewPosition((uv - vec2(1.0 / size.x, 0.0)), l1).xyz : -ce + getViewPosition((uv + vec2(1.0 / size.x, 0.0)), r1).xyz;
			vec3 dpdy = (db < dt) ? ce - getViewPosition((uv - vec2(0.0, 1.0 / size.y)), b1).xyz : -ce + getViewPosition((uv + vec2(0.0, 1.0 / size.y)), t1).xyz;
			return normalize(cross(dpdx, dpdy));
		}

		vec3 getViewNormal(const vec2 uv) {
			#if NORMAL_VECTOR_TYPE == 2
				return normalize(textureLod(tNormal, uv, 0.).rgb);
			#elif NORMAL_VECTOR_TYPE == 1
				return unpackRGBToNormal(textureLod(tNormal, uv, 0.).rgb);
			#else
				return computeNormalFromDepth(uv);
			#endif
		}

		vec3 getSceneUvAndDepth(vec3 sampleViewPos) {
			vec4 sampleClipPos = cameraProjectionMatrix * vec4(sampleViewPos, 1.);
			vec2 sampleUv = sampleClipPos.xy / sampleClipPos.w * 0.5 + 0.5;
			float sampleSceneDepth = getDepth(sampleUv);
			return vec3(sampleUv, sampleSceneDepth);
		}

		void main() {
			float depth = getDepth(vUv.xy);

			#ifdef USE_REVERSED_DEPTH_BUFFER
				if (depth <= 0.0) {
					discard;
					return;
				}
			#else
				if (depth >= 1.0) {
					discard;
					return;
				}
			#endif
			
			vec3 viewPos = getViewPosition(vUv, depth);
			vec3 viewNormal = getViewNormal(vUv);

			float radiusToUse = radius;
			float distanceFalloffToUse = thickness;
			#if SCREEN_SPACE_RADIUS == 1
				float radiusScale = getViewPosition(vec2(0.5 + float(SCREEN_SPACE_RADIUS_SCALE) / resolution.x, 0.0), depth).x;
				radiusToUse *= radiusScale;
				distanceFalloffToUse *= radiusScale;
			#endif

			#if SCENE_CLIP_BOX == 1
				vec3 worldPos = (cameraWorldMatrix * vec4(viewPos, 1.0)).xyz;
				float boxDistance = length(max(vec3(0.0), max(sceneBoxMin - worldPos, worldPos - sceneBoxMax)));
				if (boxDistance > radiusToUse) {
					discard;
					return;
				}
			#endif

			vec2 noiseResolution = vec2(textureSize(tNoise, 0));
			vec2 noiseUv = vUv * resolution / noiseResolution;
			vec4 noiseTexel = textureLod(tNoise, noiseUv, 0.0);
			vec3 randomVec = noiseTexel.xyz * 2.0 - 1.0;
			vec3 tangent = normalize(vec3(randomVec.xy, 0.));
			vec3 bitangent = vec3(-tangent.y, tangent.x, 0.);
			mat3 kernelMatrix = mat3(tangent, bitangent, vec3(0., 0., 1.));

			const int DIRECTIONS = SAMPLES < 30 ? 3 : 5;
			const int STEPS = (SAMPLES + DIRECTIONS - 1) / DIRECTIONS;
			float ao = 0.0;
			for (int i = 0; i < DIRECTIONS; ++i) {

				float angle = float(i) / float(DIRECTIONS) * PI;
				vec4 sampleDir = vec4(cos(angle), sin(angle), 0., 0.5 + 0.5 * noiseTexel.w);
				sampleDir.xyz = normalize(kernelMatrix * sampleDir.xyz);

				vec3 viewDir = normalize(-viewPos.xyz);
				vec3 sliceBitangent = normalize(cross(sampleDir.xyz, viewDir));
				vec3 sliceTangent = cross(sliceBitangent, viewDir);
				vec3 normalInSlice = normalize(viewNormal - sliceBitangent * dot(viewNormal, sliceBitangent));

				vec3 tangentToNormalInSlice = cross(normalInSlice, sliceBitangent);
				vec2 cosHorizons = vec2(dot(viewDir, tangentToNormalInSlice), dot(viewDir, -tangentToNormalInSlice));

				for (int j = 0; j < STEPS; ++j) {
					vec3 sampleViewOffset = sampleDir.xyz * radiusToUse * sampleDir.w * pow(float(j + 1) / float(STEPS), distanceExponent);

					vec3 sampleSceneUvDepth = getSceneUvAndDepth(viewPos + sampleViewOffset);
					vec3 sampleSceneViewPos = getViewPosition(sampleSceneUvDepth.xy, sampleSceneUvDepth.z);
					vec3 viewDelta = sampleSceneViewPos - viewPos;
					if (abs(viewDelta.z) < thickness) {
						float sampleCosHorizon = dot(viewDir, normalize(viewDelta));
						cosHorizons.x += max(0., (sampleCosHorizon - cosHorizons.x) * mix(1., 2. / float(j + 2), distanceFallOff));
					}

					sampleSceneUvDepth = getSceneUvAndDepth(viewPos - sampleViewOffset);
					sampleSceneViewPos = getViewPosition(sampleSceneUvDepth.xy, sampleSceneUvDepth.z);
					viewDelta = sampleSceneViewPos - viewPos;
					if (abs(viewDelta.z) < thickness) {
						float sampleCosHorizon = dot(viewDir, normalize(viewDelta));
						cosHorizons.y += max(0., (sampleCosHorizon - cosHorizons.y) * mix(1., 2. / float(j + 2), distanceFallOff));
					}
				}

				vec2 sinHorizons = sqrt(1. - cosHorizons * cosHorizons);
				float nx = dot(normalInSlice, sliceTangent);
				float ny = dot(normalInSlice, viewDir);
				float nxb = 1. / 2. * (acos(cosHorizons.y) - acos(cosHorizons.x) + sinHorizons.x * cosHorizons.x - sinHorizons.y * cosHorizons.y);
				float nyb = 1. / 2. * (2. - cosHorizons.x * cosHorizons.x - cosHorizons.y * cosHorizons.y);
				float occlusion = nx * nxb + ny * nyb;
				ao += occlusion;
			}

			ao = clamp(ao / float(DIRECTIONS), 0., 1.);
		#if SCENE_CLIP_BOX == 1
			ao = mix(ao, 1., smoothstep(0., radiusToUse, boxDistance));
		#endif
			ao = pow(ao, scale);

			gl_FragColor = FRAGMENT_OUTPUT;
		}`},xa={name:"GTAODepthShader",defines:{PERSPECTIVE_CAMERA:1},uniforms:{tDepth:{value:null},cameraNear:{value:null},cameraFar:{value:null}},vertexShader:`
		varying vec2 vUv;

		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
		}`,fragmentShader:`
		uniform sampler2D tDepth;
		uniform float cameraNear;
		uniform float cameraFar;
		varying vec2 vUv;

		#include <packing>

		float getLinearDepth( const in vec2 screenPosition ) {
			#if PERSPECTIVE_CAMERA == 1
				float fragCoordZ = texture2D( tDepth, screenPosition ).x;
				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );
			#else
				return texture2D( tDepth, screenPosition ).x;
			#endif
		}

		void main() {
			float depth = getLinearDepth( vUv );
			gl_FragColor = vec4( vec3( 1.0 - depth ), 1.0 );

		}`},fh={name:"GTAOBlendShader",uniforms:{tDiffuse:{value:null},intensity:{value:1}},vertexShader:`
		varying vec2 vUv;

		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
		}`,fragmentShader:`
		uniform float intensity;
		uniform sampler2D tDiffuse;
		varying vec2 vUv;

		void main() {
			vec4 texel = texture2D( tDiffuse, vUv );
			gl_FragColor = vec4(mix(vec3(1.), texel.rgb, intensity), texel.a);
		}`};function Sm(n=5){let e=Math.floor(n)%2===0?Math.floor(n)+1:Math.floor(n),t=pS(e),i=t.length,s=new Uint8Array(i*4);for(let o=0;o<i;++o){let a=t[o],c=2*Math.PI*a/i,l=new P(Math.cos(c),Math.sin(c),0).normalize();s[o*4]=(l.x*.5+.5)*255,s[o*4+1]=(l.y*.5+.5)*255,s[o*4+2]=127,s[o*4+3]=255}let r=new Un(s,e,e);return r.wrapS=zi,r.wrapT=zi,r.needsUpdate=!0,r}function pS(n){let e=Math.floor(n)%2===0?Math.floor(n)+1:Math.floor(n),t=e*e,i=Array(t).fill(0),s=Math.floor(e/2),r=e-1;for(let o=1;o<=t;){if(s===-1&&r===e?(r=e-2,s=0):(r===e&&(r=0),s<0&&(s=e-1)),i[s*e+r]!==0){r-=2,s++;continue}else i[s*e+r]=o++;r++,s--}return i}var va={name:"PoissonDenoiseShader",defines:{SAMPLES:16,SAMPLE_VECTORS:Sd(16,2,1),NORMAL_VECTOR_TYPE:1,DEPTH_VALUE_SOURCE:0},uniforms:{tDiffuse:{value:null},tNormal:{value:null},tDepth:{value:null},tNoise:{value:null},resolution:{value:new $},cameraProjectionMatrixInverse:{value:new rt},lumaPhi:{value:5},depthPhi:{value:5},normalPhi:{value:5},radius:{value:4},index:{value:0}},vertexShader:`

		varying vec2 vUv;

		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
		}`,fragmentShader:`

		varying vec2 vUv;

		uniform sampler2D tDiffuse;
		uniform sampler2D tNormal;
		uniform sampler2D tDepth;
		uniform sampler2D tNoise;
		uniform vec2 resolution;
		uniform mat4 cameraProjectionMatrixInverse;
		uniform float lumaPhi;
		uniform float depthPhi;
		uniform float normalPhi;
		uniform float radius;
		uniform int index;

		#include <common>
		#include <packing>

		#ifndef SAMPLE_LUMINANCE
		#define SAMPLE_LUMINANCE dot(vec3(0.2125, 0.7154, 0.0721), a)
		#endif

		#ifndef FRAGMENT_OUTPUT
		#define FRAGMENT_OUTPUT vec4(denoised, 1.)
		#endif

		float getLuminance(const in vec3 a) {
			return SAMPLE_LUMINANCE;
		}

		const vec3 poissonDisk[SAMPLES] = SAMPLE_VECTORS;

		vec3 getViewPosition( const in vec2 screenPosition, const in float depth ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				vec4 clipSpacePosition = vec4( vec2( screenPosition ) * 2.0 - 1.0, depth, 1.0 );
			#else
				vec4 clipSpacePosition = vec4( vec3( screenPosition, depth ) * 2.0 - 1.0, 1.0 );
			#endif
			vec4 viewSpacePosition = cameraProjectionMatrixInverse * clipSpacePosition;
			return viewSpacePosition.xyz / viewSpacePosition.w;
		}

		float getDepth(const vec2 uv) {
		#if DEPTH_VALUE_SOURCE == 1
			return textureLod(tDepth, uv.xy, 0.0).a;
		#else
			return textureLod(tDepth, uv.xy, 0.0).r;
		#endif
		}

		float fetchDepth(const ivec2 uv) {
			#if DEPTH_VALUE_SOURCE == 1
				return texelFetch(tDepth, uv.xy, 0).a;
			#else
				return texelFetch(tDepth, uv.xy, 0).r;
			#endif
		}

		vec3 computeNormalFromDepth(const vec2 uv) {
			vec2 size = vec2(textureSize(tDepth, 0));
			ivec2 p = ivec2(uv * size);
			float c0 = fetchDepth(p);
			float l2 = fetchDepth(p - ivec2(2, 0));
			float l1 = fetchDepth(p - ivec2(1, 0));
			float r1 = fetchDepth(p + ivec2(1, 0));
			float r2 = fetchDepth(p + ivec2(2, 0));
			float b2 = fetchDepth(p - ivec2(0, 2));
			float b1 = fetchDepth(p - ivec2(0, 1));
			float t1 = fetchDepth(p + ivec2(0, 1));
			float t2 = fetchDepth(p + ivec2(0, 2));
			float dl = abs((2.0 * l1 - l2) - c0);
			float dr = abs((2.0 * r1 - r2) - c0);
			float db = abs((2.0 * b1 - b2) - c0);
			float dt = abs((2.0 * t1 - t2) - c0);
			vec3 ce = getViewPosition(uv, c0).xyz;
			vec3 dpdx = (dl < dr) ?  ce - getViewPosition((uv - vec2(1.0 / size.x, 0.0)), l1).xyz
									: -ce + getViewPosition((uv + vec2(1.0 / size.x, 0.0)), r1).xyz;
			vec3 dpdy = (db < dt) ?  ce - getViewPosition((uv - vec2(0.0, 1.0 / size.y)), b1).xyz
									: -ce + getViewPosition((uv + vec2(0.0, 1.0 / size.y)), t1).xyz;
			return normalize(cross(dpdx, dpdy));
		}

		vec3 getViewNormal(const vec2 uv) {
		#if NORMAL_VECTOR_TYPE == 2
			return normalize(textureLod(tNormal, uv, 0.).rgb);
		#elif NORMAL_VECTOR_TYPE == 1
			return unpackRGBToNormal(textureLod(tNormal, uv, 0.).rgb);
		#else
			return computeNormalFromDepth(uv);
		#endif
		}

		void denoiseSample(in vec3 center, in vec3 viewNormal, in vec3 viewPos, in vec2 sampleUv, inout vec3 denoised, inout float totalWeight) {
			vec4 sampleTexel = textureLod(tDiffuse, sampleUv, 0.0);
			float sampleDepth = getDepth(sampleUv);
			vec3 sampleNormal = getViewNormal(sampleUv);
			vec3 neighborColor = sampleTexel.rgb;
			vec3 viewPosSample = getViewPosition(sampleUv, sampleDepth);

			float normalDiff = dot(viewNormal, sampleNormal);
			float normalSimilarity = pow(max(normalDiff, 0.), normalPhi);
			float lumaDiff = abs(getLuminance(neighborColor) - getLuminance(center));
			float lumaSimilarity = max(1.0 - lumaDiff / lumaPhi, 0.0);
			float depthDiff = abs(dot(viewPos - viewPosSample, viewNormal));
			float depthSimilarity = max(1. - depthDiff / depthPhi, 0.);
			float w = lumaSimilarity * depthSimilarity * normalSimilarity;

			denoised += w * neighborColor;
			totalWeight += w;
		}

		void main() {
			float depth = getDepth(vUv.xy);
			vec3 viewNormal = getViewNormal(vUv);
			if (depth == 1. || dot(viewNormal, viewNormal) == 0.) {
				discard;
				return;
			}
			vec4 texel = textureLod(tDiffuse, vUv, 0.0);
			vec3 center = texel.rgb;
			vec3 viewPos = getViewPosition(vUv, depth);

			vec2 noiseResolution = vec2(textureSize(tNoise, 0));
			vec2 noiseUv = vUv * resolution / noiseResolution;
			vec4 noiseTexel = textureLod(tNoise, noiseUv, 0.0);
      		vec2 noiseVec = vec2(sin(noiseTexel[index % 4] * 2. * PI), cos(noiseTexel[index % 4] * 2. * PI));
    		mat2 rotationMatrix = mat2(noiseVec.x, -noiseVec.y, noiseVec.x, noiseVec.y);

			float totalWeight = 1.0;
			vec3 denoised = texel.rgb;
			for (int i = 0; i < SAMPLES; i++) {
				vec3 sampleDir = poissonDisk[i];
				vec2 offset = rotationMatrix * (sampleDir.xy * (1. + sampleDir.z * (radius - 1.)) / resolution);
				vec2 sampleUv = vUv + offset;
				denoiseSample(center, viewNormal, viewPos, sampleUv, denoised, totalWeight);
			}

			if (totalWeight > 0.) {
				denoised /= totalWeight;
			}
			gl_FragColor = FRAGMENT_OUTPUT;
		}`};function Sd(n,e,t){let i=mS(n,e,t),s="vec3[SAMPLES](";for(let r=0;r<n;r++){let o=i[r];s+=`vec3(${o.x}, ${o.y}, ${o.z})${r<n-1?",":")"}`}return s}function mS(n,e,t){let i=[];for(let s=0;s<n;s++){let r=2*Math.PI*e*s/n,o=Math.pow(s/(n-1),t);i.push(new P(Math.cos(r),Math.sin(r),o))}return i}var ph=class{constructor(e=Math){this.grad3=[[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]],this.grad4=[[0,1,1,1],[0,1,1,-1],[0,1,-1,1],[0,1,-1,-1],[0,-1,1,1],[0,-1,1,-1],[0,-1,-1,1],[0,-1,-1,-1],[1,0,1,1],[1,0,1,-1],[1,0,-1,1],[1,0,-1,-1],[-1,0,1,1],[-1,0,1,-1],[-1,0,-1,1],[-1,0,-1,-1],[1,1,0,1],[1,1,0,-1],[1,-1,0,1],[1,-1,0,-1],[-1,1,0,1],[-1,1,0,-1],[-1,-1,0,1],[-1,-1,0,-1],[1,1,1,0],[1,1,-1,0],[1,-1,1,0],[1,-1,-1,0],[-1,1,1,0],[-1,1,-1,0],[-1,-1,1,0],[-1,-1,-1,0]],this.p=[];for(let t=0;t<256;t++)this.p[t]=Math.floor(e.random()*256);this.perm=[];for(let t=0;t<512;t++)this.perm[t]=this.p[t&255];this.simplex=[[0,1,2,3],[0,1,3,2],[0,0,0,0],[0,2,3,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,3,0],[0,2,1,3],[0,0,0,0],[0,3,1,2],[0,3,2,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,3,2,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,0,3],[0,0,0,0],[1,3,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,3,0,1],[2,3,1,0],[1,0,2,3],[1,0,3,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,3,1],[0,0,0,0],[2,1,3,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,1,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,0,1,2],[3,0,2,1],[0,0,0,0],[3,1,2,0],[2,1,0,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,1,0,2],[0,0,0,0],[3,2,0,1],[3,2,1,0]]}noise(e,t){let i,s,r,o=.5*(Math.sqrt(3)-1),a=(e+t)*o,c=Math.floor(e+a),l=Math.floor(t+a),h=(3-Math.sqrt(3))/6,u=(c+l)*h,d=c-u,f=l-u,g=e-d,x=t-f,p,m;g>x?(p=1,m=0):(p=0,m=1);let M=g-p+h,b=x-m+h,y=g-1+2*h,T=x-1+2*h,S=c&255,A=l&255,_=this.perm[S+this.perm[A]]%12,E=this.perm[S+p+this.perm[A+m]]%12,C=this.perm[S+1+this.perm[A+1]]%12,I=.5-g*g-x*x;I<0?i=0:(I*=I,i=I*I*this._dot(this.grad3[_],g,x));let L=.5-M*M-b*b;L<0?s=0:(L*=L,s=L*L*this._dot(this.grad3[E],M,b));let V=.5-y*y-T*T;return V<0?r=0:(V*=V,r=V*V*this._dot(this.grad3[C],y,T)),70*(i+s+r)}noise3d(e,t,i){let s,r,o,a,l=(e+t+i)*.3333333333333333,h=Math.floor(e+l),u=Math.floor(t+l),d=Math.floor(i+l),f=1/6,g=(h+u+d)*f,x=h-g,p=u-g,m=d-g,M=e-x,b=t-p,y=i-m,T,S,A,_,E,C;M>=b?b>=y?(T=1,S=0,A=0,_=1,E=1,C=0):M>=y?(T=1,S=0,A=0,_=1,E=0,C=1):(T=0,S=0,A=1,_=1,E=0,C=1):b<y?(T=0,S=0,A=1,_=0,E=1,C=1):M<y?(T=0,S=1,A=0,_=0,E=1,C=1):(T=0,S=1,A=0,_=1,E=1,C=0);let I=M-T+f,L=b-S+f,V=y-A+f,q=M-_+2*f,N=b-E+2*f,Y=y-C+2*f,X=M-1+3*f,ne=b-1+3*f,ie=y-1+3*f,ge=h&255,ue=u&255,xe=d&255,Ne=this.perm[ge+this.perm[ue+this.perm[xe]]]%12,st=this.perm[ge+T+this.perm[ue+S+this.perm[xe+A]]]%12,Xe=this.perm[ge+_+this.perm[ue+E+this.perm[xe+C]]]%12,j=this.perm[ge+1+this.perm[ue+1+this.perm[xe+1]]]%12,he=.6-M*M-b*b-y*y;he<0?s=0:(he*=he,s=he*he*this._dot3(this.grad3[Ne],M,b,y));let le=.6-I*I-L*L-V*V;le<0?r=0:(le*=le,r=le*le*this._dot3(this.grad3[st],I,L,V));let Ae=.6-q*q-N*N-Y*Y;Ae<0?o=0:(Ae*=Ae,o=Ae*Ae*this._dot3(this.grad3[Xe],q,N,Y));let Fe=.6-X*X-ne*ne-ie*ie;return Fe<0?a=0:(Fe*=Fe,a=Fe*Fe*this._dot3(this.grad3[j],X,ne,ie)),32*(s+r+o+a)}noise4d(e,t,i,s){let r=this.grad4,o=this.simplex,a=this.perm,c=(Math.sqrt(5)-1)/4,l=(5-Math.sqrt(5))/20,h,u,d,f,g,x=(e+t+i+s)*c,p=Math.floor(e+x),m=Math.floor(t+x),M=Math.floor(i+x),b=Math.floor(s+x),y=(p+m+M+b)*l,T=p-y,S=m-y,A=M-y,_=b-y,E=e-T,C=t-S,I=i-A,L=s-_,V=E>C?32:0,q=E>I?16:0,N=C>I?8:0,Y=E>L?4:0,X=C>L?2:0,ne=I>L?1:0,ie=V+q+N+Y+X+ne,ge=o[ie][0]>=3?1:0,ue=o[ie][1]>=3?1:0,xe=o[ie][2]>=3?1:0,Ne=o[ie][3]>=3?1:0,st=o[ie][0]>=2?1:0,Xe=o[ie][1]>=2?1:0,j=o[ie][2]>=2?1:0,he=o[ie][3]>=2?1:0,le=o[ie][0]>=1?1:0,Ae=o[ie][1]>=1?1:0,Fe=o[ie][2]>=1?1:0,ke=o[ie][3]>=1?1:0,ae=E-ge+l,ee=C-ue+l,O=I-xe+l,H=L-Ne+l,Q=E-st+2*l,W=C-Xe+2*l,G=I-j+2*l,se=L-he+2*l,ce=E-le+3*l,fe=C-Ae+3*l,me=I-Fe+3*l,D=L-ke+3*l,Me=E-1+4*l,Ve=C-1+4*l,R=I-1+4*l,v=L-1+4*l,U=p&255,B=m&255,k=M&255,pe=b&255,_e=a[U+a[B+a[k+a[pe]]]]%32,te=a[U+ge+a[B+ue+a[k+xe+a[pe+Ne]]]]%32,re=a[U+st+a[B+Xe+a[k+j+a[pe+he]]]]%32,Se=a[U+le+a[B+Ae+a[k+Fe+a[pe+ke]]]]%32,Ie=a[U+1+a[B+1+a[k+1+a[pe+1]]]]%32,ve=.6-E*E-C*C-I*I-L*L;ve<0?h=0:(ve*=ve,h=ve*ve*this._dot4(r[_e],E,C,I,L));let ye=.6-ae*ae-ee*ee-O*O-H*H;ye<0?u=0:(ye*=ye,u=ye*ye*this._dot4(r[te],ae,ee,O,H));let Be=.6-Q*Q-W*W-G*G-se*se;Be<0?d=0:(Be*=Be,d=Be*Be*this._dot4(r[re],Q,W,G,se));let qe=.6-ce*ce-fe*fe-me*me-D*D;qe<0?f=0:(qe*=qe,f=qe*qe*this._dot4(r[Se],ce,fe,me,D));let Je=.6-Me*Me-Ve*Ve-R*R-v*v;return Je<0?g=0:(Je*=Je,g=Je*Je*this._dot4(r[Ie],Me,Ve,R,v)),27*(h+u+d+f+g)}_dot(e,t,i){return e[0]*t+e[1]*i}_dot3(e,t,i,s){return e[0]*t+e[1]*i+e[2]*s}_dot4(e,t,i,s,r){return e[0]*t+e[1]*i+e[2]*s+e[3]*r}};var ya=class n extends Ni{constructor(e,t,i=512,s=512,r,o,a){super(),this.width=i,this.height=s,this.clear=!0,this.camera=t,this.scene=e,this.output=0,this._renderGBuffer=!0,this._visibilityCache=[],this.blendIntensity=1,this.pdRings=2,this.pdRadiusExponent=2,this.pdSamples=16,this.gtaoNoiseTexture=Sm(),this.pdNoiseTexture=this._generateNoise(),this.gtaoRenderTarget=new Ht(this.width,this.height,{type:ii}),this.pdRenderTarget=this.gtaoRenderTarget.clone(),this.gtaoMaterial=new bt({defines:Object.assign({},_a.defines),uniforms:mi.clone(_a.uniforms),vertexShader:_a.vertexShader,fragmentShader:_a.fragmentShader,blending:zt,depthTest:!1,depthWrite:!1}),this.gtaoMaterial.defines.PERSPECTIVE_CAMERA=this.camera.isPerspectiveCamera?1:0,this.gtaoMaterial.uniforms.tNoise.value=this.gtaoNoiseTexture,this.gtaoMaterial.uniforms.resolution.value.set(this.width,this.height),this.gtaoMaterial.uniforms.cameraNear.value=this.camera.near,this.gtaoMaterial.uniforms.cameraFar.value=this.camera.far,this.normalMaterial=new Fo,this.normalMaterial.blending=zt,this.pdMaterial=new bt({defines:Object.assign({},va.defines),uniforms:mi.clone(va.uniforms),vertexShader:va.vertexShader,fragmentShader:va.fragmentShader,depthTest:!1,depthWrite:!1}),this.pdMaterial.uniforms.tDiffuse.value=this.gtaoRenderTarget.texture,this.pdMaterial.uniforms.tNoise.value=this.pdNoiseTexture,this.pdMaterial.uniforms.resolution.value.set(this.width,this.height),this.pdMaterial.uniforms.lumaPhi.value=10,this.pdMaterial.uniforms.depthPhi.value=2,this.pdMaterial.uniforms.normalPhi.value=3,this.pdMaterial.uniforms.radius.value=8,this.depthRenderMaterial=new bt({defines:Object.assign({},xa.defines),uniforms:mi.clone(xa.uniforms),vertexShader:xa.vertexShader,fragmentShader:xa.fragmentShader,blending:zt}),this.depthRenderMaterial.uniforms.cameraNear.value=this.camera.near,this.depthRenderMaterial.uniforms.cameraFar.value=this.camera.far,this.copyMaterial=new bt({uniforms:mi.clone(Xr.uniforms),vertexShader:Xr.vertexShader,fragmentShader:Xr.fragmentShader,transparent:!0,depthTest:!1,depthWrite:!1,blendSrc:$o,blendDst:Ns,blendEquation:Ci,blendSrcAlpha:Yo,blendDstAlpha:Ns,blendEquationAlpha:Ci}),this.blendMaterial=new bt({uniforms:mi.clone(fh.uniforms),vertexShader:fh.vertexShader,fragmentShader:fh.fragmentShader,transparent:!0,depthTest:!1,depthWrite:!1,blending:Zl,blendSrc:$o,blendDst:Ns,blendEquation:Ci,blendSrcAlpha:Yo,blendDstAlpha:Ns,blendEquationAlpha:Ci}),this._fsQuad=new vs(null),this._originalClearColor=new Te,this.setGBuffer(r?r.depthTexture:void 0,r?r.normalTexture:void 0),o!==void 0&&this.updateGtaoMaterial(o),a!==void 0&&this.updatePdMaterial(a)}setSize(e,t){this.width=e,this.height=t,this.gtaoRenderTarget.setSize(e,t),this.normalRenderTarget.setSize(e,t),this.pdRenderTarget.setSize(e,t),this.gtaoMaterial.uniforms.resolution.value.set(e,t),this.gtaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.gtaoMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this.pdMaterial.uniforms.resolution.value.set(e,t),this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse)}dispose(){this.gtaoNoiseTexture.dispose(),this.pdNoiseTexture.dispose(),this.normalRenderTarget.dispose(),this.gtaoRenderTarget.dispose(),this.pdRenderTarget.dispose(),this.normalMaterial.dispose(),this.pdMaterial.dispose(),this.copyMaterial.dispose(),this.depthRenderMaterial.dispose(),this._fsQuad.dispose()}get gtaoMap(){return this.pdRenderTarget.texture}setGBuffer(e,t){e!==void 0?(this.depthTexture=e,this.normalTexture=t,this._renderGBuffer=!1):(this.depthTexture=new ji,this.depthTexture.format=_n,this.depthTexture.type=ps,this.normalRenderTarget=new Ht(this.width,this.height,{minFilter:Ot,magFilter:Ot,type:ii,depthTexture:this.depthTexture}),this.normalTexture=this.normalRenderTarget.texture,this._renderGBuffer=!0);let i=this.normalTexture?1:0,s=this.depthTexture===this.normalTexture?"w":"x";this.gtaoMaterial.defines.NORMAL_VECTOR_TYPE=i,this.gtaoMaterial.defines.DEPTH_SWIZZLING=s,this.gtaoMaterial.uniforms.tNormal.value=this.normalTexture,this.gtaoMaterial.uniforms.tDepth.value=this.depthTexture,this.pdMaterial.defines.NORMAL_VECTOR_TYPE=i,this.pdMaterial.defines.DEPTH_SWIZZLING=s,this.pdMaterial.uniforms.tNormal.value=this.normalTexture,this.pdMaterial.uniforms.tDepth.value=this.depthTexture,this.depthRenderMaterial.uniforms.tDepth.value=this.normalRenderTarget.depthTexture}setSceneClipBox(e){e?(this.gtaoMaterial.needsUpdate=this.gtaoMaterial.defines.SCENE_CLIP_BOX!==1,this.gtaoMaterial.defines.SCENE_CLIP_BOX=1,this.gtaoMaterial.uniforms.sceneBoxMin.value.copy(e.min),this.gtaoMaterial.uniforms.sceneBoxMax.value.copy(e.max)):(this.gtaoMaterial.needsUpdate=this.gtaoMaterial.defines.SCENE_CLIP_BOX===0,this.gtaoMaterial.defines.SCENE_CLIP_BOX=0)}updateGtaoMaterial(e){e.radius!==void 0&&(this.gtaoMaterial.uniforms.radius.value=e.radius),e.distanceExponent!==void 0&&(this.gtaoMaterial.uniforms.distanceExponent.value=e.distanceExponent),e.thickness!==void 0&&(this.gtaoMaterial.uniforms.thickness.value=e.thickness),e.distanceFallOff!==void 0&&(this.gtaoMaterial.uniforms.distanceFallOff.value=e.distanceFallOff,this.gtaoMaterial.needsUpdate=!0),e.scale!==void 0&&(this.gtaoMaterial.uniforms.scale.value=e.scale),e.samples!==void 0&&e.samples!==this.gtaoMaterial.defines.SAMPLES&&(this.gtaoMaterial.defines.SAMPLES=e.samples,this.gtaoMaterial.needsUpdate=!0),e.screenSpaceRadius!==void 0&&(e.screenSpaceRadius?1:0)!==this.gtaoMaterial.defines.SCREEN_SPACE_RADIUS&&(this.gtaoMaterial.defines.SCREEN_SPACE_RADIUS=e.screenSpaceRadius?1:0,this.gtaoMaterial.needsUpdate=!0)}updatePdMaterial(e){let t=!1;e.lumaPhi!==void 0&&(this.pdMaterial.uniforms.lumaPhi.value=e.lumaPhi),e.depthPhi!==void 0&&(this.pdMaterial.uniforms.depthPhi.value=e.depthPhi),e.normalPhi!==void 0&&(this.pdMaterial.uniforms.normalPhi.value=e.normalPhi),e.radius!==void 0&&e.radius!==this.radius&&(this.pdMaterial.uniforms.radius.value=e.radius),e.radiusExponent!==void 0&&e.radiusExponent!==this.pdRadiusExponent&&(this.pdRadiusExponent=e.radiusExponent,t=!0),e.rings!==void 0&&e.rings!==this.pdRings&&(this.pdRings=e.rings,t=!0),e.samples!==void 0&&e.samples!==this.pdSamples&&(this.pdSamples=e.samples,t=!0),t&&(this.pdMaterial.defines.SAMPLES=this.pdSamples,this.pdMaterial.defines.SAMPLE_VECTORS=Sd(this.pdSamples,this.pdRings,this.pdRadiusExponent),this.pdMaterial.needsUpdate=!0)}render(e,t,i){switch(this._renderGBuffer&&(this._overrideVisibility(),this._renderOverride(e,this.normalMaterial,this.normalRenderTarget,7829503,1),this._restoreVisibility()),this.gtaoMaterial.uniforms.cameraNear.value=this.camera.near,this.gtaoMaterial.uniforms.cameraFar.value=this.camera.far,this.gtaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.gtaoMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this.gtaoMaterial.uniforms.cameraWorldMatrix.value.copy(this.camera.matrixWorld),this._renderPass(e,this.gtaoMaterial,this.gtaoRenderTarget,16777215,1),this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this._renderPass(e,this.pdMaterial,this.pdRenderTarget,16777215,1),this.output){case n.OUTPUT.Off:break;case n.OUTPUT.Diffuse:this.copyMaterial.uniforms.tDiffuse.value=i.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.AO:this.copyMaterial.uniforms.tDiffuse.value=this.gtaoRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Denoise:this.copyMaterial.uniforms.tDiffuse.value=this.pdRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Depth:this.depthRenderMaterial.uniforms.cameraNear.value=this.camera.near,this.depthRenderMaterial.uniforms.cameraFar.value=this.camera.far,this._renderPass(e,this.depthRenderMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Normal:this.copyMaterial.uniforms.tDiffuse.value=this.normalRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Default:this.copyMaterial.uniforms.tDiffuse.value=i.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t),this.blendMaterial.uniforms.intensity.value=this.blendIntensity,this.blendMaterial.uniforms.tDiffuse.value=this.pdRenderTarget.texture,this._renderPass(e,this.blendMaterial,this.renderToScreen?null:t);break;default:console.warn("THREE.GTAOPass: Unknown output type.")}}_renderPass(e,t,i,s,r){e.getClearColor(this._originalClearColor);let o=e.getClearAlpha(),a=e.autoClear;e.setRenderTarget(i),e.autoClear=!1,s!=null&&(e.setClearColor(s),e.setClearAlpha(r||0),e.clear()),this._fsQuad.material=t,this._fsQuad.render(e),e.autoClear=a,e.setClearColor(this._originalClearColor),e.setClearAlpha(o)}_renderOverride(e,t,i,s,r){e.getClearColor(this._originalClearColor);let o=e.getClearAlpha(),a=e.autoClear;e.setRenderTarget(i),e.autoClear=!1,s=t.clearColor||s,r=t.clearAlpha||r,s!=null&&(e.setClearColor(s),e.setClearAlpha(r||0),e.clear()),this.scene.overrideMaterial=t,e.render(this.scene,this.camera),this.scene.overrideMaterial=null,e.autoClear=a,e.setClearColor(this._originalClearColor),e.setClearAlpha(o)}_overrideVisibility(){let e=this.scene,t=this._visibilityCache;e.traverse(function(i){(i.isPoints||i.isLine||i.isLine2)&&i.visible&&(i.visible=!1,t.push(i))})}_restoreVisibility(){let e=this._visibilityCache;for(let t=0;t<e.length;t++)e[t].visible=!0;e.length=0}_generateNoise(e=64){let t=new ph,i=e*e*4,s=new Uint8Array(i);for(let o=0;o<e;o++)for(let a=0;a<e;a++){let c=o,l=a;s[(o*e+a)*4]=(t.noise(c,l)*.5+.5)*255,s[(o*e+a)*4+1]=(t.noise(c+e,l)*.5+.5)*255,s[(o*e+a)*4+2]=(t.noise(c,l+e)*.5+.5)*255,s[(o*e+a)*4+3]=(t.noise(c+e,l+e)*.5+.5)*255}let r=new Un(s,e,e,Si,ci);return r.wrapS=zi,r.wrapT=zi,r.needsUpdate=!0,r}};ya.OUTPUT={Off:-1,Default:0,Diffuse:1,Depth:2,Normal:3,AO:4,Denoise:5};var Ma={name:"OutputShader",uniforms:{tDiffuse:{value:null},toneMappingExposure:{value:1}},vertexShader:`
		precision highp float;

		uniform mat4 modelViewMatrix;
		uniform mat4 projectionMatrix;

		attribute vec3 position;
		attribute vec2 uv;

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`

		precision highp float;

		uniform sampler2D tDiffuse;

		#include <tonemapping_pars_fragment>
		#include <colorspace_pars_fragment>

		varying vec2 vUv;

		void main() {

			gl_FragColor = texture2D( tDiffuse, vUv );

			// tone mapping

			#ifdef LINEAR_TONE_MAPPING

				gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );

			#elif defined( REINHARD_TONE_MAPPING )

				gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );

			#elif defined( CINEON_TONE_MAPPING )

				gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );

			#elif defined( ACES_FILMIC_TONE_MAPPING )

				gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );

			#elif defined( AGX_TONE_MAPPING )

				gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );

			#elif defined( NEUTRAL_TONE_MAPPING )

				gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );

			#elif defined( CUSTOM_TONE_MAPPING )

				gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );

			#endif

			// color space

			#ifdef SRGB_TRANSFER

				gl_FragColor = sRGBTransferOETF( gl_FragColor );

			#endif

		}`};var mh=class extends Ni{constructor(){super(),this.isOutputPass=!0,this.uniforms=mi.clone(Ma.uniforms),this.material=new Ar({name:Ma.name,uniforms:this.uniforms,vertexShader:Ma.vertexShader,fragmentShader:Ma.fragmentShader}),this._fsQuad=new vs(this.material),this._outputColorSpace=null,this._toneMapping=null}render(e,t,i){this.uniforms.tDiffuse.value=i.texture,this.uniforms.toneMappingExposure.value=e.toneMappingExposure,(this._outputColorSpace!==e.outputColorSpace||this._toneMapping!==e.toneMapping)&&(this._outputColorSpace=e.outputColorSpace,this._toneMapping=e.toneMapping,this.material.defines={},ht.getTransfer(this._outputColorSpace)===pt&&(this.material.defines.SRGB_TRANSFER=""),this._toneMapping===Zo?this.material.defines.LINEAR_TONE_MAPPING="":this._toneMapping===Jo?this.material.defines.REINHARD_TONE_MAPPING="":this._toneMapping===jo?this.material.defines.CINEON_TONE_MAPPING="":this._toneMapping===us?this.material.defines.ACES_FILMIC_TONE_MAPPING="":this._toneMapping===Qo?this.material.defines.AGX_TONE_MAPPING="":this._toneMapping===Fs?this.material.defines.NEUTRAL_TONE_MAPPING="":this._toneMapping===Ko&&(this.material.defines.CUSTOM_TONE_MAPPING=""),this.material.needsUpdate=!0),this.renderToScreen===!0?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(t),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this._fsQuad.render(e))}dispose(){this.material.dispose(),this._fsQuad.dispose()}};var bm={name:"FXAAShader",uniforms:{tDiffuse:{value:null},resolution:{value:new $(1/1024,1/512)}},vertexShader:`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`

		uniform sampler2D tDiffuse;
		uniform vec2 resolution;
		varying vec2 vUv;

		#define EDGE_STEP_COUNT 6
		#define EDGE_GUESS 8.0
		#define EDGE_STEPS 1.0, 1.5, 2.0, 2.0, 2.0, 4.0
		const float edgeSteps[EDGE_STEP_COUNT] = float[EDGE_STEP_COUNT]( EDGE_STEPS );

		float _ContrastThreshold = 0.0312;
		float _RelativeThreshold = 0.063;
		float _SubpixelBlending = 1.0;

		vec4 Sample( sampler2D  tex2D, vec2 uv ) {

			return texture( tex2D, uv );

		}

		float SampleLuminance( sampler2D tex2D, vec2 uv ) {

			return dot( Sample( tex2D, uv ).rgb, vec3( 0.3, 0.59, 0.11 ) );

		}

		float SampleLuminance( sampler2D tex2D, vec2 texSize, vec2 uv, float uOffset, float vOffset ) {

			uv += texSize * vec2(uOffset, vOffset);
			return SampleLuminance(tex2D, uv);

		}

		struct LuminanceData {

			float m, n, e, s, w;
			float ne, nw, se, sw;
			float highest, lowest, contrast;

		};

		LuminanceData SampleLuminanceNeighborhood( sampler2D tex2D, vec2 texSize, vec2 uv ) {

			LuminanceData l;
			l.m = SampleLuminance( tex2D, uv );
			l.n = SampleLuminance( tex2D, texSize, uv,  0.0,  1.0 );
			l.e = SampleLuminance( tex2D, texSize, uv,  1.0,  0.0 );
			l.s = SampleLuminance( tex2D, texSize, uv,  0.0, -1.0 );
			l.w = SampleLuminance( tex2D, texSize, uv, -1.0,  0.0 );

			l.ne = SampleLuminance( tex2D, texSize, uv,  1.0,  1.0 );
			l.nw = SampleLuminance( tex2D, texSize, uv, -1.0,  1.0 );
			l.se = SampleLuminance( tex2D, texSize, uv,  1.0, -1.0 );
			l.sw = SampleLuminance( tex2D, texSize, uv, -1.0, -1.0 );

			l.highest = max( max( max( max( l.n, l.e ), l.s ), l.w ), l.m );
			l.lowest = min( min( min( min( l.n, l.e ), l.s ), l.w ), l.m );
			l.contrast = l.highest - l.lowest;
			return l;

		}

		bool ShouldSkipPixel( LuminanceData l ) {

			float threshold = max( _ContrastThreshold, _RelativeThreshold * l.highest );
			return l.contrast < threshold;

		}

		float DeterminePixelBlendFactor( LuminanceData l ) {

			float f = 2.0 * ( l.n + l.e + l.s + l.w );
			f += l.ne + l.nw + l.se + l.sw;
			f *= 1.0 / 12.0;
			f = abs( f - l.m );
			f = clamp( f / l.contrast, 0.0, 1.0 );

			float blendFactor = smoothstep( 0.0, 1.0, f );
			return blendFactor * blendFactor * _SubpixelBlending;

		}

		struct EdgeData {

			bool isHorizontal;
			float pixelStep;
			float oppositeLuminance, gradient;

		};

		EdgeData DetermineEdge( vec2 texSize, LuminanceData l ) {

			EdgeData e;
			float horizontal =
				abs( l.n + l.s - 2.0 * l.m ) * 2.0 +
				abs( l.ne + l.se - 2.0 * l.e ) +
				abs( l.nw + l.sw - 2.0 * l.w );
			float vertical =
				abs( l.e + l.w - 2.0 * l.m ) * 2.0 +
				abs( l.ne + l.nw - 2.0 * l.n ) +
				abs( l.se + l.sw - 2.0 * l.s );
			e.isHorizontal = horizontal >= vertical;

			float pLuminance = e.isHorizontal ? l.n : l.e;
			float nLuminance = e.isHorizontal ? l.s : l.w;
			float pGradient = abs( pLuminance - l.m );
			float nGradient = abs( nLuminance - l.m );

			e.pixelStep = e.isHorizontal ? texSize.y : texSize.x;

			if (pGradient < nGradient) {

				e.pixelStep = -e.pixelStep;
				e.oppositeLuminance = nLuminance;
				e.gradient = nGradient;

			} else {

				e.oppositeLuminance = pLuminance;
				e.gradient = pGradient;

			}

			return e;

		}

		float DetermineEdgeBlendFactor( sampler2D  tex2D, vec2 texSize, LuminanceData l, EdgeData e, vec2 uv ) {

			vec2 uvEdge = uv;
			vec2 edgeStep;
			if (e.isHorizontal) {

				uvEdge.y += e.pixelStep * 0.5;
				edgeStep = vec2( texSize.x, 0.0 );

			} else {

				uvEdge.x += e.pixelStep * 0.5;
				edgeStep = vec2( 0.0, texSize.y );

			}

			float edgeLuminance = ( l.m + e.oppositeLuminance ) * 0.5;
			float gradientThreshold = e.gradient * 0.25;

			vec2 puv = uvEdge + edgeStep * edgeSteps[0];
			float pLuminanceDelta = SampleLuminance( tex2D, puv ) - edgeLuminance;
			bool pAtEnd = abs( pLuminanceDelta ) >= gradientThreshold;

			for ( int i = 1; i < EDGE_STEP_COUNT && !pAtEnd; i++ ) {

				puv += edgeStep * edgeSteps[i];
				pLuminanceDelta = SampleLuminance( tex2D, puv ) - edgeLuminance;
				pAtEnd = abs( pLuminanceDelta ) >= gradientThreshold;

			}

			if ( !pAtEnd ) {

				puv += edgeStep * EDGE_GUESS;

			}

			vec2 nuv = uvEdge - edgeStep * edgeSteps[0];
			float nLuminanceDelta = SampleLuminance( tex2D, nuv ) - edgeLuminance;
			bool nAtEnd = abs( nLuminanceDelta ) >= gradientThreshold;

			for ( int i = 1; i < EDGE_STEP_COUNT && !nAtEnd; i++ ) {

				nuv -= edgeStep * edgeSteps[i];
				nLuminanceDelta = SampleLuminance( tex2D, nuv ) - edgeLuminance;
				nAtEnd = abs( nLuminanceDelta ) >= gradientThreshold;

			}

			if ( !nAtEnd ) {

				nuv -= edgeStep * EDGE_GUESS;

			}

			float pDistance, nDistance;
			if ( e.isHorizontal ) {

				pDistance = puv.x - uv.x;
				nDistance = uv.x - nuv.x;

			} else {

				pDistance = puv.y - uv.y;
				nDistance = uv.y - nuv.y;

			}

			float shortestDistance;
			bool deltaSign;
			if ( pDistance <= nDistance ) {

				shortestDistance = pDistance;
				deltaSign = pLuminanceDelta >= 0.0;

			} else {

				shortestDistance = nDistance;
				deltaSign = nLuminanceDelta >= 0.0;

			}

			if ( deltaSign == ( l.m - edgeLuminance >= 0.0 ) ) {

				return 0.0;

			}

			return 0.5 - shortestDistance / ( pDistance + nDistance );

		}

		vec4 ApplyFXAA( sampler2D  tex2D, vec2 texSize, vec2 uv ) {

			LuminanceData luminance = SampleLuminanceNeighborhood( tex2D, texSize, uv );
			if ( ShouldSkipPixel( luminance ) ) {

				return Sample( tex2D, uv );

			}

			float pixelBlend = DeterminePixelBlendFactor( luminance );
			EdgeData edge = DetermineEdge( texSize, luminance );
			float edgeBlend = DetermineEdgeBlendFactor( tex2D, texSize, luminance, edge, uv );
			float finalBlend = max( pixelBlend, edgeBlend );

			if (edge.isHorizontal) {

				uv.y += edge.pixelStep * finalBlend;

			} else {

				uv.x += edge.pixelStep * finalBlend;

			}

			return Sample( tex2D, uv );

		}

		void main() {

			gl_FragColor = ApplyFXAA( tDiffuse, resolution.xy, vUv );

		}`};var gh=class extends ys{constructor(){super(bm)}setSize(e,t){this.material.uniforms.resolution.value.set(1/e,1/t)}};var bd=class extends ya{_overrideVisibility(){super._overrideVisibility();let e=this._visibilityCache;this.scene.traverse(t=>{(t.isSprite||t.isLineSegments2||t.userData.skipAO)&&t.visible&&(t.visible=!1,e.push(t))})}_renderOverride(e,...t){let i=this.scene.background;this.scene.background=null,super._renderOverride(e,...t),this.scene.background=i}},gS={uniforms:{tDiffuse:{value:null},tDepth:{value:null},tNormal:{value:null},resolution:{value:new $(1,1)},cameraNear:{value:.1},cameraFar:{value:1e3},thickness:{value:1},strength:{value:.92},vignette:{value:.16}},vertexShader:"varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }",fragmentShader:`
    #include <packing>
    uniform sampler2D tDiffuse, tDepth, tNormal;
    uniform vec2 resolution; uniform float cameraNear, cameraFar, thickness, strength, vignette;
    varying vec2 vUv;
    float depthAt(vec2 uv) { float z = texture2D(tDepth, uv).x; return z >= 1. ? 1e6 : -perspectiveDepthToViewZ(z, cameraNear, cameraFar); }
    vec3 normalAt(vec2 uv) { return texture2D(tNormal, uv).xyz * 2. - 1.; }
    void main() {
      // Optional soft vignette (diorama style only).
      vec2 centered = vUv - .5;
      vec4 base = texture2D(tDiffuse, vUv) * vec4(vec3(1. - smoothstep(.35, .95, length(centered * vec2(1.1, 1.))) * vignette), 1.);
      float c = depthAt(vUv);
      if (c > 1e5) { gl_FragColor = base; return; }
      vec2 px = thickness / resolution;
      vec2 ox = vec2(px.x, 0.), oy = vec2(0., px.y);
      float l = depthAt(vUv - ox), r = depthAt(vUv + ox), d = depthAt(vUv - oy), u = depthAt(vUv + oy);
      // Silhouettes: draw on the near side where a neighbor is clearly farther.
      float silhouette = smoothstep(.012, .03, (max(max(l, r), max(d, u)) - c) / c);
      // 1/z is affine across a plane: its Laplacian separates real creases
      // from sub-pixel cracks between coplanar cells, which stay unlined.
      float ic = 1. / c;
      float curvature = (abs(1. / l + 1. / r - 2. * ic) + abs(1. / d + 1. / u - 2. * ic)) / ic;
      vec3 n = normalAt(vUv);
      float bend = max(max(1. - dot(n, normalAt(vUv - ox)), 1. - dot(n, normalAt(vUv + ox))), max(1. - dot(n, normalAt(vUv - oy)), 1. - dot(n, normalAt(vUv + oy))));
      float crease = smoothstep(.3, .7, bend) * smoothstep(.0015, .006, curvature);
      float edge = max(silhouette, crease) * strength;
      gl_FragColor = vec4(mix(base.rgb, base.rgb * vec3(.3, .26, .28), edge), base.a);
    }`},_h=class{constructor(e,t,i){this.renderer=e,this.scene=t,this.camera=i;let s=e.getDrawingBufferSize(new $),r=new Ht(s.x,s.y,{type:ii,samples:4});this.composer=new uh(e,r),this.composer.addPass(new dh(t,i)),this.ao=new bd(t,i,s.x,s.y),this.ao.blendIntensity=.9,this.composer.addPass(this.ao),this.ink=new ys(gS),this.ink.uniforms.tDepth.value=this.ao.depthTexture,this.ink.uniforms.tNormal.value=this.ao.normalTexture,this.composer.addPass(this.ink),this.composer.addPass(new mh),this.composer.addPass(new gh)}configure({span:e,tile:t}){this.ao.updateGtaoMaterial({radius:Math.max(t*1.6,e*.018),distanceExponent:1.4,thickness:1.2,scale:1.05,samples:16}),this.ao.updatePdMaterial({lumaPhi:10,depthPhi:2,normalPhi:3,radius:6,rings:2,samples:12})}setSize(e,t,i){this.composer.setPixelRatio(i),this.composer.setSize(e,t);let s=this.renderer.getDrawingBufferSize(new $);this.ink.uniforms.resolution.value.copy(s),this.ink.uniforms.thickness.value=1.35*i}setLook({vignette:e=0,ink:t=.92}={}){this.ink.uniforms.vignette.value=e,this.ink.uniforms.strength.value=t}setSamples(e){for(let t of[this.composer.renderTarget1,this.composer.renderTarget2])t.samples!==e&&(t.samples=e,t.dispose())}render(){this.ink.uniforms.cameraNear.value=this.camera.near,this.ink.uniforms.cameraFar.value=this.camera.far,this.composer.render()}dispose(){this.ao.dispose(),this.composer.dispose()}};var Ws={dig:{color:15113984,opacity:.55,map:"target",pattern:"hatch",test:n=>n<0},dump:{color:40563,opacity:.5,map:"target",pattern:"dots",test:n=>n>0},restricted:{color:13983232,opacity:.42,map:"dumpability_static",pattern:"cross",test:n=>!n},dumpability:{color:29362,opacity:.28,map:"dumpability",pattern:"solid",test:n=>!!n},interaction:{color:5682409,opacity:.26,map:"interaction",pattern:"solid",test:n=>!!n}},Hw=new rt,Gn=new ft,Em=new Te,wm=n=>n*n*(3-2*n),_S=n=>n<.5?4*n*n*n:1-(-2*n+2)**3/2,xS=n=>1+(1.4+1)*(n-1)**3+1.4*(n-1)**2,xh=(n,e,t)=>n+(e-n)*t,bn=(n,e,t)=>Math.min(t,Math.max(e,n)),vS=["dig","dump","transfer"],Tm=["studio","paper","diorama"],yS={dig:{color:14256668,opacity:.42},dump:{color:3116906,opacity:.2,pattern:"solid"}},MS=n=>n==="studio"?Object.fromEntries(Object.entries(Ws).map(([e,t])=>[e,{...t,...yS[e]}])):Ws,Am="terra-viewer3d-quality",Rm="terra-viewer3d-presentation";function Cm(n){if(!n)return;let e=new Set,t=new Set;n.traverse(i=>{if(i.geometry&&e.add(i.geometry),i.material)for(let s of Array.isArray(i.material)?i.material:[i.material])t.add(s)});for(let i of e)i.dispose();for(let i of t)i.dispose();n.removeFromParent(),n.clear()}var Ed=class extends et{update(){}setLayer(){}};function Pm(n){try{return localStorage.getItem(n)}catch{return null}}function Im(n,e){try{localStorage.setItem(n,e)}catch{}}var vh=class{constructor(e,{onPick:t,onCameraChange:i,onError:s,onQualityChange:r}={}){this.element=e,this.onPick=t,this.onCameraChange=i,this.onError=s,this.onQualityChange=r,this.scene=new Ds,this.renderer=new Vc({antialias:!0,alpha:!1,preserveDrawingBuffer:!0,powerPreference:"high-performance"}),this.pixelRatio=Math.min(window.devicePixelRatio||1,2),this.renderer.setPixelRatio(this.pixelRatio),this.renderer.shadowMap.enabled=!0,this.renderer.shadowMap.type=Us,this.renderer.toneMapping=us,this.renderer.toneMappingExposure=1,this.renderer.outputColorSpace=Lt,e.appendChild(this.renderer.domElement),this.renderer.domElement.addEventListener("webglcontextlost",h=>{h.preventDefault(),this.onError?.(new Error("The graphics context was lost. Reload the viewer to reconnect to the scene. Your live episode remains on the server."))});let o=new Fr(this.renderer);this.scene.environment=o.fromScene(new Yc,.04).texture,this.scene.environmentIntensity=.28,o.dispose(),this.camera=new Qt(32,1,.1,2e3),this.controls=new qc(this.camera,this.renderer.domElement),this.controls.enableDamping=!0,this.controls.dampingFactor=.085,this.controls.maxPolarAngle=Math.PI*.47,this.controls.minPolarAngle=.001,this.controls.screenSpacePanning=!0,this.controls.addEventListener("start",()=>{this.tween=null,this.follow&&(this.follow=!1,this.onCameraChange?.({follow:!1}))}),this.hemi=new zo(13624575,10122832,.8),this.scene.add(this.hemi),this.sun=new Cr(16771529,2.7),this.sun.castShadow=!0,this.sun.shadow.mapSize.set(2048,2048),this.sun.shadow.bias=-4e-4,this.sun.shadow.radius=3,this.scene.add(this.sun),this.scene.add(this.sun.target),this.fill=new Cr(11127295,.35),this.scene.add(this.fill),this.world=new et,this.scene.add(this.world),this.machines=new Map,this.heightScale=1,this.visibility={dig:!0,dump:!0,restricted:!1,dumpability:!1,interaction:!0,grid:!1,tags:!0},this.raycaster=new Wo,this.pointer=new $,this.selected=null,this.reducedMotion=window.matchMedia("(prefers-reduced-motion: reduce)").matches,this.effects=new ch({groundHeight:(h,u)=>this.groundAt(h,u)}),this.scene.add(this.effects);let a=Pm(Am);this.quality=a==="fast"?"fast":"high",this.perf=a?null:{frames:0,elapsed:0};let c=Pm(Rm);this.presentation=Tm.includes(c)?c:"studio";try{this.post=new _h(this.renderer,this.scene,this.camera)}catch(h){console.warn("Post-processing unavailable",h),this.post=null,this.quality="fast"}this.lineMaterials=new Set,this.applyLook();let l=null;e.addEventListener("pointerdown",h=>{l={x:h.clientX,y:h.clientY,button:h.button}}),e.addEventListener("pointerup",h=>{l?.button===0&&Math.hypot(h.clientX-l.x,h.clientY-l.y)<5&&this.pick(h),l=null}),this.resizeObserver=new ResizeObserver(()=>this.resize()),this.resizeObserver.observe(e),this.resize(),this.clock={last:performance.now(),idle:0},this.renderer.setAnimationLoop(h=>{this.update(h),this.controls.update(),this.render()})}render(){this.quality==="high"&&this.post?this.post.render():this.renderer.render(this.scene,this.camera)}measure(e){if(!this.perf||!this.frame||this.quality!=="high"||document.hidden||(++this.perf.frames>20&&(this.perf.elapsed+=e),this.perf.frames<110))return;let t=this.perf.elapsed/(this.perf.frames-20);this.perf=null,t>1/24&&(this.quality="fast",this.onQualityChange?.("fast"))}setQuality(e){return this.quality=e==="fast"||!this.post?"fast":"high",Im(Am,this.quality),this.quality}applyLook(){let e=this.presentation,t=e==="paper",i=e==="studio",s=e==="diorama";this.palette=ma[e],this.scene.background=t?new Te(12,12,12):i?this.backdrop||(this.backdrop=cm()):this.sky||(this.sky=hm()),this.scene.fog=s&&this.span?new fo(new Te(kn.sky[1]),this.span*3.2,this.span*7.5):null,this.renderer.toneMapping=t?Fs:us,this.renderer.toneMappingExposure=1,this.scene.environmentIntensity=i?.5:.28,this.hemi.color.set(t?16777215:i?14805747:13624575),this.hemi.groundColor.set(t?9275520:i?8022616:10122832),this.hemi.intensity=t?.9:i?.6:.8,this.sun.color.set(t?16777215:i?16773340:16771529),this.sun.intensity=t?2.3:i?2.9:2.7,this.sun.shadow.radius=i?5:3,this.fill.color.set(t?16777215:i?13031925:11127295),this.fill.intensity=t?.45:i?.4:.35,this.effects.puffsEnabled=s,this.effects.dustEnabled=i,this.effects.palette=this.palette.clods,sn.uMotion.value=s?1:0,this.post?.setLook({vignette:s?.16:i?.12:0,ink:i?0:.92}),this.element.ownerDocument?.body&&(this.element.ownerDocument.body.dataset.presentation=this.presentation)}setPresentation(e){if(this.presentation=Tm.includes(e)?e:"studio",Im(Rm,this.presentation),this.applyLook(),this.frame){let t=this.camera.position.clone(),i=this.controls.target.clone();this.setFrame(this.frame,{reset:!0}),this.tween=null,this.camera.position.copy(t),this.controls.target.copy(i),this.controls.update()}return this.presentation}resize(){let e=this.element.clientWidth||1,t=this.element.clientHeight||1;this.renderer.setSize(e,t,!1),this.camera.aspect=e/t,this.camera.updateProjectionMatrix(),this.post?.setSize(e,t,this.pixelRatio);let i=this.renderer.getDrawingBufferSize(new $);this.effects.setViewport(i.y,this.camera.fov);for(let s of this.lineMaterials??[])s.resolution.copy(i)}point(e,t,i=0){let{rows:s,cols:r,tile_size_m:o}=this.frame.grid;return new P((t+.5-r/2)*o,i*this.unitHeight,(e+.5-s/2)*o*(this.frame.metric?-1:1))}heightAt(e,t,i=this.frame){e=bn(Math.round(e),0,i.grid.rows-1),t=bn(Math.round(t),0,i.grid.cols-1);let s=i.maps.action[e][t];return!i.metric&&s>0&&!i.maps.padding[e][t]?this.piles.endpointHeight(e,t,i===this.piles.previous):s*this.unitHeight}displayHeightAt(e,t){e=bn(Math.round(e),0,this.frame.grid.rows-1),t=bn(Math.round(t),0,this.frame.grid.cols-1);let i=this.displayHeights?.[e]?.[t]??this.frame.maps.action[e][t];return!this.frame.metric&&i>0&&!this.frame.maps.padding[e][t]?this.piles.nodeHeight(e*2+1,t*2+1):i*this.unitHeight}surfacePoint(e,t){let i=this.point(e,t);return i.y=this.displayHeightAt(e,t),i}groundAt(e,t){if(!this.frame)return 0;let{rows:i,cols:s,tile_size_m:r}=this.frame.grid,o=Math.floor(e/r+s/2),a=Math.floor((this.frame.metric?-t:t)/r+i/2);return a<0||o<0||a>=i||o>=s?0:this.frame.maps.padding[a][o]?this.obstacleTop??0:this.displayHeightAt(a,o)}buildWorld(e){this.terrain&&this.disposeWorld();let{rows:t,cols:i,tile_size_m:s}=e.grid,r=t*i;this.span=Math.max(t,i)*s,sn.uTile.value=s,this.camera.near=Math.max(s*.025,this.span/200),this.camera.far=this.span*20,this.camera.updateProjectionMatrix(),this.controls.minDistance=Math.max(s*2,this.span*.08),this.controls.maxDistance=this.span*4.5,this.applyLook();let o=this.span*.5+Vt.clamp(this.span*.2,6,22)+2;this.sun.position.set(-this.span*.75,this.span*1.35,-this.span*.45),Object.assign(this.sun.shadow.camera,{left:-o,right:o,top:o,bottom:-o,near:.1,far:this.span*4}),this.sun.shadow.camera.updateProjectionMatrix(),this.sun.shadow.normalBias=s*.04,this.fill.position.set(this.span*.8,this.span*.6,this.span*.9),this.post?.configure({span:this.span,tile:s});let a=Hn("soil",{vertexColors:!!e.metric,polygonOffset:!0,polygonOffsetFactor:1,polygonOffsetUnits:2},this.palette),c=Hn("soil",{vertexColors:!!e.metric,color:16777215,polygonOffset:!0,polygonOffsetFactor:-1,polygonOffsetUnits:-2},this.palette),l=new Qe({color:9204051,roughness:1});for(let h of[a,c,l])h.shadowSide=Zi;this.terrain=e.metric?new Ke(new ut,[c,a]):new jt(new Bt(1,1,1),[a,a,c,l,a,a],r),e.metric?l.dispose():this.terrain.instanceMatrix.setUsage(xn),this.terrain.castShadow=!0,this.terrain.receiveShadow=!0,this.world.add(this.terrain),this.environment=xm(e,{style:this.presentation}),this.world.add(this.environment),this.layers={},this.boundaries={},this.boundaryEntries=new Map,this.layerSettings=MS(this.presentation);for(let[h,u]of Object.entries(this.layerSettings)){let d=new ki(1,1);d.rotateX(-Math.PI/2);let f=rh({color:u.color,opacity:u.opacity,pattern:u.pattern,polygonOffset:!0,polygonOffsetFactor:-2}),g=r,x=null;if(e.metric){x=new Int32Array(r).fill(-1),g=0;let m=e.maps[u.map];if(m!=null)for(let M=0;M<t;M++)for(let b=0;b<i;b++)!e.maps.padding[M][b]&&u.test(m[M][b])&&(x[M*i+b]=g++)}let p=new jt(d,f,g);if(p.userData.cellSlots=x,p.instanceMatrix.setUsage(xn),p.visible=this.visibility[h],p.renderOrder=3+Object.keys(this.layers).length,p.frustumCulled=!1,p.userData.skipAO=!0,this.layers[h]=p,this.world.add(p),h==="dig"||h==="dump"||h==="interaction"){let m=new zr({color:new Te(u.color).multiplyScalar(.82),linewidth:2.6,transparent:!0,opacity:.95,depthWrite:!1});m.resolution.copy(this.renderer.getDrawingBufferSize(new $)),this.lineMaterials.add(m);let M=new jc(new ks,m);M.renderOrder=14,M.visible=this.visibility[h],M.frustumCulled=!1,this.boundaries[h]=M,this.world.add(M)}}this.gridLines=new ns(new ut,new Nn({color:7033138,transparent:!0,opacity:.22,depthWrite:!1})),this.gridLines.renderOrder=10,this.gridLines.visible=this.visibility.grid,this.gridLines.frustumCulled=!1,this.world.add(this.gridLines),this.selection=new ns(new Eo(new Bt(s*.99,s*.04,s*.99)),new Nn({color:16776160,depthTest:!1})),this.selection.renderOrder=20,this.selection.visible=!1,this.world.add(this.selection)}disposeWorld(){this.clearMotion(),this.effects.clear();for(let i of this.machines.values())this.scene.remove(i.root),i.dispose();this.machines.clear();let e=new Set,t=new Set;this.world.traverse(i=>{if(i.geometry&&e.add(i.geometry),i.material)for(let s of Array.isArray(i.material)?i.material:[i.material])t.add(s)});for(let i of e)i.dispose();for(let i of t)i.map?.dispose(),i.dispose();this.world.clear(),this.selected=null,this.piles=null,this.obstacleProps=null,this.environment=null,this.lineMaterials.clear()}setFrame(e,{animate:t=!1,duration:i=650,reset:s=!1}={}){let r=this.frame,o=this.unitHeight;r?.metric&&this.motion?.facts.changed.length&&(this.setFloor(this.finalFloor),this.populate(r));let a=!r||!!r.metric!=!!e.metric||r.grid.rows!==e.grid.rows||r.grid.cols!==e.grid.cols||r.grid.tile_size_m!==e.grid.tile_size_m;this.clearMotion(),this.frame=e,this.unitHeight=(e.metric?1:e.grid.tile_size_m*.48)*this.heightScale,sn.uUnit.value=this.unitHeight,(a||s)&&this.buildWorld(e);let c=pa(r,e),l=t&&!s&&!a&&!this.reducedMotion&&r&&!r.done&&e.step===r.step+1,h=0;for(let f of e.maps.action)for(let g of f)h=Math.min(h,g);if(this.finalFloor=h*this.unitHeight-e.grid.tile_size_m*.85,l)for(let f of r.maps.action)for(let g of f)h=Math.min(h,g);this.setFloor(h*this.unitHeight-e.grid.tile_size_m*.85),(!e.metric||a||s||o!==this.unitHeight||!this.obstacleProps)&&(Cm(this.obstacleProps),this.obstacleProps=fm(e,{unitHeight:this.unitHeight,style:this.presentation}),this.world.add(this.obstacleProps),e.metric&&(this.obstacleProps.scale.z=-1)),e.metric&&r?.metric&&!a&&!s&&o===this.unitHeight&&!c.changed.length&&!e.metric.terrain_changes.length&&r.metric.native.every((f,g)=>f===e.metric.native[g])&&r.metric.loose.every((f,g)=>f===e.metric.loose[g])||(Cm(this.piles),this.piles=e.metric?new Ed:new ah(e,{previous:l?r:null,unitHeight:this.unitHeight,layerSettings:this.layerSettings,visibility:this.visibility,palette:this.palette,roughness:this.presentation==="studio"?.16:0}),this.world.add(this.piles),this.piles.update(l?0:1),this.populate(e));let d=new Set(e.agents.map(f=>f.id));for(let[f,g]of this.machines)d.has(f)||(this.scene.remove(g.root),g.dispose(),this.machines.delete(f));for(let f of e.agents){let g=this.machines.get(f.id);g&&(g.agent.type!==f.type||g.agent.action_type!==f.action_type||g.agent.width!==f.width||g.agent.height!==f.height||g.agent.reach.some((x,p)=>x!==f.reach[p]))&&(this.scene.remove(g.root),g.dispose(),this.machines.delete(f.id),g=null),g||(g=tm(f,e.grid.tile_size_m,{style:this.presentation}),g.setTags(this.visibility.tags),this.machines.set(f.id,g),this.scene.add(g.root)),l||(g.lastMove=null),this.poseMachine(g,f,e,1)}if(l){let f=this.actorWork(r,e),g=new Map;for(let m of f.values()){let M=this.machines.get(m.id);(m.kind==="dig"||m.kind==="dump")&&M?.plan&&(m.plan=M.plan({kind:m.kind,from:m.from,to:m.to,cells:m.cells.map(b=>this.workCell(b,r,e))}));for(let[b,y]of m.plan?.timing??[])g.set(b,y);m.events=this.planEvents(m),m.fired=new Set}let x=[...f.values()].some(m=>vS.includes(m.kind)),p=x?i*1.3:i;this.motion={previous:r,frame:e,facts:c,actors:f,timing:g,start:performance.now(),duration:bn(p,100,900)};for(let m of c.changed)this.updateCell(m.row,m.col,r.maps.action[m.row][m.col]);this.dirtyInstances(),this.update(performance.now())}this.selected&&this.highlight(this.selected.row,this.selected.col),(a||s)&&this.home({instant:!0})}populate(e){let{rows:t,cols:i,tile_size_m:s}=e.grid,r=[];this.boundaryEntries.clear(),this.gridEntries=new Map,this.displayHeights=e.maps.action.map(l=>[...l]);let o=this.palette.dug.map(l=>new Te(l)),a=new Te(this.palette.sand),c=new Te(this.palette.loose);for(let l=0;l<t;l++)for(let h=0;h<i;h++){let u=l*i+h,d=e.maps.action[l][h];if(this.updateCell(l,h,d),!e.metric){let f=(l*71+h*29+l*h%47)%31/31;Em.copy(d>0?c:d<0?o[Math.max(0,Math.min(o.length-1,Math.floor(-d)-1))]:a).multiplyScalar(.98+f*.04),this.terrain.setColorAt(u,Em)}this.visibility.grid&&(this.gridEntries.set(u,r.length),r.push(...this.flatGridCell(l,h,d)))}this.dirtyInstances(),e.metric||(this.terrain.instanceColor.needsUpdate=!0,this.terrain.computeBoundingSphere()),this.gridLines.geometry.dispose(),this.gridLines.geometry=new ut,this.gridLines.geometry.setAttribute("position",new it(r,3));for(let[l,h]of Object.entries(this.layers))h.visible=this.visibility[l]&&e.maps[Ws[l].map]!=null;for(let[l,h]of Object.entries(this.boundaries)){let u=[],d=Ws[l].test,f=e.maps[Ws[l].map],g=(x,p)=>f!=null&&x>=0&&x<t&&p>=0&&p<i&&!e.maps.padding[x][p]&&d(f[x][p]);for(let x=0;x<t;x++)for(let p=0;p<i;p++)if(g(x,p)){let m=x*i+p,M=[[x-1,p,[[0,0],[0,1],[0,2]]],[x+1,p,[[2,0],[2,1],[2,2]]],[x,p-1,[[0,0],[1,0],[2,0]]],[x,p+1,[[0,2],[1,2],[2,2]]]];for(let[b,y,T]of M)if(!g(b,y)){this.boundaryEntries.has(m)||this.boundaryEntries.set(m,[]);for(let S of[T[0],T[1],T[1],T[2]]){let A=x*2+S[0],_=p*2+S[1];this.boundaryEntries.get(m).push({name:l,y:u.length+1,row2:A,col2:_}),u.push((_/2-i/2)*s,this.boundaryHeight(x,p,A,_),(A/2-t/2)*s*(e.metric?-1:1))}}}h.geometry.dispose(),h.geometry=new ks,u.length&&h.geometry.setPositions(u),h.visible=this.visibility[l]&&u.length>0}}flatGridCell(e,t,i){let s=this.frame.grid.tile_size_m,r=this.point(e,t,i),o=r.y+s*.022,a=!this.frame.metric&&i>0&&!this.frame.maps.padding[e][t]?0:s/2,c=r.x,l=r.z;return[c-a,o,l-a,c+a,o,l-a,c+a,o,l-a,c+a,o,l+a,c+a,o,l+a,c-a,o,l+a,c-a,o,l+a,c-a,o,l-a]}setFloor(e){this.floor=e,this.environment?.setFloor(e)}boundaryHeight(e,t,i,s){let r=this.displayHeights[e][t];return(!this.frame.metric&&r>0?this.piles.nodeHeight(i,s):r*this.unitHeight)+this.frame.grid.tile_size_m*.035}updateCell(e,t,i){let s=this.frame,r=s.grid.tile_size_m,o=e*s.grid.cols+t,a=!s.metric&&i>0&&!s.maps.padding[e][t],c=this.point(e,t,i);this.displayHeights[e][t]=i;let l=Math.max(r*.02,(a?0:c.y)-this.floor);Gn.rotation.set(0,0,0),s.metric?this.metricTerrainDirty=!0:(Gn.position.set(c.x,this.floor+l/2,c.z),Gn.scale.set(r,l,r),Gn.updateMatrix(),this.terrain.setMatrixAt(o,Gn.matrix));let h=0;for(let[u,d]of Object.entries(this.layers)){let f=s.metric?d.userData.cellSlots[o]:o;if(f<0)continue;let g=Ws[u],x=s.maps[g.map],p=u==="dig"&&this.presentation==="studio"&&x!=null&&i<=x[e][t],m=!a&&!p&&x!=null&&g.test(x[e][t])&&!s.maps.padding[e][t];Gn.position.set(c.x,c.y+r*(.008+h*.003),c.z),Gn.scale.set(m?r:0,1,m?r:0),Gn.updateMatrix(),d.setMatrixAt(f,Gn.matrix),h++}this.gridEntries.has(o)&&this.gridLines.geometry.attributes.position.array.set(this.flatGridCell(e,t,i),this.gridEntries.get(o))}dirtyInstances(){if(this.frame.metric&&this.metricTerrainDirty){let t=vm(this.frame,this.displayHeights,this.unitHeight,this.floor,this.palette);this.terrain.geometry.dispose(),this.terrain.geometry=t,this.metricTerrainDirty=!1}else this.frame.metric||(this.terrain.instanceMatrix.needsUpdate=!0);for(let t of Object.values(this.layers))t.instanceMatrix.needsUpdate=!0;let e=this.frame.grid.cols;for(let[t,i]of this.boundaryEntries)for(let s of i){let r=this.boundaries[s.name].geometry.attributes.instanceStart?.data.array;r&&(r[s.y]=this.boundaryHeight(Math.floor(t/e),t%e,s.row2,s.col2))}for(let t of Object.values(this.boundaries)){let i=t.geometry.attributes.instanceStart?.data;i&&(i.needsUpdate=!0)}this.gridLines.geometry.attributes.position&&(this.gridLines.geometry.attributes.position.needsUpdate=!0)}poseMachine(e,t,i,s,r,o,a="",c=null){let l=r||t,h=a==="turn"&&!i.metric?xS(s):s,u=t.position.map((f,g)=>xh(l.position[g],f,s)),d=this.point(u[0],u[1]);d.y=xh(this.displayHeightAt(...l.position),this.displayHeightAt(...t.position),s),e.root.position.copy(d),e.root.rotation.y=xd(l.base_yaw,t.base_yaw,h),e.setPose({...t,previous_loaded:l.loaded,cabin_yaw:xd(l.cabin_yaw,t.cabin_yaw,h),wheel_angle:xh(l.wheel_angle,t.wheel_angle,s)},t.id===i.current_agent,s,a,c),e.drive?.(e.root.position,e.root.rotation.y)}actorWork(e,t){let i=new Map(e.agents.map(a=>[a.id,a]));if(t.metric){let a=new Map(t.metric.work.map(l=>[l.agent_id,l])),c=t.grid.cols;return new Map(t.agents.map(l=>{let h=i.get(l.id)??l,u=a.get(l.id),d=l.position.some((p,m)=>p!==h.position[m]),f=l.base_yaw!==h.base_yaw||l.cabin_yaw!==h.cabin_yaw||l.wheel_angle!==h.wheel_angle||l.shovel_lifted!==h.shovel_lifted,g=(u?.changed_indices??[]).map(p=>{let m=Math.floor(p/c),M=p%c;return{row:m,col:M,delta:t.maps.action[m][M]-e.maps.action[m][M]}}),x=u?.kind==="collect"?"dig":u?.kind??(d?"move":f?"turn":"");return[l.id,{id:l.id,kind:x,cells:g,load:l.loaded-h.loaded,from:h,to:l,swing:x==="turn"&&l.cabin_yaw!==h.cabin_yaw,recipient:t.agents.find(p=>p.id===u?.recipient_id)??null}]}))}let s=new Map(t.agents.map(a=>[a.id,a.loaded-(i.get(a.id)?.loaded??a.loaded)])),r=new Map(t.agents.map(a=>[a.id,[]]));for(let a of pa(e,t).changed){let c=t.agents.filter(u=>a.delta<0?s.get(u.id)>0:s.get(u.id)<0),l=null,h=1/0;for(let u of c.length?c:t.agents){let d=Math.hypot(u.position[0]-a.row,u.position[1]-a.col);d<h&&(h=d,l=u)}r.get(l.id).push(a)}let o=new Map;for(let a of t.agents){let c=i.get(a.id);if(!c)continue;let l=r.get(a.id),h=s.get(a.id),u=t.agents.find(p=>p.id!==a.id&&s.get(p.id)>0&&!r.get(p.id).some(m=>m.delta<0)),d=a.position.some((p,m)=>p!==c.position[m]),f=a.cabin_yaw!==c.cabin_yaw,g=f||a.base_yaw!==c.base_yaw||a.wheel_angle!==c.wheel_angle||a.shovel_lifted!==c.shovel_lifted,x="";l.some(p=>p.delta<0)&&h>0?x="dig":l.some(p=>p.delta>0)&&h<0?x="dump":h<0&&u?x="transfer":l.length?x="terrain":d?x="move":g&&(x="turn"),o.set(a.id,{id:a.id,kind:x,cells:l,load:h,from:c,to:a,swing:x==="turn"&&f&&a.base_yaw===c.base_yaw,recipient:x==="transfer"?u:null})}return o}workCell(e,t,i){let s=this.point(e.row,e.col),r=i.maps.padding[e.row][e.col],o=(a,c)=>!i.metric&&a>0&&!r?this.piles.endpointHeight(e.row,e.col,c):a*this.unitHeight;return{key:e.row*i.grid.cols+e.col,row:e.row,col:e.col,delta:e.delta,x:s.x,z:s.z,before:o(t.maps.action[e.row][e.col],!0),after:o(i.maps.action[e.row][e.col],!1)}}planEvents(e){let t=[],{kind:i,plan:s,from:r,to:o}=e;if(!i||i==="terrain")return t;let a=o.position.some((l,h)=>l!==r.position[h]);t.push({at:0,once:"exhaust-start"}),a&&t.push({from:.05,to:.9,stream:"tracks",rate:16});let c=l=>e.cells.filter(h=>l<0?h.delta<0:h.delta>0);if(i==="dig"){let l=s?.events.bite??(o.type===2?.38:.32),[h,u]=s?.events.drag??[l,l+.22];t.push({at:l,once:"bite",cells:c(-1)}),t.push({from:h,to:u,stream:"scoop",cells:c(-1),rate:s?34:70}),s?.events.breakout&&t.push({at:s.events.breakout,once:"spill"})}else if(i==="dump"){let[l,h]=s?.events.pour??(o.type===1?[.32,.72]:o.type===2?[.34,.62]:[.44,.74]);t.push({from:l,to:h,stream:o.type===1?"bed":"pour",cells:c(1),rate:s?230:60}),t.push({at:Math.min(.95,(l+h)/2+.1),once:"landing",cells:c(1)})}else i==="transfer"&&t.push({from:.44,to:.72,stream:"transfer",rate:55});return t}centroid(e){let t=new P;if(!e?.length)return null;for(let i of e)t.add(this.surfacePoint(i.row,i.col));return t.multiplyScalar(1/e.length)}runEvents(e,t,i){let s=e.frame.grid.tile_size_m,r=this.effects,o=this.presentation==="studio";for(let a of e.actors.values()){let c=this.machines.get(a.id);if(!(!c||!a.events.length)){c.root.updateMatrixWorld(!0);for(let[l,h]of a.events.entries())if(h.once){if(a.fired.has(l)||t<h.at)continue;if(a.fired.add(l),h.once==="exhaust-start")o||r.puff(c.exhaust(),{count:4,size:s*.28,rise:1.4,spread:s*.15,color:6185835,life:1.1});else if(h.once==="bite"){let u=o?c.teeth():this.centroid(h.cells)??c.tip();r.burst(u,{count:o?9:14,speed:o?1.6:2.4,size:s*(o?.06:.09)}),r.puff(u,{count:7,size:s*.38,spread:s*.6,rise:.5}),r.dust(u,{count:6,size:s*1.1,spread:s*.5,life:1.6})}else if(h.once==="landing"){let u=this.centroid(h.cells);u&&(r.puff(u,{count:8,size:s*.42,spread:s*.7,rise:.4}),r.dust(u,{count:9,size:s*1.5,spread:s*.8,life:2}))}else h.once==="spill"&&r.throwClods(c.lip(),c.lip().setY(this.groundAt(c.lip().x,c.lip().z)),{count:5,flight:.4,spread:s*.3,size:s*.06})}else if(t>=h.from&&t<=h.to){h.carry=(h.carry??0)+h.rate*i;let u=Math.floor(h.carry);for(h.carry-=u;u-- >0;)this.emitStream(h,c,e,s)}}}}emitStream(e,t,i,s){let r=this.effects,o=this.presentation==="studio",a=c=>c?.length?this.surfacePoint(...Object.values(c[Math.floor(Math.random()*c.length)]).slice(0,2)):null;if(e.stream==="tracks"){let c=i.frame.agents.find(h=>h.id===t.agent.id),l=new P(-t.agent.height*s*.45,0,(Math.random()<.5?-1:1)*t.agent.width*s*.35).applyAxisAngle(new P(0,1,0),t.root.rotation.y).add(t.root.position);c&&Math.random()<.5&&(r.puff(l,{count:1,size:s*.3,spread:s*.2,rise:.35,life:.8}),Math.random()<.4&&r.dust(l,{count:1,size:s*.9,spread:s*.3,life:1.3,opacity:.16})),Math.random()<.25&&r.puff(t.exhaust(),{count:1,size:s*.2,rise:1.3,spread:s*.08,color:6975351,life:1})}else if(e.stream==="scoop")if(o){let c=t.teeth(),l=c.clone().add(new P((Math.random()-.5)*s*1.2,0,(Math.random()-.5)*s*1.2));l.y=this.groundAt(l.x,l.z),r.throwClods(c.clone().add(new P(0,s*.12,0)),l,{count:1,flight:.28,spread:s*.15,size:s*.055,jitter:s*.2}),Math.random()<.12&&r.dust(c,{count:1,size:s*.8,spread:s*.3,life:1.2,opacity:.14})}else{let c=a(e.cells);c&&r.throwClods(c,t.tip(),{count:1,flight:.22,spread:0,size:s*.08,settle:!1,jitter:s*.3})}else if(e.stream==="pour"||e.stream==="bed"){let c=a(e.cells);if(!c)return;let l=e.stream==="bed"?t.bedLip():o?t.lip():t.tip();r.throwClods(l,c,{count:1,flight:o?.42:.34,spread:s*.45,size:s*(o?.06:.095),jitter:s*(o?.22:.12)}),o&&Math.random()<.06&&r.dust(l,{count:1,size:s*.9,spread:s*.3,life:1.4,opacity:.12})}else if(e.stream==="transfer"){let c=this.machines.get(i.actors.get(t.agent.id)?.recipient?.id);if(!c)return;c.root.updateMatrixWorld(!0),r.throwClods(t.tip(),c.tip(),{count:1,flight:.3,spread:s*.2,size:s*.09,settle:!1,jitter:s*.1})}}update(e){let t=Math.min(.1,Math.max(0,(e-(this.clock?.last??e))/1e3));if(this.clock&&(this.clock.last=e),sn.uTime.value=e/1e3,!this.frame)return;this.measure(t),this.environment?.update(e/1e3);let i=new Map;if(this.motion){let{previous:r,frame:o,facts:a,start:c,duration:l,actors:h,timing:u}=this.motion,d=bn((e-c)/l,0,1),f=wm(d),g=o.grid.cols,x=(p,m)=>{let M=u.get(p*g+m);return M?wm(bn((d-M[0])/(M[1]-M[0]),0,1)):f};a.changed.length&&this.piles.update(u.size?x:f);for(let p of a.changed)this.updateCell(p.row,p.col,xh(r.maps.action[p.row][p.col],o.maps.action[p.row][p.col],x(p.row,p.col)));for(let p of o.agents){let m=r.agents.find(y=>y.id===p.id),M=h.get(p.id),b=M?.kind==="terrain"?"":M?.kind??"";if(!b&&[...h.values()].some(y=>y.recipient?.id===p.id)&&(b="receive"),(b==="turn"||b==="move")&&(b=m&&p.position.some((y,T)=>y!==m.position[T])?"move":"turn"),this.poseMachine(this.machines.get(p.id),p,o,M?.plan||o.metric?.event.phase==="drive"?d:f,m,r,b,M?.plan),m&&p.position.some((y,T)=>y!==m.position[T])){let y=new $(Math.cos(p.base_yaw),Math.sin(p.base_yaw)),T=new $(p.position[1]-m.position[1],(p.position[0]-m.position[0])*(o.metric?-1:1));i.set(p.id,{move:d,direction:Math.sign(y.x*T.x-y.y*T.y)||1})}}a.changed.length&&(this.dirtyInstances(),this.selected&&this.highlight(this.selected.row,this.selected.col)),this.reducedMotion||this.runEvents(this.motion,d,t),d>=1&&(this.clearMotion(),this.floor!==this.finalFloor&&(this.setFloor(this.finalFloor),this.populate(this.frame)))}let s=e/1e3;for(let r of this.machines.values())r.tick?.(s,{...i.get(r.agent.id)||{},reducedMotion:this.reducedMotion});if(this.clock.idle-=t,!this.reducedMotion&&this.clock.idle<=0){this.clock.idle=1.3+Math.random()*.8;let r=this.machines.get(this.frame.current_agent),o=this.frame.grid.tile_size_m;r&&!this.frame.done&&(r.root.updateMatrixWorld(!0),this.effects.puff(r.exhaust(),{count:1,size:o*.18,rise:1.1,spread:o*.05,color:7764867,life:1.2}))}if(this.effects.update(t),this.tween){let{from:r,to:o,start:a,duration:c}=this.tween,l=_S(bn((e-a)/c,0,1));this.camera.position.lerpVectors(r.position,o.position,l),this.controls.target.lerpVectors(r.target,o.target,l),l>=1&&(this.tween=null)}if(this.follow){let r=this.machines.get(this.frame.current_agent);if(r){let o=r.root.position.clone();o.y+=this.frame.grid.tile_size_m;let a=o.sub(this.controls.target).multiplyScalar(.055);this.camera.position.add(a),this.controls.target.add(a)}}}clearMotion(){if(this.motion)for(let e of this.motion.frame.agents){let t=this.machines.get(e.id);t&&t.tick?.(performance.now()/1e3,{reducedMotion:this.reducedMotion})}this.motion=null}setLayer(e,t){if(this.visibility[e]=t,e==="tags"){for(let i of this.machines.values())i.setTags(t);return}this.frame&&(this.piles?.setLayer(e,t),e==="grid"?(this.gridLines.visible=t,this.populate(this.frame)):this.layers[e]&&(this.layers[e].visible=t&&this.frame.maps[Ws[e].map]!=null),this.boundaries[e]&&(this.boundaries[e].visible=t))}setHeight(e){this.heightScale=e,this.frame&&this.setFrame(this.frame)}flyTo(e,t,{instant:i=!1}={}){if(i||this.reducedMotion){this.tween=null,this.camera.position.copy(e),this.controls.target.copy(t),this.controls.update();return}this.tween={from:{position:this.camera.position.clone(),target:this.controls.target.clone()},to:{position:e,target:t},start:performance.now(),duration:750}}home({instant:e=!1}={}){if(!this.frame)return;this.follow=!1;let t=this.camera.aspect,i=this.presentation==="diorama"?1.75:1.45,s=this.span*(t<1?i/t:i);this.flyTo(new P(s*.72,s*.66,s*.84),new P(0,-this.span*.04,0),{instant:e}),this.onCameraChange?.({view:"home",follow:!1})}top(){this.frame&&(this.follow=!1,this.camera.up.set(0,1,0),this.flyTo(new P(0,this.span*(this.presentation==="diorama"?1.92:1.62)/Math.min(this.camera.aspect,1),this.span*.001),new P(0,0,0)),this.onCameraChange?.({view:"top",follow:!1}))}setFollow(e){if(this.follow=e,this.tween=null,e&&this.frame){let t=this.machines.get(this.frame.current_agent);if(t){let i=this.frame.agents.find(u=>u.id===this.frame.current_agent),s=this.frame.grid.tile_size_m,r=t.root.position.clone();r.y+=s;let o=Vt.degToRad(this.camera.fov),a=2*Math.atan(Math.tan(o/2)*this.camera.aspect),c=Math.max(i.reach[1],Math.hypot(i.width,i.height)*.65)*s,l=Math.max(this.span*.5,c*1.15/Math.sin(Math.min(o,a)/2)),h=this.camera.position.clone().sub(this.controls.target).normalize().multiplyScalar(l);this.flyTo(r.clone().add(h),r)}}this.onCameraChange?.({follow:e})}pick(e){if(!this.terrain)return;let t=this.renderer.domElement.getBoundingClientRect();this.pointer.set((e.clientX-t.left)/t.width*2-1,-(e.clientY-t.top)/t.height*2+1),this.camera.updateMatrixWorld(),this.world.updateMatrixWorld(!0),this.raycaster.setFromCamera(this.pointer,this.camera);let i=[this.terrain,this.obstacleProps];this.piles.surface?.visible&&i.push(this.piles.surface);let s=this.raycaster.intersectObjects(i.filter(Boolean),!0)[0],r;if(s&&s.object===this.piles.surface)r=this.piles.cellForHit(s);else if(s?.object===this.terrain&&s.instanceId!==void 0)r={row:Math.floor(s.instanceId/this.frame.grid.cols),col:s.instanceId%this.frame.grid.cols};else if(s){let{rows:o,cols:a,tile_size_m:c}=this.frame.grid;r={row:bn(Math.floor((this.frame.metric?-s.point.z:s.point.z)/c+o/2),0,o-1),col:bn(Math.floor(s.point.x/c+a/2),0,a-1)}}r&&(this.highlight(r.row,r.col),this.onPick?.(r))}highlight(e,t){if(e>=this.frame.grid.rows||t>=this.frame.grid.cols){this.selected=null,this.selection.visible=!1;return}this.selected={row:e,col:t},this.selection.position.copy(this.surfacePoint(e,t)),this.selection.position.y+=this.frame.grid.tile_size_m*.03,this.selection.visible=!0}capture({scale:e=2}={}){let t=this.element.clientWidth||1,i=this.element.clientHeight||1,s=Math.min(Math.max(e,this.pixelRatio),this.renderer.capabilities.maxTextureSize/Math.max(t,i));this.renderer.setPixelRatio(s),this.renderer.setSize(t,i,!1),this.post?.setSamples(0),this.post?.setSize(t,i,s);let r=this.renderer.getDrawingBufferSize(new $);for(let o of this.lineMaterials)o.resolution.copy(r);try{return this.render(),{url:this.renderer.domElement.toDataURL("image/png"),width:r.x,height:r.y}}finally{this.renderer.setPixelRatio(this.pixelRatio),this.post?.setSamples(4),this.resize()}}};/*!
 * Three.js 0.185.1 — The MIT License
 * Copyright © 2010-2026 three.js authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */var De=n=>document.getElementById(n),Xs=n=>new Intl.NumberFormat(void 0,{maximumFractionDigits:2}).format(n),wt=(n,e)=>{De(n).textContent=e},Tt,ct,Rt=0,Xn="replay",Bm=null,Ms=!1,Pt=!1,En=!1,Sa=null,wd,yh=0,zm=De("terra-replay"),km=!!zm?.textContent.trim();function qs(n){De("loading").hidden=!0,wt("error-message",n?.message||String(n)),De("error").hidden=!1}function wn(n,e=3100){clearTimeout(wd),wt("event",n),De("event").classList.add("visible"),wd=setTimeout(()=>De("event").classList.remove("visible"),e)}function Yn(){return ct?.frames[Rt]}function Hm(){return!!ct&&Xn==="manual"&&!Ms&&!Pt&&!En&&Rt===ct.frames.length-1&&!Yn().done}function rn(){let n=Hm(),e=Yn();document.querySelectorAll("[data-action]").forEach(i=>{i.disabled=!n}),De("reset").disabled=Pt||Xn!=="manual"||Ms,De("export").disabled=!ct||Pt,De("screenshot").disabled=!Tt||!ct,De("open-file").disabled=Pt,De("previous").disabled=!ct||Rt<=0||Pt,De("next").disabled=!ct||Rt>=ct.frames.length-1||Pt,De("play").disabled=!ct||ct.frames.length<2||Pt,De("seek").disabled=!ct||ct.frames.length<2||Pt,De("play").textContent=En?"\u2161":"\u25B6",De("play").setAttribute("aria-label",En?"Pause replay":"Play replay"),De("resume-live").hidden=!Bm||km||!Ms&&!(Xn==="manual"&&ct&&Rt<ct.frames.length-1),De("resume-live").disabled=Pt;let t=ct&&Rt<ct.frames.length-1;wt("manual-status",Pt?"STEPPING\u2026":Ms||Xn!=="manual"?"REPLAY ONLY":t?"HISTORY":e?.done?"ENDED":En?"PLAYBACK":"LIVE"),wt("session-mode",Ms?"Imported replay":Xn==="manual"?t?"Manual \xB7 history":"Manual session":"Replay session")}function Rd(){if(!Sa||!Yn())return;let{row:n,col:e}=Sa,t=Yn(),{maps:i}=t;if(n>=t.grid.rows||e>=t.grid.cols){Sa=null;return}let s=De("cell-inspector");s.replaceChildren();let r=document.createElement("span");r.className="eyebrow",r.textContent="CELL INSPECTOR",s.append(r);let o=document.createElement("div");o.className="cell-heading",o.textContent=`ROW ${n}  \xB7  COL ${e}`,s.append(o);let a=document.createElement("div");a.className="cell-data",s.append(a);let c=(u,d,f)=>i[u]==null?"Unavailable":i[u][n][e]?d:f,l=i.target[n][e],h=[["Raw soil height",`${i.action[n][e]} units`],["Target",l<0?`Dig ${-l}`:l>0?`Dump ${l}`:"Neutral"],["Obstacle",i.padding[n][e]?"Yes":"No"],["Static dumping",c("dumpability_static","Allowed","Prohibited")],["Dumpable now",c("dumpability","Yes","No")],["Workspace",c("interaction","Inside","Outside")],["Traversability feature",i.traversability==null?"Unavailable":{"-1":"Occupied (\u22121)",0:"Clear (0)",1:"Blocked (1)"}[Number(i.traversability[n][e])]]];for(let[u,d]of h){let f=document.createElement("span"),g=document.createElement("strong");f.textContent=u,g.textContent=d,a.append(f,g)}}function SS(){let n=Yn(),e=n.agents.find(a=>a.id===n.current_agent),t=rm(n);wt("title",ct.metadata.title),wt("source",ct.metadata.source),wt("grid-spec",`${n.grid.rows} \xD7 ${n.grid.cols} \xB7 ${Xs(n.grid.tile_size_m)} m / cell`),wt("agent-count",`${n.agents.length} machine${n.agents.length===1?"":"s"}`),wt("agent-id",String(e.id+1).padStart(2,"0")),wt("machine-name",md[e.type]),wt("embodiment",e.action_type===1?"Wheeled":"Tracked"),De("load").replaceChildren(document.createTextNode(Xs(e.loaded)));let i=document.createElement("small");i.textContent=" units",De("load").append(i),wt("reward",om(n.reward)),wt("outcome",n.task_done?"Task complete":n.done?"Episode ended \xB7 task incomplete":`Ready \xB7 machine ${e.id+1} acts next`),De("outcome").classList.toggle("done",n.done);let s=De("agent-list");s.replaceChildren(),s.hidden=n.agents.length<=1;for(let a of n.agents){let c=document.createElement("span");c.className=`agent-tag${a.id===n.current_agent?" active":""}`,c.textContent=`${String(a.id+1).padStart(2,"0")} ${md[a.type]} \xB7 ${a.loaded}`,s.append(c)}wt("cut-units",`${Xs(t.cut)} units`),wt("fill-units",`${Xs(t.fill)} units`),wt("scene-caption",`${Xs(n.grid.cols*n.grid.tile_size_m)} \xD7 ${Xs(n.grid.rows*n.grid.tile_size_m)} m worksite \xB7 illustrative soil mounds`),wt("step",n.step),wt("frame-count",`${Rt+1} / ${ct.frames.length}`),wt("action-label",_d(n,ct.frames[Rt-1])),De("seek").max=String(ct.frames.length-1),De("seek").value=String(Rt),De("seek").setAttribute("aria-valuetext",`Snapshot ${Rt+1} of ${ct.frames.length}, step ${n.step}`);let r=De("left-action"),o=De("right-action");r.dataset.action=e.action_type===1?"2":"3",o.dataset.action=e.action_type===1?"3":"2",r.querySelector(".turn-label").textContent=e.action_type===1?"Steer left":"Turn left",o.querySelector(".turn-label").textContent=e.action_type===1?"Steer right":"Turn right",r.title=`${e.action_type===1?"Steer left":"Turn anticlockwise"} \xB7 Left or A`,o.title=`${e.action_type===1?"Steer right":"Turn clockwise"} \xB7 Right or D`,wt("work-label",e.type===2?e.shovel_lifted?"Lower shovel / dump":"Lift shovel":e.loaded>0?"Dump / transfer soil":e.type===1?"Dump (empty)":"Dig soil");for(let[a,c]of[["interaction","interaction"],["restricted","dumpability_static"],["dumpability","dumpability"]]){let l=document.querySelector(`[data-layer="${a}"]`);l.disabled=n.maps[c]==null,l.closest("label").title=n.maps[c]==null?"This diagnostic layer is unavailable in the recording.":""}Rd(),rn()}function qn(n,{animate:e=!1,reset:t=!1,announce:i=!1}={}){if(!ct)return;let s=Rt;Rt=Math.max(0,Math.min(ct.frames.length-1,n));let r=Yn();if(Tt.setFrame(r,{animate:e&&Rt===s+1,reset:t,duration:Math.min(650,800/Number(De("speed").value))}),SS(),i&&Rt>0){let o=ct.frames[Rt-1];r.step>o.step&&!o.done?wn(`${_d(r,o)} \xB7 ${pa(o,r).message}`):wn("Episode boundary \xB7 initial snapshot")}else t&&(clearTimeout(wd),De("event").classList.remove("visible"))}function Wn(n){En=n,yh=performance.now(),rn()}function Dm(){!ct||ct.frames.length<2||Pt||(!En&&Rt===ct.frames.length-1&&qn(0),Wn(!En))}function Vm(n){En&&!document.hidden&&n-yh>=900/Number(De("speed").value)&&(yh=n,Rt<ct.frames.length-1&&qn(Rt+1,{animate:!0,announce:!0}),Rt>=ct.frames.length-1&&Wn(!1)),requestAnimationFrame(Vm)}async function Mh(n,e){let t=await fetch(n,e===void 0?{cache:"no-store"}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(e)}),i=await t.json().catch(()=>{throw new Error(`The server returned an unreadable response (${t.status}).`)});if(!t.ok)throw new Error(i.error||`Request failed (${t.status}).`);return i}function ba(n,{local:e=!1}={}){if(sm(n.replay),!["manual","replay"].includes(n.mode))throw new Error("Unknown viewer session mode.");ct=n.replay,Xn=n.mode,Ms=e,Rt=0,Sa=null,En=!1,e||(Bm=n.mode),De("cell-inspector").replaceChildren();let t=document.createElement("span");t.className="eyebrow",t.textContent="CELL INSPECTOR";let i=document.createElement("span");i.className="cell-hint",i.textContent="Click the terrain to inspect a cell",De("cell-inspector").append(t,i),qn(Xn==="manual"&&!e?ct.frames.length-1:0,{reset:!0}),De("loading").hidden=!0,De("error").hidden=!0}async function Lm(n){if(Hm()){Pt=!0,rn();try{let{frame:e}=await Mh("/api/action",{action:n});gd(e),ct.frames.push(e),qn(ct.frames.length-1,{animate:!0,announce:!0})}catch(e){qs(e)}finally{Pt=!1,rn()}}}async function Um(){if(!(Pt||Xn!=="manual"||Ms)){Pt=!0,Wn(!1),rn();try{ba(await Mh("/api/reset",{})),wn("Episode reset \xB7 ready to play")}catch(n){qs(n)}finally{Pt=!1,rn()}}}function Gm(n,e,t=!1){let i=document.createElement("a");i.href=n,i.download=e,document.body.append(i),i.click(),i.remove(),t&&setTimeout(()=>URL.revokeObjectURL(n),1e3)}function bS(){if(!ct)return;let n=new Blob([JSON.stringify(ct)],{type:"application/json"});Gm(URL.createObjectURL(n),"terra-replay.json",!0),wn(`Exported ${ct.frames.length} recorded snapshots`)}function ES(){document.querySelectorAll("[data-action]").forEach(n=>n.addEventListener("click",e=>{Lm(Number(n.dataset.action)),e.detail>0&&De("viewport").focus({preventScroll:!0})})),De("reset").addEventListener("click",n=>{Um(),n.detail>0&&De("viewport").focus({preventScroll:!0})}),De("previous").addEventListener("click",()=>{Wn(!1),qn(Rt-1)}),De("next").addEventListener("click",()=>{Wn(!1),qn(Rt+1,{animate:!0,announce:!0})}),De("play").addEventListener("click",Dm),De("seek").addEventListener("input",()=>{Wn(!1),qn(Number(De("seek").value))}),De("speed").addEventListener("change",()=>{yh=performance.now()}),De("camera-home").addEventListener("click",()=>Tt?.home()),De("brand-home").addEventListener("click",n=>{n.preventDefault(),Tt?.home()}),De("camera-top").addEventListener("click",()=>Tt?.top()),De("camera-follow").addEventListener("click",()=>Tt?.setFollow(!Tt.follow)),De("quality").addEventListener("click",Fm),De("presentation").addEventListener("click",Nm),De("height-scale").addEventListener("input",()=>{let n=Number(De("height-scale").value);wt("height-value",`${Xs(n)}\xD7`),Tt?.setHeight(n)}),document.querySelectorAll("[data-layer]").forEach(n=>n.addEventListener("change",()=>{Tt?.setLayer(n.dataset.layer,n.checked),Om()})),De("layers-toggle").addEventListener("click",()=>{let n=[...document.querySelectorAll("[data-layer]")].filter(t=>!t.disabled),e=!n.some(t=>t.checked);for(let t of n)t.checked=e,Tt?.setLayer(t.dataset.layer,e);Om()}),De("export").addEventListener("click",bS),De("screenshot").addEventListener("click",()=>{try{let n=Tt.capture();Gm(n.url,`terra-step-${Yn().step}.png`),wn(`Scene captured \xB7 ${n.width} \xD7 ${n.height} PNG`)}catch(n){qs(n)}}),De("open-file").addEventListener("click",()=>De("replay-file").click()),De("replay-file").addEventListener("change",async n=>{let e=n.target.files[0];if(e){if(Pt){n.target.value="",wn("Wait for the current action before opening a replay.");return}Pt=!0,Wn(!1),rn();try{if(e.size>256*1024*1024)throw new Error("Please use a JSON recording smaller than 256 MB. Large recordings can be opened through Python with --replay.");let t=JSON.parse(await e.text());ba({mode:"replay",replay:t},{local:!0}),wn(`Opened ${e.name}`)}catch(t){qs(t)}finally{n.target.value="",Pt=!1,rn()}}}),De("resume-live").addEventListener("click",async()=>{if(!Pt){Pt=!0,Wn(!1),rn();try{ba(await Mh("/api/session"))}catch(n){qs(n)}finally{Pt=!1,rn()}}}),De("dismiss-error").addEventListener("click",()=>{De("error").hidden=!0}),document.addEventListener("keydown",n=>{if(n.ctrlKey||n.metaKey||n.altKey||n.repeat||["INPUT","SELECT","TEXTAREA","BUTTON"].includes(n.target.tagName)||n.target.isContentEditable||!De("error").hidden)return;let e=n.key.toLowerCase();if(e==="g"){n.preventDefault(),Fm();return}if(e==="p"){n.preventDefault(),Nm();return}if(e==="h"){n.preventDefault(),Tt?.home();return}if(e==="t"){n.preventDefault(),Tt?.top();return}if(e==="f"){n.preventDefault(),Tt?.setFollow(!Tt.follow);return}if(!ct||Pt)return;if(Xn!=="manual"||Ms||Rt<ct.frames.length-1||En){e===" "&&(n.preventDefault(),Dm()),(e==="arrowleft"||e==="arrowright")&&(n.preventDefault(),Wn(!1),qn(Rt+(e==="arrowright"?1:-1)));return}if(e==="r"){n.preventDefault(),Um();return}let t=Yn().agents.find(o=>o.id===Yn().current_agent),i=t.action_type===1?2:3,s=t.action_type===1?3:2,r={arrowup:0,w:0,arrowdown:1,s:1,arrowleft:i,a:i,arrowright:s,d:s,q:5,e:4," ":6,n:7};e in r&&(n.preventDefault(),Lm(r[e]))})}function Td(n){De("quality").setAttribute("aria-pressed",String(n==="high"))}var Ad={studio:["Studio","Studio style \xB7 earth block on a studio floor"],paper:["Paper","Paper style \xB7 plain figure look"],diorama:["Diorama","Diorama style \xB7 stylized island"]};function Wm(n){De("presentation").querySelector("span").textContent=Ad[n][0]}function Nm(){if(!Tt)return;let n=Object.keys(Ad),e=Tt.setPresentation(n[(n.indexOf(Tt.presentation)+1)%n.length]);Wm(e),Rd(),wn(Ad[e][1])}function Fm(){if(!Tt)return;let n=Tt.setQuality(Tt.quality==="high"?"fast":"high");Td(n),wn(n==="high"?"Rich lighting on \xB7 ambient occlusion and outlines":"Fast graphics \xB7 plain lighting")}function Om(){wt("layers-toggle",[...document.querySelectorAll("[data-layer]")].some(n=>n.checked&&!n.disabled)?"Hide all":"Show all")}async function wS(){ES(),rn();try{Tt=new vh(De("viewport"),{onPick:n=>{Sa=n,Rd()},onCameraChange:({view:n,follow:e})=>{n&&(De("camera-home").classList.toggle("selected",n==="home"),De("camera-top").classList.toggle("selected",n==="top")),e!==void 0&&De("camera-follow").setAttribute("aria-pressed",String(e))},onError:qs,onQualityChange:n=>{Td(n),wn("Switched to fast graphics for smoother motion \xB7 press G to restore")}}),Td(Tt.quality),Wm(Tt.presentation),window.terraViewer={scene:Tt,show:(n,e)=>qn(n,e)},km?ba({mode:"replay",replay:JSON.parse(zm.textContent)},{local:!0}):ba(await Mh("/api/session")),requestAnimationFrame(Vm)}catch(n){qs(n),wt("session-mode","Unavailable"),wt("title","Open a Terra worksite"),wt("source","Check the error message to continue.")}}wS();})();
