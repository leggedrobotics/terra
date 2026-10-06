(()=>{/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */var ls={LEFT:0,MIDDLE:1,RIGHT:2,ROTATE:0,DOLLY:1,PAN:2},cs={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},wf=0,pu=1,Tf=2;var Us=1,Af=2,Ir=3,Jn=0,tn=1,Sn=2,zt=0,Ps=1,mu=2,gu=3,_u=4,$l=5;var Pn=100,Rf=101,Cf=102,Pf=103,If=104,Ns=200,Df=201,Lf=202,Uf=203,ol=204,ll=205,qa=206,Nf=207,Ya=208,Ff=209,Of=210,Bf=211,zf=212,kf=213,Hf=214,cl=0,hl=1,ul=2,Is=3,dl=4,fl=5,pl=6,ml=7,Jl=0,Vf=1,Gf=2,ti=0,Za=1,$a=2,Ja=3,hs=4,ja=5,Ka=6,Fs=7;var xu=300,us=301,Os=302,jl=303,Kl=304,Qa=306,kn=1e3,ui=1001,gl=1002,Ot=1003,Wf=1004;var eo=1005;var en=1006,Ql=1007;var ds=1008;var hn=1009,vu=1010,yu=1011,Dr=1012,ec=1013,ni=1014,Vn=1015,nn=1016,tc=1017,nc=1018,fs=1020,Mu=35902,Su=35899,bu=1021,Eu=1022,bn=1023,fi=1026,_i=1027,ic=1028,sc=1029,ps=1030,rc=1031;var ac=1033,to=33776,no=33777,io=33778,so=33779,oc=35840,lc=35841,cc=35842,hc=35843,uc=36196,dc=37492,fc=37496,pc=37488,mc=37489,ro=37490,gc=37491,_c=37808,xc=37809,vc=37810,yc=37811,Mc=37812,Sc=37813,bc=37814,Ec=37815,wc=37816,Tc=37817,Ac=37818,Rc=37819,Cc=37820,Pc=37821,Ic=36492,Dc=36494,Lc=36495,Uc=36283,Nc=36284,ao=36285,Fc=36286;var aa=2300,_l=2301,al=2302,Qh=2303,eu=2400,tu=2401,nu=2402;var Xf=3200;var Lr=0,qf=1,Oi="",Lt="srgb",oa="srgb-linear",la="linear",pt="srgb";var As=7680;var iu=519,Yf=512,Zf=513,$f=514,Oc=515,Jf=516,jf=517,Bc=518,Kf=519,xl=35044,xi=35048;var wu="300 es",$n=2e3,gr=2001;function Jm(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function jm(i){return ArrayBuffer.isView(i)&&!(i instanceof DataView)}function ca(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function Qf(){let i=ca("canvas");return i.style.display="block",i}var Od={},_r=null;function ha(...i){let e="THREE."+i.shift();_r?_r("log",e,...i):console.log(e,...i)}function ep(i){let e=i[0];if(typeof e=="string"&&e.startsWith("TSL:")){let t=i[1];t&&t.isStackTrace?i[0]+=" "+t.getLocation():i[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return i}function Ze(...i){i=ep(i);let e="THREE."+i.shift();if(_r)_r("warn",e,...i);else{let t=i[0];t&&t.isStackTrace?console.warn(t.getError(e)):console.warn(e,...i)}}function $e(...i){i=ep(i);let e="THREE."+i.shift();if(_r)_r("error",e,...i);else{let t=i[0];t&&t.isStackTrace?console.error(t.getError(e)):console.error(e,...i)}}function Cs(...i){let e=i.join(" ");e in Od||(Od[e]=!0,Ze(...i))}function tp(i,e,t){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:n()}}setTimeout(r,t)})}var np={[cl]:hl,[ul]:pl,[dl]:ml,[Is]:fl,[hl]:cl,[pl]:ul,[ml]:dl,[fl]:Is},jn=class{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});let n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){let n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){let n=this._listeners;if(n===void 0)return;let s=n[e];if(s!==void 0){let r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){let t=this._listeners;if(t===void 0)return;let n=t[e.type];if(n!==void 0){e.target=this;let s=n.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,e);e.target=null}}},ln=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],Bd=1234567,pr=Math.PI/180,xr=180/Math.PI;function di(){let i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(ln[i&255]+ln[i>>8&255]+ln[i>>16&255]+ln[i>>24&255]+"-"+ln[e&255]+ln[e>>8&255]+"-"+ln[e>>16&15|64]+ln[e>>24&255]+"-"+ln[t&63|128]+ln[t>>8&255]+"-"+ln[t>>16&255]+ln[t>>24&255]+ln[n&255]+ln[n>>8&255]+ln[n>>16&255]+ln[n>>24&255]).toLowerCase()}function je(i,e,t){return Math.max(e,Math.min(t,i))}function Tu(i,e){return(i%e+e)%e}function Km(i,e,t,n,s){return n+(i-e)*(s-n)/(t-e)}function Qm(i,e,t){return i!==e?(t-i)/(e-i):0}function ia(i,e,t){return(1-t)*i+t*e}function eg(i,e,t,n){return ia(i,e,1-Math.exp(-t*n))}function tg(i,e=1){return e-Math.abs(Tu(i,e*2)-e)}function ng(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*(3-2*i))}function ig(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*i*(i*(i*6-15)+10))}function sg(i,e){return i+Math.floor(Math.random()*(e-i+1))}function rg(i,e){return i+Math.random()*(e-i)}function ag(i){return i*(.5-Math.random())}function og(i){i!==void 0&&(Bd=i);let e=Bd+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function lg(i){return i*pr}function cg(i){return i*xr}function hg(i){return(i&i-1)===0&&i!==0}function ug(i){return Math.pow(2,Math.ceil(Math.log(i)/Math.LN2))}function dg(i){return Math.pow(2,Math.floor(Math.log(i)/Math.LN2))}function fg(i,e,t,n,s){let r=Math.cos,a=Math.sin,o=r(t/2),c=a(t/2),l=r((e+n)/2),h=a((e+n)/2),d=r((e-n)/2),u=a((e-n)/2),f=r((n-e)/2),g=a((n-e)/2);switch(s){case"XYX":i.set(o*h,c*d,c*u,o*l);break;case"YZY":i.set(c*u,o*h,c*d,o*l);break;case"ZXZ":i.set(c*d,c*u,o*h,o*l);break;case"XZX":i.set(o*h,c*g,c*f,o*l);break;case"YXY":i.set(c*f,o*h,c*g,o*l);break;case"ZYZ":i.set(c*g,c*f,o*h,o*l);break;default:Ze("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function Zn(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("THREE.MathUtils: Invalid component type.")}}function gt(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("THREE.MathUtils: Invalid component type.")}}var Vt={DEG2RAD:pr,RAD2DEG:xr,generateUUID:di,clamp:je,euclideanModulo:Tu,mapLinear:Km,inverseLerp:Qm,lerp:ia,damp:eg,pingpong:tg,smoothstep:ng,smootherstep:ig,randInt:sg,randFloat:rg,randFloatSpread:ag,seededRandom:og,degToRad:lg,radToDeg:cg,isPowerOfTwo:hg,ceilPowerOfTwo:ug,floorPowerOfTwo:dg,setQuaternionFromProperEuler:fg,normalize:gt,denormalize:Zn},Du=class Du{constructor(e=0,t=0){this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("THREE.Vector2: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("THREE.Vector2: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){let t=this.x,n=this.y,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6],this.y=s[1]*t+s[4]*n+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=je(this.x,e.x,t.x),this.y=je(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=je(this.x,e,t),this.y=je(this.y,e,t),this}clampLength(e,t){let n=this.length();return this.divideScalar(n||1).multiplyScalar(je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let n=this.dot(e)/t;return Math.acos(je(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){let n=Math.cos(t),s=Math.sin(t),r=this.x-e.x,a=this.y-e.y;return this.x=r*n-a*s+e.x,this.y=r*s+a*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}};Du.prototype.isVector2=!0;var Z=Du,In=class{constructor(e=0,t=0,n=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=s}static slerpFlat(e,t,n,s,r,a,o){let c=n[s+0],l=n[s+1],h=n[s+2],d=n[s+3],u=r[a+0],f=r[a+1],g=r[a+2],_=r[a+3];if(d!==_||c!==u||l!==f||h!==g){let p=c*u+l*f+h*g+d*_;p<0&&(u=-u,f=-f,g=-g,_=-_,p=-p);let m=1-o;if(p<.9995){let M=Math.acos(p),S=Math.sin(M);m=Math.sin(m*M)/S,o=Math.sin(o*M)/S,c=c*m+u*o,l=l*m+f*o,h=h*m+g*o,d=d*m+_*o}else{c=c*m+u*o,l=l*m+f*o,h=h*m+g*o,d=d*m+_*o;let M=1/Math.sqrt(c*c+l*l+h*h+d*d);c*=M,l*=M,h*=M,d*=M}}e[t]=c,e[t+1]=l,e[t+2]=h,e[t+3]=d}static multiplyQuaternionsFlat(e,t,n,s,r,a){let o=n[s],c=n[s+1],l=n[s+2],h=n[s+3],d=r[a],u=r[a+1],f=r[a+2],g=r[a+3];return e[t]=o*g+h*d+c*f-l*u,e[t+1]=c*g+h*u+l*d-o*f,e[t+2]=l*g+h*f+o*u-c*d,e[t+3]=h*g-o*d-c*u-l*f,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,s){return this._x=e,this._y=t,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){let n=e._x,s=e._y,r=e._z,a=e._order,o=Math.cos,c=Math.sin,l=o(n/2),h=o(s/2),d=o(r/2),u=c(n/2),f=c(s/2),g=c(r/2);switch(a){case"XYZ":this._x=u*h*d+l*f*g,this._y=l*f*d-u*h*g,this._z=l*h*g+u*f*d,this._w=l*h*d-u*f*g;break;case"YXZ":this._x=u*h*d+l*f*g,this._y=l*f*d-u*h*g,this._z=l*h*g-u*f*d,this._w=l*h*d+u*f*g;break;case"ZXY":this._x=u*h*d-l*f*g,this._y=l*f*d+u*h*g,this._z=l*h*g+u*f*d,this._w=l*h*d-u*f*g;break;case"ZYX":this._x=u*h*d-l*f*g,this._y=l*f*d+u*h*g,this._z=l*h*g-u*f*d,this._w=l*h*d+u*f*g;break;case"YZX":this._x=u*h*d+l*f*g,this._y=l*f*d+u*h*g,this._z=l*h*g-u*f*d,this._w=l*h*d-u*f*g;break;case"XZY":this._x=u*h*d-l*f*g,this._y=l*f*d-u*h*g,this._z=l*h*g+u*f*d,this._w=l*h*d+u*f*g;break;default:Ze("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){let n=t/2,s=Math.sin(n);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){let t=e.elements,n=t[0],s=t[4],r=t[8],a=t[1],o=t[5],c=t[9],l=t[2],h=t[6],d=t[10],u=n+o+d;if(u>0){let f=.5/Math.sqrt(u+1);this._w=.25/f,this._x=(h-c)*f,this._y=(r-l)*f,this._z=(a-s)*f}else if(n>o&&n>d){let f=2*Math.sqrt(1+n-o-d);this._w=(h-c)/f,this._x=.25*f,this._y=(s+a)/f,this._z=(r+l)/f}else if(o>d){let f=2*Math.sqrt(1+o-n-d);this._w=(r-l)/f,this._x=(s+a)/f,this._y=.25*f,this._z=(c+h)/f}else{let f=2*Math.sqrt(1+d-n-o);this._w=(a-s)/f,this._x=(r+l)/f,this._y=(c+h)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(je(this.dot(e),-1,1)))}rotateTowards(e,t){let n=this.angleTo(e);if(n===0)return this;let s=Math.min(1,t/n);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){let n=e._x,s=e._y,r=e._z,a=e._w,o=t._x,c=t._y,l=t._z,h=t._w;return this._x=n*h+a*o+s*l-r*c,this._y=s*h+a*c+r*o-n*l,this._z=r*h+a*l+n*c-s*o,this._w=a*h-n*o-s*c-r*l,this._onChangeCallback(),this}slerp(e,t){let n=e._x,s=e._y,r=e._z,a=e._w,o=this.dot(e);o<0&&(n=-n,s=-s,r=-r,a=-a,o=-o);let c=1-t;if(o<.9995){let l=Math.acos(o),h=Math.sin(l);c=Math.sin(c*l)/h,t=Math.sin(t*l)/h,this._x=this._x*c+n*t,this._y=this._y*c+s*t,this._z=this._z*c+r*t,this._w=this._w*c+a*t,this._onChangeCallback()}else this._x=this._x*c+n*t,this._y=this._y*c+s*t,this._z=this._z*c+r*t,this._w=this._w*c+a*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){let e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},Lu=class Lu{constructor(e=0,t=0,n=0){this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("THREE.Vector3: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("THREE.Vector3: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(zd.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(zd.setFromAxisAngle(e,t))}applyMatrix3(e){let t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6]*s,this.y=r[1]*t+r[4]*n+r[7]*s,this.z=r[2]*t+r[5]*n+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){let t=this.x,n=this.y,s=this.z,r=e.elements,a=1/(r[3]*t+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*n+r[8]*s+r[12])*a,this.y=(r[1]*t+r[5]*n+r[9]*s+r[13])*a,this.z=(r[2]*t+r[6]*n+r[10]*s+r[14])*a,this}applyQuaternion(e){let t=this.x,n=this.y,s=this.z,r=e.x,a=e.y,o=e.z,c=e.w,l=2*(a*s-o*n),h=2*(o*t-r*s),d=2*(r*n-a*t);return this.x=t+c*l+a*d-o*h,this.y=n+c*h+o*l-r*d,this.z=s+c*d+r*h-a*l,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){let t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*n+r[8]*s,this.y=r[1]*t+r[5]*n+r[9]*s,this.z=r[2]*t+r[6]*n+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=je(this.x,e.x,t.x),this.y=je(this.y,e.y,t.y),this.z=je(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=je(this.x,e,t),this.y=je(this.y,e,t),this.z=je(this.z,e,t),this}clampLength(e,t){let n=this.length();return this.divideScalar(n||1).multiplyScalar(je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){let n=e.x,s=e.y,r=e.z,a=t.x,o=t.y,c=t.z;return this.x=s*c-r*o,this.y=r*a-n*c,this.z=n*o-s*a,this}projectOnVector(e){let t=e.lengthSq();if(t===0)return this.set(0,0,0);let n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return Eh.copy(this).projectOnVector(e),this.sub(Eh)}reflect(e){return this.sub(Eh.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let n=this.dot(e)/t;return Math.acos(je(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,n=this.y-e.y,s=this.z-e.z;return t*t+n*n+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){let s=Math.sin(t)*e;return this.x=s*Math.sin(n),this.y=Math.cos(t)*e,this.z=s*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){let t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}};Lu.prototype.isVector3=!0;var R=Lu,Eh=new R,zd=new In,Uu=class Uu{constructor(e,t,n,s,r,a,o,c,l){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,a,o,c,l)}set(e,t,n,s,r,a,o,c,l){let h=this.elements;return h[0]=e,h[1]=s,h[2]=o,h[3]=t,h[4]=r,h[5]=c,h[6]=n,h[7]=a,h[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){let t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){let t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let n=e.elements,s=t.elements,r=this.elements,a=n[0],o=n[3],c=n[6],l=n[1],h=n[4],d=n[7],u=n[2],f=n[5],g=n[8],_=s[0],p=s[3],m=s[6],M=s[1],S=s[4],y=s[7],T=s[2],b=s[5],P=s[8];return r[0]=a*_+o*M+c*T,r[3]=a*p+o*S+c*b,r[6]=a*m+o*y+c*P,r[1]=l*_+h*M+d*T,r[4]=l*p+h*S+d*b,r[7]=l*m+h*y+d*P,r[2]=u*_+f*M+g*T,r[5]=u*p+f*S+g*b,r[8]=u*m+f*y+g*P,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){let e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],a=e[4],o=e[5],c=e[6],l=e[7],h=e[8];return t*a*h-t*o*l-n*r*h+n*o*c+s*r*l-s*a*c}invert(){let e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],a=e[4],o=e[5],c=e[6],l=e[7],h=e[8],d=h*a-o*l,u=o*c-h*r,f=l*r-a*c,g=t*d+n*u+s*f;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);let _=1/g;return e[0]=d*_,e[1]=(s*l-h*n)*_,e[2]=(o*n-s*a)*_,e[3]=u*_,e[4]=(h*t-s*c)*_,e[5]=(s*r-o*t)*_,e[6]=f*_,e[7]=(n*c-l*t)*_,e[8]=(a*t-n*r)*_,this}transpose(){let e,t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){let t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,s,r,a,o){let c=Math.cos(r),l=Math.sin(r);return this.set(n*c,n*l,-n*(c*a+l*o)+a+e,-s*l,s*c,-s*(-l*a+c*o)+o+t,0,0,1),this}scale(e,t){return Cs("Matrix3: .scale() is deprecated. Use .makeScale() instead."),this.premultiply(wh.makeScale(e,t)),this}rotate(e){return Cs("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."),this.premultiply(wh.makeRotation(-e)),this}translate(e,t){return Cs("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."),this.premultiply(wh.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){let t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){let t=this.elements,n=e.elements;for(let s=0;s<9;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){let n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}};Uu.prototype.isMatrix3=!0;var Qe=Uu,wh=new Qe,kd=new Qe().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Hd=new Qe().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function pg(){let i={enabled:!0,workingColorSpace:oa,spaces:{},convert:function(s,r,a){return this.enabled===!1||r===a||!r||!a||(this.spaces[r].transfer===pt&&(s.r=Ii(s.r),s.g=Ii(s.g),s.b=Ii(s.b)),this.spaces[r].primaries!==this.spaces[a].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===pt&&(s.r=mr(s.r),s.g=mr(s.g),s.b=mr(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===Oi?la:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,a){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[a].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return Cs("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return Cs("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[oa]:{primaries:e,whitePoint:n,transfer:la,toXYZ:kd,fromXYZ:Hd,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Lt},outputColorSpaceConfig:{drawingBufferColorSpace:Lt}},[Lt]:{primaries:e,whitePoint:n,transfer:pt,toXYZ:kd,fromXYZ:Hd,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Lt}}}),i}var ht=pg();function Ii(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function mr(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}var $s,vl=class{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{$s===void 0&&($s=ca("canvas")),$s.width=e.width,$s.height=e.height;let s=$s.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),n=$s}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){let t=ca("canvas");t.width=e.width,t.height=e.height;let n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);let s=n.getImageData(0,0,e.width,e.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=Ii(r[a]/255)*255;return n.putImageData(s,0,0),t}else if(e.data){let t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(Ii(t[n]/255)*255):t[n]=Ii(t[n]);return{data:t,width:e.width,height:e.height}}else return Ze("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}},mg=0,vr=class{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:mg++}),this.uuid=di(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){let t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):typeof VideoFrame<"u"&&t instanceof VideoFrame?e.set(t.displayWidth,t.displayHeight,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){let t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];let n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(Th(s[a].image)):r.push(Th(s[a]))}else r=Th(s);n.url=r}return t||(e.images[this.uuid]=n),n}};function Th(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?vl.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(Ze("Texture: Unable to serialize Texture."),{})}var gg=0,Ah=new R,pn=class i extends jn{constructor(e=i.DEFAULT_IMAGE,t=i.DEFAULT_MAPPING,n=ui,s=ui,r=en,a=ds,o=bn,c=hn,l=i.DEFAULT_ANISOTROPY,h=Oi){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:gg++}),this.uuid=di(),this.name="",this.source=new vr(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=l,this.format=o,this.internalFormat=null,this.type=c,this.offset=new Z(0,0),this.repeat=new Z(1,1),this.center=new Z(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Qe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0,this.normalized=!1}get width(){return this.source.getSize(Ah).x}get height(){return this.source.getSize(Ah).y}get depth(){return this.source.getSize(Ah).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(let t in e){let n=e[t];if(n===void 0){Ze(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}let s=this[t];if(s===void 0){Ze(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[t]=n}}toJSON(e){let t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];let n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==xu)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case kn:e.x=e.x-Math.floor(e.x);break;case ui:e.x=e.x<0?0:1;break;case gl:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case kn:e.y=e.y-Math.floor(e.y);break;case ui:e.y=e.y<0?0:1;break;case gl:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}};pn.DEFAULT_IMAGE=null;pn.DEFAULT_MAPPING=xu;pn.DEFAULT_ANISOTROPY=1;var Nu=class Nu{constructor(e=0,t=0,n=0,s=1){this.x=e,this.y=t,this.z=n,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,s){return this.x=e,this.y=t,this.z=n,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("THREE.Vector4: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("THREE.Vector4: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){let t=this.x,n=this.y,s=this.z,r=this.w,a=e.elements;return this.x=a[0]*t+a[4]*n+a[8]*s+a[12]*r,this.y=a[1]*t+a[5]*n+a[9]*s+a[13]*r,this.z=a[2]*t+a[6]*n+a[10]*s+a[14]*r,this.w=a[3]*t+a[7]*n+a[11]*s+a[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);let t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,s,r,c=e.elements,l=c[0],h=c[4],d=c[8],u=c[1],f=c[5],g=c[9],_=c[2],p=c[6],m=c[10];if(Math.abs(h-u)<.01&&Math.abs(d-_)<.01&&Math.abs(g-p)<.01){if(Math.abs(h+u)<.1&&Math.abs(d+_)<.1&&Math.abs(g+p)<.1&&Math.abs(l+f+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;let S=(l+1)/2,y=(f+1)/2,T=(m+1)/2,b=(h+u)/4,P=(d+_)/4,x=(g+p)/4;return S>y&&S>T?S<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(S),s=b/n,r=P/n):y>T?y<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(y),n=b/s,r=x/s):T<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(T),n=P/r,s=x/r),this.set(n,s,r,t),this}let M=Math.sqrt((p-g)*(p-g)+(d-_)*(d-_)+(u-h)*(u-h));return Math.abs(M)<.001&&(M=1),this.x=(p-g)/M,this.y=(d-_)/M,this.z=(u-h)/M,this.w=Math.acos((l+f+m-1)/2),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=je(this.x,e.x,t.x),this.y=je(this.y,e.y,t.y),this.z=je(this.z,e.z,t.z),this.w=je(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=je(this.x,e,t),this.y=je(this.y,e,t),this.z=je(this.z,e,t),this.w=je(this.w,e,t),this}clampLength(e,t){let n=this.length();return this.divideScalar(n||1).multiplyScalar(je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}};Nu.prototype.isVector4=!0;var mt=Nu,yl=class extends jn{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:en,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new mt(0,0,e,t),this.scissorTest=!1,this.viewport=new mt(0,0,e,t),this.textures=[];let s={width:e,height:t,depth:n.depth},r=new pn(s),a=n.count;for(let o=0;o<a;o++)this.textures[o]=r.clone(),this.textures[o].isRenderTargetTexture=!0,this.textures[o].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview,this.useArrayDepthTexture=n.useArrayDepthTexture}_setTextureOptions(e={}){let t={minFilter:en,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;let s=Object.assign({},e.textures[t].image);this.textures[t].source=new vr(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this.multiview=e.multiview,this.useArrayDepthTexture=e.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}},Ht=class extends yl{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}},ua=class extends pn{constructor(e=null,t=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=Ot,this.minFilter=Ot,this.wrapR=ui,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}};var Ml=class extends pn{constructor(e=null,t=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=Ot,this.minFilter=Ot,this.wrapR=ui,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var Zl=class Zl{constructor(e,t,n,s,r,a,o,c,l,h,d,u,f,g,_,p){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,a,o,c,l,h,d,u,f,g,_,p)}set(e,t,n,s,r,a,o,c,l,h,d,u,f,g,_,p){let m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=s,m[1]=r,m[5]=a,m[9]=o,m[13]=c,m[2]=l,m[6]=h,m[10]=d,m[14]=u,m[3]=f,m[7]=g,m[11]=_,m[15]=p,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Zl().fromArray(this.elements)}copy(e){let t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){let t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){let t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return this.determinantAffine()===0?(e.set(1,0,0),t.set(0,1,0),n.set(0,0,1),this):(e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this)}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){if(e.determinantAffine()===0)return this.identity();let t=this.elements,n=e.elements,s=1/Js.setFromMatrixColumn(e,0).length(),r=1/Js.setFromMatrixColumn(e,1).length(),a=1/Js.setFromMatrixColumn(e,2).length();return t[0]=n[0]*s,t[1]=n[1]*s,t[2]=n[2]*s,t[3]=0,t[4]=n[4]*r,t[5]=n[5]*r,t[6]=n[6]*r,t[7]=0,t[8]=n[8]*a,t[9]=n[9]*a,t[10]=n[10]*a,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){let t=this.elements,n=e.x,s=e.y,r=e.z,a=Math.cos(n),o=Math.sin(n),c=Math.cos(s),l=Math.sin(s),h=Math.cos(r),d=Math.sin(r);if(e.order==="XYZ"){let u=a*h,f=a*d,g=o*h,_=o*d;t[0]=c*h,t[4]=-c*d,t[8]=l,t[1]=f+g*l,t[5]=u-_*l,t[9]=-o*c,t[2]=_-u*l,t[6]=g+f*l,t[10]=a*c}else if(e.order==="YXZ"){let u=c*h,f=c*d,g=l*h,_=l*d;t[0]=u+_*o,t[4]=g*o-f,t[8]=a*l,t[1]=a*d,t[5]=a*h,t[9]=-o,t[2]=f*o-g,t[6]=_+u*o,t[10]=a*c}else if(e.order==="ZXY"){let u=c*h,f=c*d,g=l*h,_=l*d;t[0]=u-_*o,t[4]=-a*d,t[8]=g+f*o,t[1]=f+g*o,t[5]=a*h,t[9]=_-u*o,t[2]=-a*l,t[6]=o,t[10]=a*c}else if(e.order==="ZYX"){let u=a*h,f=a*d,g=o*h,_=o*d;t[0]=c*h,t[4]=g*l-f,t[8]=u*l+_,t[1]=c*d,t[5]=_*l+u,t[9]=f*l-g,t[2]=-l,t[6]=o*c,t[10]=a*c}else if(e.order==="YZX"){let u=a*c,f=a*l,g=o*c,_=o*l;t[0]=c*h,t[4]=_-u*d,t[8]=g*d+f,t[1]=d,t[5]=a*h,t[9]=-o*h,t[2]=-l*h,t[6]=f*d+g,t[10]=u-_*d}else if(e.order==="XZY"){let u=a*c,f=a*l,g=o*c,_=o*l;t[0]=c*h,t[4]=-d,t[8]=l*h,t[1]=u*d+_,t[5]=a*h,t[9]=f*d-g,t[2]=g*d-f,t[6]=o*h,t[10]=_*d+u}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(_g,e,xg)}lookAt(e,t,n){let s=this.elements;return Rn.subVectors(e,t),Rn.lengthSq()===0&&(Rn.z=1),Rn.normalize(),Ji.crossVectors(n,Rn),Ji.lengthSq()===0&&(Math.abs(n.z)===1?Rn.x+=1e-4:Rn.z+=1e-4,Rn.normalize(),Ji.crossVectors(n,Rn)),Ji.normalize(),To.crossVectors(Rn,Ji),s[0]=Ji.x,s[4]=To.x,s[8]=Rn.x,s[1]=Ji.y,s[5]=To.y,s[9]=Rn.y,s[2]=Ji.z,s[6]=To.z,s[10]=Rn.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let n=e.elements,s=t.elements,r=this.elements,a=n[0],o=n[4],c=n[8],l=n[12],h=n[1],d=n[5],u=n[9],f=n[13],g=n[2],_=n[6],p=n[10],m=n[14],M=n[3],S=n[7],y=n[11],T=n[15],b=s[0],P=s[4],x=s[8],E=s[12],C=s[1],I=s[5],L=s[9],X=s[13],q=s[2],F=s[6],Y=s[10],W=s[14],ie=s[3],ne=s[7],ge=s[11],ue=s[15];return r[0]=a*b+o*C+c*q+l*ie,r[4]=a*P+o*I+c*F+l*ne,r[8]=a*x+o*L+c*Y+l*ge,r[12]=a*E+o*X+c*W+l*ue,r[1]=h*b+d*C+u*q+f*ie,r[5]=h*P+d*I+u*F+f*ne,r[9]=h*x+d*L+u*Y+f*ge,r[13]=h*E+d*X+u*W+f*ue,r[2]=g*b+_*C+p*q+m*ie,r[6]=g*P+_*I+p*F+m*ne,r[10]=g*x+_*L+p*Y+m*ge,r[14]=g*E+_*X+p*W+m*ue,r[3]=M*b+S*C+y*q+T*ie,r[7]=M*P+S*I+y*F+T*ne,r[11]=M*x+S*L+y*Y+T*ge,r[15]=M*E+S*X+y*W+T*ue,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){let e=this.elements,t=e[0],n=e[4],s=e[8],r=e[12],a=e[1],o=e[5],c=e[9],l=e[13],h=e[2],d=e[6],u=e[10],f=e[14],g=e[3],_=e[7],p=e[11],m=e[15],M=c*f-l*u,S=o*f-l*d,y=o*u-c*d,T=a*f-l*h,b=a*u-c*h,P=a*d-o*h;return t*(_*M-p*S+m*y)-n*(g*M-p*T+m*b)+s*(g*S-_*T+m*P)-r*(g*y-_*b+p*P)}determinantAffine(){let e=this.elements,t=e[0],n=e[4],s=e[8],r=e[1],a=e[5],o=e[9],c=e[2],l=e[6],h=e[10];return t*(a*h-o*l)-n*(r*h-o*c)+s*(r*l-a*c)}transpose(){let e=this.elements,t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){let s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=n),this}invert(){let e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],a=e[4],o=e[5],c=e[6],l=e[7],h=e[8],d=e[9],u=e[10],f=e[11],g=e[12],_=e[13],p=e[14],m=e[15],M=t*o-n*a,S=t*c-s*a,y=t*l-r*a,T=n*c-s*o,b=n*l-r*o,P=s*l-r*c,x=h*_-d*g,E=h*p-u*g,C=h*m-f*g,I=d*p-u*_,L=d*m-f*_,X=u*m-f*p,q=M*X-S*L+y*I+T*C-b*E+P*x;if(q===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let F=1/q;return e[0]=(o*X-c*L+l*I)*F,e[1]=(s*L-n*X-r*I)*F,e[2]=(_*P-p*b+m*T)*F,e[3]=(u*b-d*P-f*T)*F,e[4]=(c*C-a*X-l*E)*F,e[5]=(t*X-s*C+r*E)*F,e[6]=(p*y-g*P-m*S)*F,e[7]=(h*P-u*y+f*S)*F,e[8]=(a*L-o*C+l*x)*F,e[9]=(n*C-t*L-r*x)*F,e[10]=(g*b-_*y+m*M)*F,e[11]=(d*y-h*b-f*M)*F,e[12]=(o*E-a*I-c*x)*F,e[13]=(t*I-n*E+s*x)*F,e[14]=(_*S-g*T-p*M)*F,e[15]=(h*T-d*S+u*M)*F,this}scale(e){let t=this.elements,n=e.x,s=e.y,r=e.z;return t[0]*=n,t[4]*=s,t[8]*=r,t[1]*=n,t[5]*=s,t[9]*=r,t[2]*=n,t[6]*=s,t[10]*=r,t[3]*=n,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){let e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,s))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){let t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){let t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){let t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){let n=Math.cos(t),s=Math.sin(t),r=1-n,a=e.x,o=e.y,c=e.z,l=r*a,h=r*o;return this.set(l*a+n,l*o-s*c,l*c+s*o,0,l*o+s*c,h*o+n,h*c-s*a,0,l*c-s*o,h*c+s*a,r*c*c+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,s,r,a){return this.set(1,n,r,0,e,1,a,0,t,s,1,0,0,0,0,1),this}compose(e,t,n){let s=this.elements,r=t._x,a=t._y,o=t._z,c=t._w,l=r+r,h=a+a,d=o+o,u=r*l,f=r*h,g=r*d,_=a*h,p=a*d,m=o*d,M=c*l,S=c*h,y=c*d,T=n.x,b=n.y,P=n.z;return s[0]=(1-(_+m))*T,s[1]=(f+y)*T,s[2]=(g-S)*T,s[3]=0,s[4]=(f-y)*b,s[5]=(1-(u+m))*b,s[6]=(p+M)*b,s[7]=0,s[8]=(g+S)*P,s[9]=(p-M)*P,s[10]=(1-(u+_))*P,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,n){let s=this.elements;e.x=s[12],e.y=s[13],e.z=s[14];let r=this.determinantAffine();if(r===0)return n.set(1,1,1),t.identity(),this;let a=Js.set(s[0],s[1],s[2]).length(),o=Js.set(s[4],s[5],s[6]).length(),c=Js.set(s[8],s[9],s[10]).length();r<0&&(a=-a),Xn.copy(this);let l=1/a,h=1/o,d=1/c;return Xn.elements[0]*=l,Xn.elements[1]*=l,Xn.elements[2]*=l,Xn.elements[4]*=h,Xn.elements[5]*=h,Xn.elements[6]*=h,Xn.elements[8]*=d,Xn.elements[9]*=d,Xn.elements[10]*=d,t.setFromRotationMatrix(Xn),n.x=a,n.y=o,n.z=c,this}makePerspective(e,t,n,s,r,a,o=$n,c=!1){let l=this.elements,h=2*r/(t-e),d=2*r/(n-s),u=(t+e)/(t-e),f=(n+s)/(n-s),g,_;if(c)g=r/(a-r),_=a*r/(a-r);else if(o===$n)g=-(a+r)/(a-r),_=-2*a*r/(a-r);else if(o===gr)g=-a/(a-r),_=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return l[0]=h,l[4]=0,l[8]=u,l[12]=0,l[1]=0,l[5]=d,l[9]=f,l[13]=0,l[2]=0,l[6]=0,l[10]=g,l[14]=_,l[3]=0,l[7]=0,l[11]=-1,l[15]=0,this}makeOrthographic(e,t,n,s,r,a,o=$n,c=!1){let l=this.elements,h=2/(t-e),d=2/(n-s),u=-(t+e)/(t-e),f=-(n+s)/(n-s),g,_;if(c)g=1/(a-r),_=a/(a-r);else if(o===$n)g=-2/(a-r),_=-(a+r)/(a-r);else if(o===gr)g=-1/(a-r),_=-r/(a-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return l[0]=h,l[4]=0,l[8]=0,l[12]=u,l[1]=0,l[5]=d,l[9]=0,l[13]=f,l[2]=0,l[6]=0,l[10]=g,l[14]=_,l[3]=0,l[7]=0,l[11]=0,l[15]=1,this}equals(e){let t=this.elements,n=e.elements;for(let s=0;s<16;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){let n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}};Zl.prototype.isMatrix4=!0;var st=Zl,Js=new R,Xn=new st,_g=new R(0,0,0),xg=new R(1,1,1),Ji=new R,To=new R,Rn=new R,Vd=new st,Gd=new In,Dn=class i{constructor(e=0,t=0,n=0,s=i.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,s=this._order){return this._x=e,this._y=t,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){let s=e.elements,r=s[0],a=s[4],o=s[8],c=s[1],l=s[5],h=s[9],d=s[2],u=s[6],f=s[10];switch(t){case"XYZ":this._y=Math.asin(je(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,f),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(u,l),this._z=0);break;case"YXZ":this._x=Math.asin(-je(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,f),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-d,r),this._z=0);break;case"ZXY":this._x=Math.asin(je(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(-d,f),this._z=Math.atan2(-a,l)):(this._y=0,this._z=Math.atan2(c,r));break;case"ZYX":this._y=Math.asin(-je(d,-1,1)),Math.abs(d)<.9999999?(this._x=Math.atan2(u,f),this._z=Math.atan2(c,r)):(this._x=0,this._z=Math.atan2(-a,l));break;case"YZX":this._z=Math.asin(je(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-h,l),this._y=Math.atan2(-d,r)):(this._x=0,this._y=Math.atan2(o,f));break;case"XZY":this._z=Math.asin(-je(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(u,l),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-h,f),this._y=0);break;default:Ze("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return Vd.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Vd,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Gd.setFromEuler(this),this.setFromQuaternion(Gd,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};Dn.DEFAULT_ORDER="XYZ";var yr=class{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}},vg=0,Wd=new R,js=new In,Ti=new st,Ao=new R,qr=new R,yg=new R,Mg=new In,Xd=new R(1,0,0),qd=new R(0,1,0),Yd=new R(0,0,1),Zd={type:"added"},Sg={type:"removed"},Ks={type:"childadded",child:null},Rh={type:"childremoved",child:null},ft=class i extends jn{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:vg++}),this.uuid=di(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=i.DEFAULT_UP.clone();let e=new R,t=new Dn,n=new In,s=new R(1,1,1);function r(){n.setFromEuler(t,!1)}function a(){t.setFromQuaternion(n,void 0,!1)}t._onChange(r),n._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new st},normalMatrix:{value:new Qe}}),this.matrix=new st,this.matrixWorld=new st,this.matrixAutoUpdate=i.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=i.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new yr,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return js.setFromAxisAngle(e,t),this.quaternion.multiply(js),this}rotateOnWorldAxis(e,t){return js.setFromAxisAngle(e,t),this.quaternion.premultiply(js),this}rotateX(e){return this.rotateOnAxis(Xd,e)}rotateY(e){return this.rotateOnAxis(qd,e)}rotateZ(e){return this.rotateOnAxis(Yd,e)}translateOnAxis(e,t){return Wd.copy(e).applyQuaternion(this.quaternion),this.position.add(Wd.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(Xd,e)}translateY(e){return this.translateOnAxis(qd,e)}translateZ(e){return this.translateOnAxis(Yd,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Ti.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Ao.copy(e):Ao.set(e,t,n);let s=this.parent;this.updateWorldMatrix(!0,!1),qr.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Ti.lookAt(qr,Ao,this.up):Ti.lookAt(Ao,qr,this.up),this.quaternion.setFromRotationMatrix(Ti),s&&(Ti.extractRotation(s.matrixWorld),js.setFromRotationMatrix(Ti),this.quaternion.premultiply(js.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?($e("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Zd),Ks.child=e,this.dispatchEvent(Ks),Ks.child=null):$e("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}let t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(Sg),Rh.child=e,this.dispatchEvent(Rh),Rh.child=null),this}removeFromParent(){let e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Ti.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Ti.multiply(e.parent.matrixWorld)),e.applyMatrix4(Ti),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Zd),Ks.child=e,this.dispatchEvent(Ks),Ks.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,s=this.children.length;n<s;n++){let a=this.children[n].getObjectByProperty(e,t);if(a!==void 0)return a}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);let s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(qr,e,yg),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(qr,Mg,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);let t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);let t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);let t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverseVisible(e)}traverseAncestors(e){let t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let e=this.pivot;if(e!==null){let t=e.x,n=e.y,s=e.z,r=this.matrix.elements;r[12]+=t-r[0]*t-r[4]*n-r[8]*s,r[13]+=n-r[1]*t-r[5]*n-r[9]*s,r[14]+=s-r[2]*t-r[6]*n-r[10]*s}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);let t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t,n=!1){let s=this.parent;if(e===!0&&s!==null&&s.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||n)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,n=!0),t===!0){let r=this.children;for(let a=0,o=r.length;a<o;a++)r[a].updateWorldMatrix(!1,!0,n)}}toJSON(e){let t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),this.static!==!1&&(s.static=this.static),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.pivot!==null&&(s.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(s.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(s.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(o=>({...o})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(o,c){return o[c.uuid]===void 0&&(o[c.uuid]=c.toJSON(e)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);let o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){let c=o.shapes;if(Array.isArray(c))for(let l=0,h=c.length;l<h;l++){let d=c[l];r(e.shapes,d)}else r(e.shapes,c)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let o=[];for(let c=0,l=this.material.length;c<l;c++)o.push(r(e.materials,this.material[c]));s.material=o}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){let c=this.animations[o];s.animations.push(r(e.animations,c))}}if(t){let o=a(e.geometries),c=a(e.materials),l=a(e.textures),h=a(e.images),d=a(e.shapes),u=a(e.skeletons),f=a(e.animations),g=a(e.nodes);o.length>0&&(n.geometries=o),c.length>0&&(n.materials=c),l.length>0&&(n.textures=l),h.length>0&&(n.images=h),d.length>0&&(n.shapes=d),u.length>0&&(n.skeletons=u),f.length>0&&(n.animations=f),g.length>0&&(n.nodes=g)}return n.object=s,n;function a(o){let c=[];for(let l in o){let h=o[l];delete h.metadata,c.push(h)}return c}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot!==null?e.pivot.clone():null,this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){let s=e.children[n];this.add(s.clone())}return this}};ft.DEFAULT_UP=new R(0,1,0);ft.DEFAULT_MATRIX_AUTO_UPDATE=!0;ft.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var tt=class extends ft{constructor(){super(),this.isGroup=!0,this.type="Group"}},bg={type:"move"},Mr=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new tt,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new tt,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new R,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new R),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new tt,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new R,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new R,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){let t=this._hand;if(t)for(let n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let s=null,r=null,a=null,o=this._targetRay,c=this._grip,l=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(l&&e.hand){a=!0;for(let _ of e.hand.values()){let p=t.getJointPose(_,n),m=this._getHandJoint(l,_);p!==null&&(m.matrix.fromArray(p.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=p.radius),m.visible=p!==null}let h=l.joints["index-finger-tip"],d=l.joints["thumb-tip"],u=h.position.distanceTo(d.position),f=.02,g=.005;l.inputState.pinching&&u>f+g?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!l.inputState.pinching&&u<=f-g&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else c!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,n),r!==null&&(c.matrix.fromArray(r.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),c.matrixWorldNeedsUpdate=!0,r.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(r.linearVelocity)):c.hasLinearVelocity=!1,r.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(r.angularVelocity)):c.hasAngularVelocity=!1,c.eventsEnabled&&c.dispatchEvent({type:"gripUpdated",data:e,target:this})));o!==null&&(s=t.getPose(e.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(bg)))}return o!==null&&(o.visible=s!==null),c!==null&&(c.visible=r!==null),l!==null&&(l.visible=a!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){let n=new tt;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}},ip={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},ji={h:0,s:0,l:0},Ro={h:0,s:0,l:0};function Ch(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}var Pe=class{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){let s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Lt){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,ht.colorSpaceToWorking(this,t),this}setRGB(e,t,n,s=ht.workingColorSpace){return this.r=e,this.g=t,this.b=n,ht.colorSpaceToWorking(this,s),this}setHSL(e,t,n,s=ht.workingColorSpace){if(e=Tu(e,1),t=je(t,0,1),n=je(n,0,1),t===0)this.r=this.g=this.b=n;else{let r=n<=.5?n*(1+t):n+t-n*t,a=2*n-r;this.r=Ch(a,r,e+1/3),this.g=Ch(a,r,e),this.b=Ch(a,r,e-1/3)}return ht.colorSpaceToWorking(this,s),this}setStyle(e,t=Lt){function n(r){r!==void 0&&parseFloat(r)<1&&Ze("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r,a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:Ze("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){let r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(a===6)return this.setHex(parseInt(r,16),t);Ze("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Lt){let n=ip[e.toLowerCase()];return n!==void 0?this.setHex(n,t):Ze("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Ii(e.r),this.g=Ii(e.g),this.b=Ii(e.b),this}copyLinearToSRGB(e){return this.r=mr(e.r),this.g=mr(e.g),this.b=mr(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Lt){return ht.workingToColorSpace(cn.copy(this),e),Math.round(je(cn.r*255,0,255))*65536+Math.round(je(cn.g*255,0,255))*256+Math.round(je(cn.b*255,0,255))}getHexString(e=Lt){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=ht.workingColorSpace){ht.workingToColorSpace(cn.copy(this),t);let n=cn.r,s=cn.g,r=cn.b,a=Math.max(n,s,r),o=Math.min(n,s,r),c,l,h=(o+a)/2;if(o===a)c=0,l=0;else{let d=a-o;switch(l=h<=.5?d/(a+o):d/(2-a-o),a){case n:c=(s-r)/d+(s<r?6:0);break;case s:c=(r-n)/d+2;break;case r:c=(n-s)/d+4;break}c/=6}return e.h=c,e.s=l,e.l=h,e}getRGB(e,t=ht.workingColorSpace){return ht.workingToColorSpace(cn.copy(this),t),e.r=cn.r,e.g=cn.g,e.b=cn.b,e}getStyle(e=Lt){ht.workingToColorSpace(cn.copy(this),e);let t=cn.r,n=cn.g,s=cn.b;return e!==Lt?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(e,t,n){return this.getHSL(ji),this.setHSL(ji.h+e,ji.s+t,ji.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(ji),e.getHSL(Ro);let n=ia(ji.h,Ro.h,t),s=ia(ji.s,Ro.s,t),r=ia(ji.l,Ro.l,t);return this.setHSL(n,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){let t=this.r,n=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*n+r[6]*s,this.g=r[1]*t+r[4]*n+r[7]*s,this.b=r[2]*t+r[5]*n+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},cn=new Pe;Pe.NAMES=ip;var da=class i{constructor(e,t=1,n=1e3){this.isFog=!0,this.name="",this.color=new Pe(e),this.near=t,this.far=n}clone(){return new i(this.color,this.near,this.far)}toJSON(){return{type:"Fog",name:this.name,color:this.color.getHex(),near:this.near,far:this.far}}},Ds=class extends ft{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Dn,this.environmentIntensity=1,this.environmentRotation=new Dn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){let t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}},qn=new R,Ai=new R,Ph=new R,Ri=new R,Qs=new R,er=new R,$d=new R,Ih=new R,Dh=new R,Lh=new R,Uh=new mt,Nh=new mt,Fh=new mt,hi=class i{constructor(e=new R,t=new R,n=new R){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,s){s.subVectors(n,t),qn.subVectors(e,t),s.cross(qn);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,n,s,r){qn.subVectors(s,t),Ai.subVectors(n,t),Ph.subVectors(e,t);let a=qn.dot(qn),o=qn.dot(Ai),c=qn.dot(Ph),l=Ai.dot(Ai),h=Ai.dot(Ph),d=a*l-o*o;if(d===0)return r.set(0,0,0),null;let u=1/d,f=(l*c-o*h)*u,g=(a*h-o*c)*u;return r.set(1-f-g,g,f)}static containsPoint(e,t,n,s){return this.getBarycoord(e,t,n,s,Ri)===null?!1:Ri.x>=0&&Ri.y>=0&&Ri.x+Ri.y<=1}static getInterpolation(e,t,n,s,r,a,o,c){return this.getBarycoord(e,t,n,s,Ri)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(r,Ri.x),c.addScaledVector(a,Ri.y),c.addScaledVector(o,Ri.z),c)}static getInterpolatedAttribute(e,t,n,s,r,a){return Uh.setScalar(0),Nh.setScalar(0),Fh.setScalar(0),Uh.fromBufferAttribute(e,t),Nh.fromBufferAttribute(e,n),Fh.fromBufferAttribute(e,s),a.setScalar(0),a.addScaledVector(Uh,r.x),a.addScaledVector(Nh,r.y),a.addScaledVector(Fh,r.z),a}static isFrontFacing(e,t,n,s){return qn.subVectors(n,t),Ai.subVectors(e,t),qn.cross(Ai).dot(s)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,s){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,n,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return qn.subVectors(this.c,this.b),Ai.subVectors(this.a,this.b),qn.cross(Ai).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return i.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return i.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,s,r){return i.getInterpolation(e,this.a,this.b,this.c,t,n,s,r)}containsPoint(e){return i.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return i.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){let n=this.a,s=this.b,r=this.c,a,o;Qs.subVectors(s,n),er.subVectors(r,n),Ih.subVectors(e,n);let c=Qs.dot(Ih),l=er.dot(Ih);if(c<=0&&l<=0)return t.copy(n);Dh.subVectors(e,s);let h=Qs.dot(Dh),d=er.dot(Dh);if(h>=0&&d<=h)return t.copy(s);let u=c*d-h*l;if(u<=0&&c>=0&&h<=0)return a=c/(c-h),t.copy(n).addScaledVector(Qs,a);Lh.subVectors(e,r);let f=Qs.dot(Lh),g=er.dot(Lh);if(g>=0&&f<=g)return t.copy(r);let _=f*l-c*g;if(_<=0&&l>=0&&g<=0)return o=l/(l-g),t.copy(n).addScaledVector(er,o);let p=h*g-f*d;if(p<=0&&d-h>=0&&f-g>=0)return $d.subVectors(r,s),o=(d-h)/(d-h+(f-g)),t.copy(s).addScaledVector($d,o);let m=1/(p+_+u);return a=_*m,o=u*m,t.copy(n).addScaledVector(Qs,a).addScaledVector(er,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},mn=class{constructor(e=new R(1/0,1/0,1/0),t=new R(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(Yn.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(Yn.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){let n=Yn.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);let n=e.geometry;if(n!==void 0){let r=n.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let a=0,o=r.count;a<o;a++)e.isMesh===!0?e.getVertexPosition(a,Yn):Yn.fromBufferAttribute(r,a),Yn.applyMatrix4(e.matrixWorld),this.expandByPoint(Yn);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Co.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Co.copy(n.boundingBox)),Co.applyMatrix4(e.matrixWorld),this.union(Co)}let s=e.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Yn),Yn.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Yr),Po.subVectors(this.max,Yr),tr.subVectors(e.a,Yr),nr.subVectors(e.b,Yr),ir.subVectors(e.c,Yr),Ki.subVectors(nr,tr),Qi.subVectors(ir,nr),bs.subVectors(tr,ir);let t=[0,-Ki.z,Ki.y,0,-Qi.z,Qi.y,0,-bs.z,bs.y,Ki.z,0,-Ki.x,Qi.z,0,-Qi.x,bs.z,0,-bs.x,-Ki.y,Ki.x,0,-Qi.y,Qi.x,0,-bs.y,bs.x,0];return!Oh(t,tr,nr,ir,Po)||(t=[1,0,0,0,1,0,0,0,1],!Oh(t,tr,nr,ir,Po))?!1:(Io.crossVectors(Ki,Qi),t=[Io.x,Io.y,Io.z],Oh(t,tr,nr,ir,Po))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Yn).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Yn).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Ci[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Ci[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Ci[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Ci[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Ci[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Ci[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Ci[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Ci[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Ci),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}},Ci=[new R,new R,new R,new R,new R,new R,new R,new R],Yn=new R,Co=new mn,tr=new R,nr=new R,ir=new R,Ki=new R,Qi=new R,bs=new R,Yr=new R,Po=new R,Io=new R,Es=new R;function Oh(i,e,t,n,s){for(let r=0,a=i.length-3;r<=a;r+=3){Es.fromArray(i,r);let o=s.x*Math.abs(Es.x)+s.y*Math.abs(Es.y)+s.z*Math.abs(Es.z),c=e.dot(Es),l=t.dot(Es),h=n.dot(Es);if(Math.max(-Math.max(c,l,h),Math.min(c,l,h))>o)return!1}return!0}var kt=new R,Do=new Z,Eg=0,Ut=class extends jn{constructor(e,t,n=!1){if(super(),Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:Eg++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=xl,this.updateRanges=[],this.gpuType=Vn,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[n+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)Do.fromBufferAttribute(this,t),Do.applyMatrix3(e),this.setXY(t,Do.x,Do.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix3(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix4(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyNormalMatrix(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.transformDirection(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=Zn(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=gt(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=Zn(t,this.array)),t}setX(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=Zn(t,this.array)),t}setY(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=Zn(t,this.array)),t}setZ(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=Zn(t,this.array)),t}setW(e,t){return this.normalized&&(t=gt(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=gt(t,this.array),n=gt(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,s){return e*=this.itemSize,this.normalized&&(t=gt(t,this.array),n=gt(n,this.array),s=gt(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e*=this.itemSize,this.normalized&&(t=gt(t,this.array),n=gt(n,this.array),s=gt(s,this.array),r=gt(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==xl&&(e.usage=this.usage),e}dispose(){this.dispatchEvent({type:"dispose"})}};var fa=class extends Ut{constructor(e,t,n){super(new Uint16Array(e),t,n)}};var pa=class extends Ut{constructor(e,t,n){super(new Uint32Array(e),t,n)}};var rt=class extends Ut{constructor(e,t,n){super(new Float32Array(e),t,n)}},wg=new mn,Zr=new R,Bh=new R,yn=class{constructor(e=new R,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){let n=this.center;t!==void 0?n.copy(t):wg.setFromPoints(e).getCenter(n);let s=0;for(let r=0,a=e.length;r<a;r++)s=Math.max(s,n.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){let t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){let n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Zr.subVectors(e,this.center);let t=Zr.lengthSq();if(t>this.radius*this.radius){let n=Math.sqrt(t),s=(n-this.radius)*.5;this.center.addScaledVector(Zr,s/n),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Bh.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Zr.copy(e.center).add(Bh)),this.expandByPoint(Zr.copy(e.center).sub(Bh))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}},Tg=0,Bn=new st,zh=new ft,sr=new R,Cn=new mn,$r=new mn,Jt=new R,ut=class i extends jn{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:Tg++}),this.uuid=di(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(Jm(e)?pa:fa)(e,1):this.index=e,this}setIndirect(e,t=0){return this.indirect=e,this.indirectOffset=t,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){let t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);let n=this.attributes.normal;if(n!==void 0){let r=new Qe().getNormalMatrix(e);n.applyNormalMatrix(r),n.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(e){return Bn.makeRotationFromQuaternion(e),this.applyMatrix4(Bn),this}rotateX(e){return Bn.makeRotationX(e),this.applyMatrix4(Bn),this}rotateY(e){return Bn.makeRotationY(e),this.applyMatrix4(Bn),this}rotateZ(e){return Bn.makeRotationZ(e),this.applyMatrix4(Bn),this}translate(e,t,n){return Bn.makeTranslation(e,t,n),this.applyMatrix4(Bn),this}scale(e,t,n){return Bn.makeScale(e,t,n),this.applyMatrix4(Bn),this}lookAt(e){return zh.lookAt(e),zh.updateMatrix(),this.applyMatrix4(zh.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(sr).negate(),this.translate(sr.x,sr.y,sr.z),this}setFromPoints(e){let t=this.getAttribute("position");if(t===void 0){let n=[];for(let s=0,r=e.length;s<r;s++){let a=e[s];n.push(a.x,a.y,a.z||0)}this.setAttribute("position",new rt(n,3))}else{let n=Math.min(e.length,t.count);for(let s=0;s<n;s++){let r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&Ze("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new mn);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){$e("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new R(-1/0,-1/0,-1/0),new R(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,s=t.length;n<s;n++){let r=t[n];Cn.setFromBufferAttribute(r),this.morphTargetsRelative?(Jt.addVectors(this.boundingBox.min,Cn.min),this.boundingBox.expandByPoint(Jt),Jt.addVectors(this.boundingBox.max,Cn.max),this.boundingBox.expandByPoint(Jt)):(this.boundingBox.expandByPoint(Cn.min),this.boundingBox.expandByPoint(Cn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&$e('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new yn);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){$e("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new R,1/0);return}if(e){let n=this.boundingSphere.center;if(Cn.setFromBufferAttribute(e),t)for(let r=0,a=t.length;r<a;r++){let o=t[r];$r.setFromBufferAttribute(o),this.morphTargetsRelative?(Jt.addVectors(Cn.min,$r.min),Cn.expandByPoint(Jt),Jt.addVectors(Cn.max,$r.max),Cn.expandByPoint(Jt)):(Cn.expandByPoint($r.min),Cn.expandByPoint($r.max))}Cn.getCenter(n);let s=0;for(let r=0,a=e.count;r<a;r++)Jt.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(Jt));if(t)for(let r=0,a=t.length;r<a;r++){let o=t[r],c=this.morphTargetsRelative;for(let l=0,h=o.count;l<h;l++)Jt.fromBufferAttribute(o,l),c&&(sr.fromBufferAttribute(e,l),Jt.add(sr)),s=Math.max(s,n.distanceToSquared(Jt))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&$e('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){$e("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let n=t.position,s=t.normal,r=t.uv,a=this.getAttribute("tangent");(a===void 0||a.count!==n.count)&&(a=new Ut(new Float32Array(4*n.count),4),this.setAttribute("tangent",a));let o=[],c=[];for(let x=0;x<n.count;x++)o[x]=new R,c[x]=new R;let l=new R,h=new R,d=new R,u=new Z,f=new Z,g=new Z,_=new R,p=new R;function m(x,E,C){l.fromBufferAttribute(n,x),h.fromBufferAttribute(n,E),d.fromBufferAttribute(n,C),u.fromBufferAttribute(r,x),f.fromBufferAttribute(r,E),g.fromBufferAttribute(r,C),h.sub(l),d.sub(l),f.sub(u),g.sub(u);let I=1/(f.x*g.y-g.x*f.y);isFinite(I)&&(_.copy(h).multiplyScalar(g.y).addScaledVector(d,-f.y).multiplyScalar(I),p.copy(d).multiplyScalar(f.x).addScaledVector(h,-g.x).multiplyScalar(I),o[x].add(_),o[E].add(_),o[C].add(_),c[x].add(p),c[E].add(p),c[C].add(p))}let M=this.groups;M.length===0&&(M=[{start:0,count:e.count}]);for(let x=0,E=M.length;x<E;++x){let C=M[x],I=C.start,L=C.count;for(let X=I,q=I+L;X<q;X+=3)m(e.getX(X+0),e.getX(X+1),e.getX(X+2))}let S=new R,y=new R,T=new R,b=new R;function P(x){T.fromBufferAttribute(s,x),b.copy(T);let E=o[x];S.copy(E),S.sub(T.multiplyScalar(T.dot(E))).normalize(),y.crossVectors(b,E);let I=y.dot(c[x])<0?-1:1;a.setXYZW(x,S.x,S.y,S.z,I)}for(let x=0,E=M.length;x<E;++x){let C=M[x],I=C.start,L=C.count;for(let X=I,q=I+L;X<q;X+=3)P(e.getX(X+0)),P(e.getX(X+1)),P(e.getX(X+2))}this._transformed=!0}computeVertexNormals(){let e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0||n.count!==t.count)n=new Ut(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let u=0,f=n.count;u<f;u++)n.setXYZ(u,0,0,0);let s=new R,r=new R,a=new R,o=new R,c=new R,l=new R,h=new R,d=new R;if(e)for(let u=0,f=e.count;u<f;u+=3){let g=e.getX(u+0),_=e.getX(u+1),p=e.getX(u+2);s.fromBufferAttribute(t,g),r.fromBufferAttribute(t,_),a.fromBufferAttribute(t,p),h.subVectors(a,r),d.subVectors(s,r),h.cross(d),o.fromBufferAttribute(n,g),c.fromBufferAttribute(n,_),l.fromBufferAttribute(n,p),o.add(h),c.add(h),l.add(h),n.setXYZ(g,o.x,o.y,o.z),n.setXYZ(_,c.x,c.y,c.z),n.setXYZ(p,l.x,l.y,l.z)}else for(let u=0,f=t.count;u<f;u+=3)s.fromBufferAttribute(t,u+0),r.fromBufferAttribute(t,u+1),a.fromBufferAttribute(t,u+2),h.subVectors(a,r),d.subVectors(s,r),h.cross(d),n.setXYZ(u+0,h.x,h.y,h.z),n.setXYZ(u+1,h.x,h.y,h.z),n.setXYZ(u+2,h.x,h.y,h.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){let e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)Jt.fromBufferAttribute(e,t),Jt.normalize(),e.setXYZ(t,Jt.x,Jt.y,Jt.z)}toNonIndexed(){function e(o,c){let l=o.array,h=o.itemSize,d=o.normalized,u=new l.constructor(c.length*h),f=0,g=0;for(let _=0,p=c.length;_<p;_++){o.isInterleavedBufferAttribute?f=c[_]*o.data.stride+o.offset:f=c[_]*h;for(let m=0;m<h;m++)u[g++]=l[f++]}return new Ut(u,h,d)}if(this.index===null)return Ze("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let t=new i,n=this.index.array,s=this.attributes;for(let o in s){let c=s[o],l=e(c,n);t.setAttribute(o,l)}let r=this.morphAttributes;for(let o in r){let c=[],l=r[o];for(let h=0,d=l.length;h<d;h++){let u=l[h],f=e(u,n);c.push(f)}t.morphAttributes[o]=c}t.morphTargetsRelative=this.morphTargetsRelative;let a=this.groups;for(let o=0,c=a.length;o<c;o++){let l=a[o];t.addGroup(l.start,l.count,l.materialIndex)}return t}toJSON(){let e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.parameters!==void 0&&this._transformed===!0?"BufferGeometry":this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let c=this.parameters;for(let l in c)c[l]!==void 0&&(e[l]=c[l]);return e}e.data={attributes:{}};let t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});let n=this.attributes;for(let c in n){let l=n[c];e.data.attributes[c]=l.toJSON(e.data)}let s={},r=!1;for(let c in this.morphAttributes){let l=this.morphAttributes[c],h=[];for(let d=0,u=l.length;d<u;d++){let f=l[d];h.push(f.toJSON(e.data))}h.length>0&&(s[c]=h,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);let a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));let o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let t={};this.name=e.name;let n=e.index;n!==null&&this.setIndex(n.clone());let s=e.attributes;for(let l in s){let h=s[l];this.setAttribute(l,h.clone(t))}let r=e.morphAttributes;for(let l in r){let h=[],d=r[l];for(let u=0,f=d.length;u<f;u++)h.push(d[u].clone(t));this.morphAttributes[l]=h}this.morphTargetsRelative=e.morphTargetsRelative;let a=e.groups;for(let l=0,h=a.length;l<h;l++){let d=a[l];this.addGroup(d.start,d.count,d.materialIndex)}let o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());let c=e.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this._transformed=e._transformed,this}dispose(){this.dispatchEvent({type:"dispose"})}},ma=class{constructor(e,t){this.isInterleavedBuffer=!0,this.array=e,this.stride=t,this.count=e!==void 0?e.length/t:0,this.usage=xl,this.updateRanges=[],this.version=0,this.uuid=di()}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.array=new e.array.constructor(e.array),this.count=e.count,this.stride=e.stride,this.usage=e.usage,this}copyAt(e,t,n){e*=this.stride,n*=t.stride;for(let s=0,r=this.stride;s<r;s++)this.array[e+s]=t.array[n+s];return this}set(e,t=0){return this.array.set(e,t),this}clone(e){e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=di()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);let t=new this.array.constructor(e.arrayBuffers[this.array.buffer._uuid]),n=new this.constructor(t,this.stride);return n.setUsage(this.usage),n}onUpload(e){return this.onUploadCallback=e,this}toJSON(e){return e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=di()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=Array.from(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}},fn=new R,Ln=class i{constructor(e,t,n,s=!1){this.isInterleavedBufferAttribute=!0,this.name="",this.data=e,this.itemSize=t,this.offset=n,this.normalized=s}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(e){this.data.needsUpdate=e}applyMatrix4(e){for(let t=0,n=this.data.count;t<n;t++)fn.fromBufferAttribute(this,t),fn.applyMatrix4(e),this.setXYZ(t,fn.x,fn.y,fn.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)fn.fromBufferAttribute(this,t),fn.applyNormalMatrix(e),this.setXYZ(t,fn.x,fn.y,fn.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)fn.fromBufferAttribute(this,t),fn.transformDirection(e),this.setXYZ(t,fn.x,fn.y,fn.z);return this}getComponent(e,t){let n=this.array[e*this.data.stride+this.offset+t];return this.normalized&&(n=Zn(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=gt(n,this.array)),this.data.array[e*this.data.stride+this.offset+t]=n,this}setX(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset]=t,this}setY(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset+1]=t,this}setZ(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset+2]=t,this}setW(e,t){return this.normalized&&(t=gt(t,this.array)),this.data.array[e*this.data.stride+this.offset+3]=t,this}getX(e){let t=this.data.array[e*this.data.stride+this.offset];return this.normalized&&(t=Zn(t,this.array)),t}getY(e){let t=this.data.array[e*this.data.stride+this.offset+1];return this.normalized&&(t=Zn(t,this.array)),t}getZ(e){let t=this.data.array[e*this.data.stride+this.offset+2];return this.normalized&&(t=Zn(t,this.array)),t}getW(e){let t=this.data.array[e*this.data.stride+this.offset+3];return this.normalized&&(t=Zn(t,this.array)),t}setXY(e,t,n){return e=e*this.data.stride+this.offset,this.normalized&&(t=gt(t,this.array),n=gt(n,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=n,this}setXYZ(e,t,n,s){return e=e*this.data.stride+this.offset,this.normalized&&(t=gt(t,this.array),n=gt(n,this.array),s=gt(s,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=n,this.data.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e=e*this.data.stride+this.offset,this.normalized&&(t=gt(t,this.array),n=gt(n,this.array),s=gt(s,this.array),r=gt(r,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=n,this.data.array[e+2]=s,this.data.array[e+3]=r,this}clone(e){if(e===void 0){ha("InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.");let t=[];for(let n=0;n<this.count;n++){let s=n*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return new Ut(new this.array.constructor(t),this.itemSize,this.normalized)}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.clone(e)),new i(e.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(e){if(e===void 0){ha("InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.");let t=[];for(let n=0;n<this.count;n++){let s=n*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:t,normalized:this.normalized}}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.toJSON(e)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}},Ag=0,Mn=class extends jn{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Ag++}),this.uuid=di(),this.name="",this.type="Material",this.blending=Ps,this.side=Jn,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=ol,this.blendDst=ll,this.blendEquation=Pn,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Pe(0,0,0),this.blendAlpha=0,this.depthFunc=Is,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=iu,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=As,this.stencilZFail=As,this.stencilZPass=As,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(let t in e){let n=e[t];if(n===void 0){Ze(`Material: parameter '${t}' has value of undefined.`);continue}let s=this[t];if(s===void 0){Ze(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector2&&n&&n.isVector2||s&&s.isEuler&&n&&n.isEuler||s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[t]=n}}toJSON(e){let t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});let n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==Ps&&(n.blending=this.blending),this.side!==Jn&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==ol&&(n.blendSrc=this.blendSrc),this.blendDst!==ll&&(n.blendDst=this.blendDst),this.blendEquation!==Pn&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==Is&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==iu&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==As&&(n.stencilFail=this.stencilFail),this.stencilZFail!==As&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==As&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.allowOverride===!1&&(n.allowOverride=!1),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){let a=[];for(let o in r){let c=r[o];delete c.metadata,a.push(c)}return a}if(t){let r=s(e.textures),a=s(e.images);r.length>0&&(n.textures=r),a.length>0&&(n.images=a)}return n}fromJSON(e,t){if(e.uuid!==void 0&&(this.uuid=e.uuid),e.name!==void 0&&(this.name=e.name),e.color!==void 0&&this.color!==void 0&&this.color.setHex(e.color),e.roughness!==void 0&&(this.roughness=e.roughness),e.metalness!==void 0&&(this.metalness=e.metalness),e.sheen!==void 0&&(this.sheen=e.sheen),e.sheenColor!==void 0&&(this.sheenColor=new Pe().setHex(e.sheenColor)),e.sheenRoughness!==void 0&&(this.sheenRoughness=e.sheenRoughness),e.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(e.emissive),e.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(e.specular),e.specularIntensity!==void 0&&(this.specularIntensity=e.specularIntensity),e.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(e.specularColor),e.shininess!==void 0&&(this.shininess=e.shininess),e.clearcoat!==void 0&&(this.clearcoat=e.clearcoat),e.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=e.clearcoatRoughness),e.dispersion!==void 0&&(this.dispersion=e.dispersion),e.iridescence!==void 0&&(this.iridescence=e.iridescence),e.iridescenceIOR!==void 0&&(this.iridescenceIOR=e.iridescenceIOR),e.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessRange=e.iridescenceThicknessRange),e.transmission!==void 0&&(this.transmission=e.transmission),e.thickness!==void 0&&(this.thickness=e.thickness),e.attenuationDistance!==void 0&&(this.attenuationDistance=e.attenuationDistance),e.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(e.attenuationColor),e.anisotropy!==void 0&&(this.anisotropy=e.anisotropy),e.anisotropyRotation!==void 0&&(this.anisotropyRotation=e.anisotropyRotation),e.fog!==void 0&&(this.fog=e.fog),e.flatShading!==void 0&&(this.flatShading=e.flatShading),e.blending!==void 0&&(this.blending=e.blending),e.combine!==void 0&&(this.combine=e.combine),e.side!==void 0&&(this.side=e.side),e.shadowSide!==void 0&&(this.shadowSide=e.shadowSide),e.opacity!==void 0&&(this.opacity=e.opacity),e.transparent!==void 0&&(this.transparent=e.transparent),e.alphaTest!==void 0&&(this.alphaTest=e.alphaTest),e.alphaHash!==void 0&&(this.alphaHash=e.alphaHash),e.depthFunc!==void 0&&(this.depthFunc=e.depthFunc),e.depthTest!==void 0&&(this.depthTest=e.depthTest),e.depthWrite!==void 0&&(this.depthWrite=e.depthWrite),e.colorWrite!==void 0&&(this.colorWrite=e.colorWrite),e.blendSrc!==void 0&&(this.blendSrc=e.blendSrc),e.blendDst!==void 0&&(this.blendDst=e.blendDst),e.blendEquation!==void 0&&(this.blendEquation=e.blendEquation),e.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=e.blendSrcAlpha),e.blendDstAlpha!==void 0&&(this.blendDstAlpha=e.blendDstAlpha),e.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=e.blendEquationAlpha),e.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(e.blendColor),e.blendAlpha!==void 0&&(this.blendAlpha=e.blendAlpha),e.stencilWriteMask!==void 0&&(this.stencilWriteMask=e.stencilWriteMask),e.stencilFunc!==void 0&&(this.stencilFunc=e.stencilFunc),e.stencilRef!==void 0&&(this.stencilRef=e.stencilRef),e.stencilFuncMask!==void 0&&(this.stencilFuncMask=e.stencilFuncMask),e.stencilFail!==void 0&&(this.stencilFail=e.stencilFail),e.stencilZFail!==void 0&&(this.stencilZFail=e.stencilZFail),e.stencilZPass!==void 0&&(this.stencilZPass=e.stencilZPass),e.stencilWrite!==void 0&&(this.stencilWrite=e.stencilWrite),e.wireframe!==void 0&&(this.wireframe=e.wireframe),e.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=e.wireframeLinewidth),e.wireframeLinecap!==void 0&&(this.wireframeLinecap=e.wireframeLinecap),e.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=e.wireframeLinejoin),e.rotation!==void 0&&(this.rotation=e.rotation),e.linewidth!==void 0&&(this.linewidth=e.linewidth),e.dashSize!==void 0&&(this.dashSize=e.dashSize),e.gapSize!==void 0&&(this.gapSize=e.gapSize),e.scale!==void 0&&(this.scale=e.scale),e.polygonOffset!==void 0&&(this.polygonOffset=e.polygonOffset),e.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=e.polygonOffsetFactor),e.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=e.polygonOffsetUnits),e.dithering!==void 0&&(this.dithering=e.dithering),e.alphaToCoverage!==void 0&&(this.alphaToCoverage=e.alphaToCoverage),e.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=e.premultipliedAlpha),e.forceSinglePass!==void 0&&(this.forceSinglePass=e.forceSinglePass),e.allowOverride!==void 0&&(this.allowOverride=e.allowOverride),e.visible!==void 0&&(this.visible=e.visible),e.toneMapped!==void 0&&(this.toneMapped=e.toneMapped),e.userData!==void 0&&(this.userData=e.userData),e.vertexColors!==void 0&&(typeof e.vertexColors=="number"?this.vertexColors=e.vertexColors>0:this.vertexColors=e.vertexColors),e.size!==void 0&&(this.size=e.size),e.sizeAttenuation!==void 0&&(this.sizeAttenuation=e.sizeAttenuation),e.map!==void 0&&(this.map=t[e.map]||null),e.matcap!==void 0&&(this.matcap=t[e.matcap]||null),e.alphaMap!==void 0&&(this.alphaMap=t[e.alphaMap]||null),e.bumpMap!==void 0&&(this.bumpMap=t[e.bumpMap]||null),e.bumpScale!==void 0&&(this.bumpScale=e.bumpScale),e.normalMap!==void 0&&(this.normalMap=t[e.normalMap]||null),e.normalMapType!==void 0&&(this.normalMapType=e.normalMapType),e.normalScale!==void 0){let n=e.normalScale;Array.isArray(n)===!1&&(n=[n,n]),this.normalScale=new Z().fromArray(n)}return e.displacementMap!==void 0&&(this.displacementMap=t[e.displacementMap]||null),e.displacementScale!==void 0&&(this.displacementScale=e.displacementScale),e.displacementBias!==void 0&&(this.displacementBias=e.displacementBias),e.roughnessMap!==void 0&&(this.roughnessMap=t[e.roughnessMap]||null),e.metalnessMap!==void 0&&(this.metalnessMap=t[e.metalnessMap]||null),e.emissiveMap!==void 0&&(this.emissiveMap=t[e.emissiveMap]||null),e.emissiveIntensity!==void 0&&(this.emissiveIntensity=e.emissiveIntensity),e.specularMap!==void 0&&(this.specularMap=t[e.specularMap]||null),e.specularIntensityMap!==void 0&&(this.specularIntensityMap=t[e.specularIntensityMap]||null),e.specularColorMap!==void 0&&(this.specularColorMap=t[e.specularColorMap]||null),e.envMap!==void 0&&(this.envMap=t[e.envMap]||null),e.envMapRotation!==void 0&&this.envMapRotation.fromArray(e.envMapRotation),e.envMapIntensity!==void 0&&(this.envMapIntensity=e.envMapIntensity),e.reflectivity!==void 0&&(this.reflectivity=e.reflectivity),e.refractionRatio!==void 0&&(this.refractionRatio=e.refractionRatio),e.lightMap!==void 0&&(this.lightMap=t[e.lightMap]||null),e.lightMapIntensity!==void 0&&(this.lightMapIntensity=e.lightMapIntensity),e.aoMap!==void 0&&(this.aoMap=t[e.aoMap]||null),e.aoMapIntensity!==void 0&&(this.aoMapIntensity=e.aoMapIntensity),e.gradientMap!==void 0&&(this.gradientMap=t[e.gradientMap]||null),e.clearcoatMap!==void 0&&(this.clearcoatMap=t[e.clearcoatMap]||null),e.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=t[e.clearcoatRoughnessMap]||null),e.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=t[e.clearcoatNormalMap]||null),e.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new Z().fromArray(e.clearcoatNormalScale)),e.iridescenceMap!==void 0&&(this.iridescenceMap=t[e.iridescenceMap]||null),e.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=t[e.iridescenceThicknessMap]||null),e.transmissionMap!==void 0&&(this.transmissionMap=t[e.transmissionMap]||null),e.thicknessMap!==void 0&&(this.thicknessMap=t[e.thicknessMap]||null),e.anisotropyMap!==void 0&&(this.anisotropyMap=t[e.anisotropyMap]||null),e.sheenColorMap!==void 0&&(this.sheenColorMap=t[e.sheenColorMap]||null),e.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=t[e.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;let t=e.clippingPlanes,n=null;if(t!==null){let s=t.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}},Sr=class extends Mn{constructor(e){super(),this.isSpriteMaterial=!0,this.type="SpriteMaterial",this.color=new Pe(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.rotation=e.rotation,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}},rr,Jr=new R,ar=new R,or=new R,lr=new Z,jr=new Z,sp=new st,Lo=new R,Kr=new R,Uo=new R,Jd=new Z,kh=new Z,jd=new Z,ga=class extends ft{constructor(e=new Sr){if(super(),this.isSprite=!0,this.type="Sprite",rr===void 0){rr=new ut;let t=new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),n=new ma(t,5);rr.setIndex([0,1,2,0,2,3]),rr.setAttribute("position",new Ln(n,3,0,!1)),rr.setAttribute("uv",new Ln(n,2,3,!1))}this.geometry=rr,this.material=e,this.center=new Z(.5,.5),this.count=1}raycast(e,t){e.camera===null&&$e('Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.'),ar.setFromMatrixScale(this.matrixWorld),sp.copy(e.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(e.camera.matrixWorldInverse,this.matrixWorld),or.setFromMatrixPosition(this.modelViewMatrix),e.camera.isPerspectiveCamera&&this.material.sizeAttenuation===!1&&ar.multiplyScalar(-or.z);let n=this.material.rotation,s,r;n!==0&&(r=Math.cos(n),s=Math.sin(n));let a=this.center;No(Lo.set(-.5,-.5,0),or,a,ar,s,r),No(Kr.set(.5,-.5,0),or,a,ar,s,r),No(Uo.set(.5,.5,0),or,a,ar,s,r),Jd.set(0,0),kh.set(1,0),jd.set(1,1);let o=e.ray.intersectTriangle(Lo,Kr,Uo,!1,Jr);if(o===null&&(No(Kr.set(-.5,.5,0),or,a,ar,s,r),kh.set(0,1),o=e.ray.intersectTriangle(Lo,Uo,Kr,!1,Jr),o===null))return;let c=e.ray.origin.distanceTo(Jr);c<e.near||c>e.far||t.push({distance:c,point:Jr.clone(),uv:hi.getInterpolation(Jr,Lo,Kr,Uo,Jd,kh,jd,new Z),face:null,object:this})}copy(e,t){return super.copy(e,t),e.center!==void 0&&this.center.copy(e.center),this.material=e.material,this}};function No(i,e,t,n,s,r){lr.subVectors(i,t).addScalar(.5).multiply(n),s!==void 0?(jr.x=r*lr.x-s*lr.y,jr.y=s*lr.x+r*lr.y):jr.copy(lr),i.copy(e),i.x+=jr.x,i.y+=jr.y,i.applyMatrix4(sp)}var Pi=new R,Hh=new R,Fo=new R,es=new R,Vh=new R,Oo=new R,Gh=new R,Di=class{constructor(e=new R,t=new R(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Pi)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);let n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){let t=Pi.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(Pi.copy(this.origin).addScaledVector(this.direction,t),Pi.distanceToSquared(e))}distanceSqToSegment(e,t,n,s){Hh.copy(e).add(t).multiplyScalar(.5),Fo.copy(t).sub(e).normalize(),es.copy(this.origin).sub(Hh);let r=e.distanceTo(t)*.5,a=-this.direction.dot(Fo),o=es.dot(this.direction),c=-es.dot(Fo),l=es.lengthSq(),h=Math.abs(1-a*a),d,u,f,g;if(h>0)if(d=a*c-o,u=a*o-c,g=r*h,d>=0)if(u>=-g)if(u<=g){let _=1/h;d*=_,u*=_,f=d*(d+a*u+2*o)+u*(a*d+u+2*c)+l}else u=r,d=Math.max(0,-(a*u+o)),f=-d*d+u*(u+2*c)+l;else u=-r,d=Math.max(0,-(a*u+o)),f=-d*d+u*(u+2*c)+l;else u<=-g?(d=Math.max(0,-(-a*r+o)),u=d>0?-r:Math.min(Math.max(-r,-c),r),f=-d*d+u*(u+2*c)+l):u<=g?(d=0,u=Math.min(Math.max(-r,-c),r),f=u*(u+2*c)+l):(d=Math.max(0,-(a*r+o)),u=d>0?r:Math.min(Math.max(-r,-c),r),f=-d*d+u*(u+2*c)+l);else u=a>0?-r:r,d=Math.max(0,-(a*u+o)),f=-d*d+u*(u+2*c)+l;return n&&n.copy(this.origin).addScaledVector(this.direction,d),s&&s.copy(Hh).addScaledVector(Fo,u),f}intersectSphere(e,t){Pi.subVectors(e.center,this.origin);let n=Pi.dot(this.direction),s=Pi.dot(Pi)-n*n,r=e.radius*e.radius;if(s>r)return null;let a=Math.sqrt(r-s),o=n-a,c=n+a;return c<0?null:o<0?this.at(c,t):this.at(o,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){let t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;let n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){let n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){let t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,s,r,a,o,c,l=1/this.direction.x,h=1/this.direction.y,d=1/this.direction.z,u=this.origin;return l>=0?(n=(e.min.x-u.x)*l,s=(e.max.x-u.x)*l):(n=(e.max.x-u.x)*l,s=(e.min.x-u.x)*l),h>=0?(r=(e.min.y-u.y)*h,a=(e.max.y-u.y)*h):(r=(e.max.y-u.y)*h,a=(e.min.y-u.y)*h),n>a||r>s||((r>n||isNaN(n))&&(n=r),(a<s||isNaN(s))&&(s=a),d>=0?(o=(e.min.z-u.z)*d,c=(e.max.z-u.z)*d):(o=(e.max.z-u.z)*d,c=(e.min.z-u.z)*d),n>c||o>s)||((o>n||n!==n)&&(n=o),(c<s||s!==s)&&(s=c),s<0)?null:this.at(n>=0?n:s,t)}intersectsBox(e){return this.intersectBox(e,Pi)!==null}intersectTriangle(e,t,n,s,r){Vh.subVectors(t,e),Oo.subVectors(n,e),Gh.crossVectors(Vh,Oo);let a=this.direction.dot(Gh),o;if(a>0){if(s)return null;o=1}else if(a<0)o=-1,a=-a;else return null;es.subVectors(this.origin,e);let c=o*this.direction.dot(Oo.crossVectors(es,Oo));if(c<0)return null;let l=o*this.direction.dot(Vh.cross(es));if(l<0||c+l>a)return null;let h=-o*es.dot(Gh);return h<0?null:this.at(h/a,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},Li=class extends Mn{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Pe(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Dn,this.combine=Jl,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}},Kd=new st,ws=new Di,Bo=new yn,Qd=new R,zo=new R,ko=new R,Ho=new R,Wh=new R,Vo=new R,ef=new R,Go=new R,et=class extends ft{constructor(e=new ut,t=new Li){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){let t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){let s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(e,t){let n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,a=n.morphTargetsRelative;t.fromBufferAttribute(s,e);let o=this.morphTargetInfluences;if(r&&o){Vo.set(0,0,0);for(let c=0,l=r.length;c<l;c++){let h=o[c],d=r[c];h!==0&&(Wh.fromBufferAttribute(d,e),a?Vo.addScaledVector(Wh,h):Vo.addScaledVector(Wh.sub(t),h))}t.add(Vo)}return t}raycast(e,t){let n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Bo.copy(n.boundingSphere),Bo.applyMatrix4(r),ws.copy(e.ray).recast(e.near),!(Bo.containsPoint(ws.origin)===!1&&(ws.intersectSphere(Bo,Qd)===null||ws.origin.distanceToSquared(Qd)>(e.far-e.near)**2))&&(Kd.copy(r).invert(),ws.copy(e.ray).applyMatrix4(Kd),!(n.boundingBox!==null&&ws.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,ws)))}_computeIntersections(e,t,n){let s,r=this.geometry,a=this.material,o=r.index,c=r.attributes.position,l=r.attributes.uv,h=r.attributes.uv1,d=r.attributes.normal,u=r.groups,f=r.drawRange;if(o!==null)if(Array.isArray(a))for(let g=0,_=u.length;g<_;g++){let p=u[g],m=a[p.materialIndex],M=Math.max(p.start,f.start),S=Math.min(o.count,Math.min(p.start+p.count,f.start+f.count));for(let y=M,T=S;y<T;y+=3){let b=o.getX(y),P=o.getX(y+1),x=o.getX(y+2);s=Wo(this,m,e,n,l,h,d,b,P,x),s&&(s.faceIndex=Math.floor(y/3),s.face.materialIndex=p.materialIndex,t.push(s))}}else{let g=Math.max(0,f.start),_=Math.min(o.count,f.start+f.count);for(let p=g,m=_;p<m;p+=3){let M=o.getX(p),S=o.getX(p+1),y=o.getX(p+2);s=Wo(this,a,e,n,l,h,d,M,S,y),s&&(s.faceIndex=Math.floor(p/3),t.push(s))}}else if(c!==void 0)if(Array.isArray(a))for(let g=0,_=u.length;g<_;g++){let p=u[g],m=a[p.materialIndex],M=Math.max(p.start,f.start),S=Math.min(c.count,Math.min(p.start+p.count,f.start+f.count));for(let y=M,T=S;y<T;y+=3){let b=y,P=y+1,x=y+2;s=Wo(this,m,e,n,l,h,d,b,P,x),s&&(s.faceIndex=Math.floor(y/3),s.face.materialIndex=p.materialIndex,t.push(s))}}else{let g=Math.max(0,f.start),_=Math.min(c.count,f.start+f.count);for(let p=g,m=_;p<m;p+=3){let M=p,S=p+1,y=p+2;s=Wo(this,a,e,n,l,h,d,M,S,y),s&&(s.faceIndex=Math.floor(p/3),t.push(s))}}}};function Rg(i,e,t,n,s,r,a,o){let c;if(e.side===tn?c=n.intersectTriangle(a,r,s,!0,o):c=n.intersectTriangle(s,r,a,e.side===Jn,o),c===null)return null;Go.copy(o),Go.applyMatrix4(i.matrixWorld);let l=t.ray.origin.distanceTo(Go);return l<t.near||l>t.far?null:{distance:l,point:Go.clone(),object:i}}function Wo(i,e,t,n,s,r,a,o,c,l){i.getVertexPosition(o,zo),i.getVertexPosition(c,ko),i.getVertexPosition(l,Ho);let h=Rg(i,e,t,n,zo,ko,Ho,ef);if(h){let d=new R;hi.getBarycoord(ef,zo,ko,Ho,d),s&&(h.uv=hi.getInterpolatedAttribute(s,o,c,l,d,new Z)),r&&(h.uv1=hi.getInterpolatedAttribute(r,o,c,l,d,new Z)),a&&(h.normal=hi.getInterpolatedAttribute(a,o,c,l,d,new R),h.normal.dot(n.direction)>0&&h.normal.multiplyScalar(-1));let u={a:o,b:c,c:l,normal:new R,materialIndex:0};hi.getNormal(zo,ko,Ho,u.normal),h.face=u,h.barycoord=d}return h}var Ui=class extends pn{constructor(e=null,t=1,n=1,s,r,a,o,c,l=Ot,h=Ot,d,u){super(null,a,o,c,l,h,s,r,d,u),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var _a=class extends Ut{constructor(e,t,n,s=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){let e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}},cr=new st,tf=new st,Xo=[],nf=new mn,Cg=new st,Qr=new et,ea=new yn,jt=class extends et{constructor(e,t,n){super(e,t),this.isInstancedMesh=!0,this.instanceMatrix=new _a(new Float32Array(n*16),16),this.instanceColor=null,this.morphTexture=null,this.count=n,this.boundingBox=null,this.boundingSphere=null;for(let s=0;s<n;s++)this.setMatrixAt(s,Cg)}computeBoundingBox(){let e=this.geometry,t=this.count;this.boundingBox===null&&(this.boundingBox=new mn),e.boundingBox===null&&e.computeBoundingBox(),this.boundingBox.makeEmpty();for(let n=0;n<t;n++)this.getMatrixAt(n,cr),nf.copy(e.boundingBox).applyMatrix4(cr),this.boundingBox.union(nf)}computeBoundingSphere(){let e=this.geometry,t=this.count;this.boundingSphere===null&&(this.boundingSphere=new yn),e.boundingSphere===null&&e.computeBoundingSphere(),this.boundingSphere.makeEmpty();for(let n=0;n<t;n++)this.getMatrixAt(n,cr),ea.copy(e.boundingSphere).applyMatrix4(cr),this.boundingSphere.union(ea)}copy(e,t){return super.copy(e,t),this.instanceMatrix.copy(e.instanceMatrix),e.morphTexture!==null&&(this.morphTexture=e.morphTexture.clone()),e.instanceColor!==null&&(this.instanceColor=e.instanceColor.clone()),this.count=e.count,e.boundingBox!==null&&(this.boundingBox=e.boundingBox.clone()),e.boundingSphere!==null&&(this.boundingSphere=e.boundingSphere.clone()),this}getColorAt(e,t){return this.instanceColor===null?t.setRGB(1,1,1):t.fromArray(this.instanceColor.array,e*3)}getMatrixAt(e,t){return t.fromArray(this.instanceMatrix.array,e*16)}getMorphAt(e,t){let n=t.morphTargetInfluences,s=this.morphTexture.source.data.data,r=n.length+1,a=e*r+1;for(let o=0;o<n.length;o++)n[o]=s[a+o]}raycast(e,t){let n=this.matrixWorld,s=this.count;if(Qr.geometry=this.geometry,Qr.material=this.material,Qr.material!==void 0&&(this.boundingSphere===null&&this.computeBoundingSphere(),ea.copy(this.boundingSphere),ea.applyMatrix4(n),e.ray.intersectsSphere(ea)!==!1))for(let r=0;r<s;r++){this.getMatrixAt(r,cr),tf.multiplyMatrices(n,cr),Qr.matrixWorld=tf,Qr.raycast(e,Xo);for(let a=0,o=Xo.length;a<o;a++){let c=Xo[a];c.instanceId=r,c.object=this,t.push(c)}Xo.length=0}}setColorAt(e,t){return this.instanceColor===null&&(this.instanceColor=new _a(new Float32Array(this.instanceMatrix.count*3).fill(1),3)),t.toArray(this.instanceColor.array,e*3),this}setMatrixAt(e,t){return t.toArray(this.instanceMatrix.array,e*16),this}setMorphAt(e,t){let n=t.morphTargetInfluences,s=n.length+1;this.morphTexture===null&&(this.morphTexture=new Ui(new Float32Array(s*this.count),s,this.count,ic,Vn));let r=this.morphTexture.source.data.data,a=0;for(let l=0;l<n.length;l++)a+=n[l];let o=this.geometry.morphTargetsRelative?1:1-a,c=s*e;return r[c]=o,r.set(n,c+1),this}updateMorphTargets(){}dispose(){this.dispatchEvent({type:"dispose"}),this.morphTexture!==null&&(this.morphTexture.dispose(),this.morphTexture=null)}},Xh=new R,Pg=new R,Ig=new Qe,zn=class{constructor(e=new R(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,s){return this.normal.set(e,t,n),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){let s=Xh.subVectors(n,t).cross(Pg.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){let e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t,n=!0){let s=e.delta(Xh),r=this.normal.dot(s);if(r===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;let a=-(e.start.dot(this.normal)+this.constant)/r;return n===!0&&(a<0||a>1)?null:t.copy(e.start).addScaledVector(s,a)}intersectsLine(e){let t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){let n=t||Ig.getNormalMatrix(e),s=this.coplanarPoint(Xh).applyMatrix4(e),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}},Ts=new yn,Dg=new Z(.5,.5),qo=new R,br=class{constructor(e=new zn,t=new zn,n=new zn,s=new zn,r=new zn,a=new zn){this.planes=[e,t,n,s,r,a]}set(e,t,n,s,r,a){let o=this.planes;return o[0].copy(e),o[1].copy(t),o[2].copy(n),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(e){let t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=$n,n=!1){let s=this.planes,r=e.elements,a=r[0],o=r[1],c=r[2],l=r[3],h=r[4],d=r[5],u=r[6],f=r[7],g=r[8],_=r[9],p=r[10],m=r[11],M=r[12],S=r[13],y=r[14],T=r[15];if(s[0].setComponents(l-a,f-h,m-g,T-M).normalize(),s[1].setComponents(l+a,f+h,m+g,T+M).normalize(),s[2].setComponents(l+o,f+d,m+_,T+S).normalize(),s[3].setComponents(l-o,f-d,m-_,T-S).normalize(),n)s[4].setComponents(c,u,p,y).normalize(),s[5].setComponents(l-c,f-u,m-p,T-y).normalize();else if(s[4].setComponents(l-c,f-u,m-p,T-y).normalize(),t===$n)s[5].setComponents(l+c,f+u,m+p,T+y).normalize();else if(t===gr)s[5].setComponents(c,u,p,y).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Ts.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{let t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Ts.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Ts)}intersectsSprite(e){Ts.center.set(0,0,0);let t=Dg.distanceTo(e.center);return Ts.radius=.7071067811865476+t,Ts.applyMatrix4(e.matrixWorld),this.intersectsSphere(Ts)}intersectsSphere(e){let t=this.planes,n=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(e){let t=this.planes;for(let n=0;n<6;n++){let s=t[n];if(qo.x=s.normal.x>0?e.max.x:e.min.x,qo.y=s.normal.y>0?e.max.y:e.min.y,qo.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(qo)<0)return!1}return!0}containsPoint(e){let t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}};var Ni=class extends Mn{constructor(e){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new Pe(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.linewidth=e.linewidth,this.linecap=e.linecap,this.linejoin=e.linejoin,this.fog=e.fog,this}},Sl=new R,bl=new R,sf=new st,ta=new Di,Yo=new yn,qh=new R,rf=new R,El=class extends ft{constructor(e=new ut,t=new Ni){super(),this.isLine=!0,this.type="Line",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}computeLineDistances(){let e=this.geometry;if(e.index===null){let t=e.attributes.position,n=[0];for(let s=1,r=t.count;s<r;s++)Sl.fromBufferAttribute(t,s-1),bl.fromBufferAttribute(t,s),n[s]=n[s-1],n[s]+=Sl.distanceTo(bl);e.setAttribute("lineDistance",new rt(n,1))}else Ze("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(e,t){let n=this.geometry,s=this.matrixWorld,r=e.params.Line.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),Yo.copy(n.boundingSphere),Yo.applyMatrix4(s),Yo.radius+=r,e.ray.intersectsSphere(Yo)===!1)return;sf.copy(s).invert(),ta.copy(e.ray).applyMatrix4(sf);let o=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=o*o,l=this.isLineSegments?2:1,h=n.index,u=n.attributes.position;if(h!==null){let f=Math.max(0,a.start),g=Math.min(h.count,a.start+a.count);for(let _=f,p=g-1;_<p;_+=l){let m=h.getX(_),M=h.getX(_+1),S=Zo(this,e,ta,c,m,M,_);S&&t.push(S)}if(this.isLineLoop){let _=h.getX(g-1),p=h.getX(f),m=Zo(this,e,ta,c,_,p,g-1);m&&t.push(m)}}else{let f=Math.max(0,a.start),g=Math.min(u.count,a.start+a.count);for(let _=f,p=g-1;_<p;_+=l){let m=Zo(this,e,ta,c,_,_+1,_);m&&t.push(m)}if(this.isLineLoop){let _=Zo(this,e,ta,c,g-1,f,g-1);_&&t.push(_)}}}updateMorphTargets(){let t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){let s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}};function Zo(i,e,t,n,s,r,a){let o=i.geometry.attributes.position;if(Sl.fromBufferAttribute(o,s),bl.fromBufferAttribute(o,r),t.distanceSqToSegment(Sl,bl,qh,rf)>n)return;qh.applyMatrix4(i.matrixWorld);let l=e.ray.origin.distanceTo(qh);if(!(l<e.near||l>e.far))return{distance:l,point:rf.clone().applyMatrix4(i.matrixWorld),index:a,face:null,faceIndex:null,barycoord:null,object:i}}var af=new R,of=new R,ns=class extends El{constructor(e,t){super(e,t),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){let e=this.geometry;if(e.index===null){let t=e.attributes.position,n=[];for(let s=0,r=t.count;s<r;s+=2)af.fromBufferAttribute(t,s),of.fromBufferAttribute(t,s+1),n[s]=s===0?0:n[s-1],n[s+1]=n[s]+af.distanceTo(of);e.setAttribute("lineDistance",new rt(n,1))}else Ze("LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}};var wl=class extends Mn{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new Pe(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}},lf=new st,su=new Di,$o=new yn,Jo=new R,xa=class extends ft{constructor(e=new ut,t=new wl){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){let n=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),$o.copy(n.boundingSphere),$o.applyMatrix4(s),$o.radius+=r,e.ray.intersectsSphere($o)===!1)return;lf.copy(s).invert(),su.copy(e.ray).applyMatrix4(lf);let o=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=o*o,l=n.index,d=n.attributes.position;if(l!==null){let u=Math.max(0,a.start),f=Math.min(l.count,a.start+a.count);for(let g=u,_=f;g<_;g++){let p=l.getX(g);Jo.fromBufferAttribute(d,p),cf(Jo,p,c,s,e,t,this)}}else{let u=Math.max(0,a.start),f=Math.min(d.count,a.start+a.count);for(let g=u,_=f;g<_;g++)Jo.fromBufferAttribute(d,g),cf(Jo,g,c,s,e,t,this)}}updateMorphTargets(){let t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){let s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}};function cf(i,e,t,n,s,r,a){let o=su.distanceSqToPoint(i);if(o<t){let c=new R;su.closestPointToPoint(i,c),c.applyMatrix4(n);let l=s.ray.origin.distanceTo(c);if(l<s.near||l>s.far)return;r.push({distance:l,distanceToRay:Math.sqrt(o),point:c,index:e,face:null,faceIndex:null,barycoord:null,object:a})}}var va=class extends pn{constructor(e=[],t=us,n,s,r,a,o,c,l,h){super(e,t,n,s,r,a,o,c,l,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}},pi=class extends pn{constructor(e,t,n,s,r,a,o,c,l){super(e,t,n,s,r,a,o,c,l),this.isCanvasTexture=!0,this.needsUpdate=!0}};var Kn=class extends pn{constructor(e,t,n=ni,s,r,a,o=Ot,c=Ot,l,h=fi,d=1){if(h!==fi&&h!==_i)throw new Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");let u={width:e,height:t,depth:d};super(u,s,r,a,o,c,h,n,l),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new vr(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){let t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}},Tl=class extends Kn{constructor(e,t=ni,n=us,s,r,a=Ot,o=Ot,c,l=fi){let h={width:e,height:e,depth:1},d=[h,h,h,h,h,h];super(e,e,t,n,s,r,a,o,c,l),this.image=d,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}},ya=class extends pn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}},Bt=class i extends ut{constructor(e=1,t=1,n=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:s,heightSegments:r,depthSegments:a};let o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);let c=[],l=[],h=[],d=[],u=0,f=0;g("z","y","x",-1,-1,n,t,e,a,r,0),g("z","y","x",1,-1,n,t,-e,a,r,1),g("x","z","y",1,1,e,n,t,s,a,2),g("x","z","y",1,-1,e,n,-t,s,a,3),g("x","y","z",1,-1,e,t,n,s,r,4),g("x","y","z",-1,-1,e,t,-n,s,r,5),this.setIndex(c),this.setAttribute("position",new rt(l,3)),this.setAttribute("normal",new rt(h,3)),this.setAttribute("uv",new rt(d,2));function g(_,p,m,M,S,y,T,b,P,x,E){let C=y/P,I=T/x,L=y/2,X=T/2,q=b/2,F=P+1,Y=x+1,W=0,ie=0,ne=new R;for(let ge=0;ge<Y;ge++){let ue=ge*I-X;for(let xe=0;xe<F;xe++){let Ne=xe*C-L;ne[_]=Ne*M,ne[p]=ue*S,ne[m]=q,l.push(ne.x,ne.y,ne.z),ne[_]=0,ne[p]=0,ne[m]=b>0?1:-1,h.push(ne.x,ne.y,ne.z),d.push(xe/P),d.push(1-ge/x),W+=1}}for(let ge=0;ge<x;ge++)for(let ue=0;ue<P;ue++){let xe=u+ue+F*ge,Ne=u+ue+F*(ge+1),it=u+(ue+1)+F*(ge+1),Xe=u+(ue+1)+F*ge;c.push(xe,Ne,Xe),c.push(Ne,it,Xe),ie+=6}o.addGroup(f,ie,E),f+=ie,u+=W}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new i(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}};var Qn=class i extends ut{constructor(e=1,t=1,n=1,s=32,r=1,a=!1,o=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:s,heightSegments:r,openEnded:a,thetaStart:o,thetaLength:c};let l=this;s=Math.floor(s),r=Math.floor(r);let h=[],d=[],u=[],f=[],g=0,_=[],p=n/2,m=0;M(),a===!1&&(e>0&&S(!0),t>0&&S(!1)),this.setIndex(h),this.setAttribute("position",new rt(d,3)),this.setAttribute("normal",new rt(u,3)),this.setAttribute("uv",new rt(f,2));function M(){let y=new R,T=new R,b=0,P=(t-e)/n;for(let x=0;x<=r;x++){let E=[],C=x/r,I=C*(t-e)+e;for(let L=0;L<=s;L++){let X=L/s,q=X*c+o,F=Math.sin(q),Y=Math.cos(q);T.x=I*F,T.y=-C*n+p,T.z=I*Y,d.push(T.x,T.y,T.z),y.set(F,P,Y).normalize(),u.push(y.x,y.y,y.z),f.push(X,1-C),E.push(g++)}_.push(E)}for(let x=0;x<s;x++)for(let E=0;E<r;E++){let C=_[E][x],I=_[E+1][x],L=_[E+1][x+1],X=_[E][x+1];(e>0||E!==0)&&(h.push(C,I,X),b+=3),(t>0||E!==r-1)&&(h.push(I,L,X),b+=3)}l.addGroup(m,b,0),m+=b}function S(y){let T=g,b=new Z,P=new R,x=0,E=y===!0?e:t,C=y===!0?1:-1;for(let L=1;L<=s;L++)d.push(0,p*C,0),u.push(0,C,0),f.push(.5,.5),g++;let I=g;for(let L=0;L<=s;L++){let q=L/s*c+o,F=Math.cos(q),Y=Math.sin(q);P.x=E*Y,P.y=p*C,P.z=E*F,d.push(P.x,P.y,P.z),u.push(0,C,0),b.x=F*.5+.5,b.y=Y*.5*C+.5,f.push(b.x,b.y),g++}for(let L=0;L<s;L++){let X=T+L,q=I+L;y===!0?h.push(q,q+1,X):h.push(q+1,q,X),x+=3}l.addGroup(m,x,y===!0?1:2),m+=x}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new i(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}},Er=class i extends Qn{constructor(e=1,t=1,n=32,s=1,r=!1,a=0,o=Math.PI*2){super(0,e,t,n,s,r,a,o),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:a,thetaLength:o}}static fromJSON(e){return new i(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}},Ma=class i extends ut{constructor(e=[],t=[],n=1,s=0){super(),this.type="PolyhedronGeometry",this.parameters={vertices:e,indices:t,radius:n,detail:s};let r=[],a=[];o(s),l(n),h(),this.setAttribute("position",new rt(r,3)),this.setAttribute("normal",new rt(r.slice(),3)),this.setAttribute("uv",new rt(a,2)),s===0?this.computeVertexNormals():this.normalizeNormals();function o(M){let S=new R,y=new R,T=new R;for(let b=0;b<t.length;b+=3)f(t[b+0],S),f(t[b+1],y),f(t[b+2],T),c(S,y,T,M)}function c(M,S,y,T){let b=T+1,P=[];for(let x=0;x<=b;x++){P[x]=[];let E=M.clone().lerp(y,x/b),C=S.clone().lerp(y,x/b),I=b-x;for(let L=0;L<=I;L++)L===0&&x===b?P[x][L]=E:P[x][L]=E.clone().lerp(C,L/I)}for(let x=0;x<b;x++)for(let E=0;E<2*(b-x)-1;E++){let C=Math.floor(E/2);E%2===0?(u(P[x][C+1]),u(P[x+1][C]),u(P[x][C])):(u(P[x][C+1]),u(P[x+1][C+1]),u(P[x+1][C]))}}function l(M){let S=new R;for(let y=0;y<r.length;y+=3)S.x=r[y+0],S.y=r[y+1],S.z=r[y+2],S.normalize().multiplyScalar(M),r[y+0]=S.x,r[y+1]=S.y,r[y+2]=S.z}function h(){let M=new R;for(let S=0;S<r.length;S+=3){M.x=r[S+0],M.y=r[S+1],M.z=r[S+2];let y=p(M)/2/Math.PI+.5,T=m(M)/Math.PI+.5;a.push(y,1-T)}g(),d()}function d(){for(let M=0;M<a.length;M+=6){let S=a[M+0],y=a[M+2],T=a[M+4],b=Math.max(S,y,T),P=Math.min(S,y,T);b>.9&&P<.1&&(S<.2&&(a[M+0]+=1),y<.2&&(a[M+2]+=1),T<.2&&(a[M+4]+=1))}}function u(M){r.push(M.x,M.y,M.z)}function f(M,S){let y=M*3;S.x=e[y+0],S.y=e[y+1],S.z=e[y+2]}function g(){let M=new R,S=new R,y=new R,T=new R,b=new Z,P=new Z,x=new Z;for(let E=0,C=0;E<r.length;E+=9,C+=6){M.set(r[E+0],r[E+1],r[E+2]),S.set(r[E+3],r[E+4],r[E+5]),y.set(r[E+6],r[E+7],r[E+8]),b.set(a[C+0],a[C+1]),P.set(a[C+2],a[C+3]),x.set(a[C+4],a[C+5]),T.copy(M).add(S).add(y).divideScalar(3);let I=p(T);_(b,C+0,M,I),_(P,C+2,S,I),_(x,C+4,y,I)}}function _(M,S,y,T){T<0&&M.x===1&&(a[S]=M.x-1),y.x===0&&y.z===0&&(a[S]=T/2/Math.PI+.5)}function p(M){return Math.atan2(M.z,-M.x)}function m(M){return Math.atan2(-M.y,Math.sqrt(M.x*M.x+M.z*M.z))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new i(e.vertices,e.indices,e.radius,e.detail)}},Sa=class i extends Ma{constructor(e=1,t=0){let n=(1+Math.sqrt(5))/2,s=1/n,r=[-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,0,-s,-n,0,-s,n,0,s,-n,0,s,n,-s,-n,0,-s,n,0,s,-n,0,s,n,0,-n,0,-s,n,0,-s,-n,0,s,n,0,s],a=[3,11,7,3,7,15,3,15,13,7,19,17,7,17,6,7,6,15,17,4,8,17,8,10,17,10,6,8,0,16,8,16,2,8,2,10,0,12,1,0,1,18,0,18,16,6,10,2,6,2,13,6,13,15,2,16,18,2,18,3,2,3,13,18,1,9,18,9,11,18,11,3,4,14,12,4,12,0,4,0,8,11,9,5,11,5,19,11,19,7,19,5,14,19,14,4,19,4,17,1,12,14,1,14,5,1,5,9];super(r,a,e,t),this.type="DodecahedronGeometry",this.parameters={radius:e,detail:t}}static fromJSON(e){return new i(e.radius,e.detail)}},jo=new R,Ko=new R,Yh=new R,Qo=new hi,ba=class extends ut{constructor(e=null,t=1){if(super(),this.type="EdgesGeometry",this.parameters={geometry:e,thresholdAngle:t},e!==null){let s=Math.pow(10,4),r=Math.cos(pr*t),a=e.getIndex(),o=e.getAttribute("position"),c=a?a.count:o.count,l=[0,0,0],h=["a","b","c"],d=new Array(3),u={},f=[];for(let g=0;g<c;g+=3){a?(l[0]=a.getX(g),l[1]=a.getX(g+1),l[2]=a.getX(g+2)):(l[0]=g,l[1]=g+1,l[2]=g+2);let{a:_,b:p,c:m}=Qo;if(_.fromBufferAttribute(o,l[0]),p.fromBufferAttribute(o,l[1]),m.fromBufferAttribute(o,l[2]),Qo.getNormal(Yh),d[0]=`${Math.round(_.x*s)},${Math.round(_.y*s)},${Math.round(_.z*s)}`,d[1]=`${Math.round(p.x*s)},${Math.round(p.y*s)},${Math.round(p.z*s)}`,d[2]=`${Math.round(m.x*s)},${Math.round(m.y*s)},${Math.round(m.z*s)}`,!(d[0]===d[1]||d[1]===d[2]||d[2]===d[0]))for(let M=0;M<3;M++){let S=(M+1)%3,y=d[M],T=d[S],b=Qo[h[M]],P=Qo[h[S]],x=`${y}_${T}`,E=`${T}_${y}`;E in u&&u[E]?(Yh.dot(u[E].normal)<=r&&(f.push(b.x,b.y,b.z),f.push(P.x,P.y,P.z)),u[E]=null):x in u||(u[x]={index0:l[M],index1:l[S],normal:Yh.clone()})}}for(let g in u)if(u[g]){let{index0:_,index1:p}=u[g];jo.fromBufferAttribute(o,_),Ko.fromBufferAttribute(o,p),f.push(jo.x,jo.y,jo.z),f.push(Ko.x,Ko.y,Ko.z)}this.setAttribute("position",new rt(f,3))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}},Un=class{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){Ze("Curve: .getPoint() not implemented.")}getPointAt(e,t){let n=this.getUtoTmapping(e);return this.getPoint(n,t)}getPoints(e=5){let t=[];for(let n=0;n<=e;n++)t.push(this.getPoint(n/e));return t}getSpacedPoints(e=5){let t=[];for(let n=0;n<=e;n++)t.push(this.getPointAt(n/e));return t}getLength(){let e=this.getLengths();return e[e.length-1]}getLengths(e=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===e+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;let t=[],n,s=this.getPoint(0),r=0;t.push(0);for(let a=1;a<=e;a++)n=this.getPoint(a/e),r+=n.distanceTo(s),t.push(r),s=n;return this.cacheArcLengths=t,t}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(e,t=null){let n=this.getLengths(),s=0,r=n.length,a;t?a=t:a=e*n[r-1];let o=0,c=r-1,l;for(;o<=c;)if(s=Math.floor(o+(c-o)/2),l=n[s]-a,l<0)o=s+1;else if(l>0)c=s-1;else{c=s;break}if(s=c,n[s]===a)return s/(r-1);let h=n[s],u=n[s+1]-h,f=(a-h)/u;return(s+f)/(r-1)}getTangent(e,t){let s=e-1e-4,r=e+1e-4;s<0&&(s=0),r>1&&(r=1);let a=this.getPoint(s),o=this.getPoint(r),c=t||(a.isVector2?new Z:new R);return c.copy(o).sub(a).normalize(),c}getTangentAt(e,t){let n=this.getUtoTmapping(e);return this.getTangent(n,t)}computeFrenetFrames(e,t=!1){let n=new R,s=[],r=[],a=[],o=new R,c=new st;for(let f=0;f<=e;f++){let g=f/e;s[f]=this.getTangentAt(g,new R)}r[0]=new R,a[0]=new R;let l=Number.MAX_VALUE,h=Math.abs(s[0].x),d=Math.abs(s[0].y),u=Math.abs(s[0].z);h<=l&&(l=h,n.set(1,0,0)),d<=l&&(l=d,n.set(0,1,0)),u<=l&&n.set(0,0,1),o.crossVectors(s[0],n).normalize(),r[0].crossVectors(s[0],o),a[0].crossVectors(s[0],r[0]);for(let f=1;f<=e;f++){if(r[f]=r[f-1].clone(),a[f]=a[f-1].clone(),o.crossVectors(s[f-1],s[f]),o.length()>Number.EPSILON){o.normalize();let g=Math.acos(je(s[f-1].dot(s[f]),-1,1));r[f].applyMatrix4(c.makeRotationAxis(o,g))}a[f].crossVectors(s[f],r[f])}if(t===!0){let f=Math.acos(je(r[0].dot(r[e]),-1,1));f/=e,s[0].dot(o.crossVectors(r[0],r[e]))>0&&(f=-f);for(let g=1;g<=e;g++)r[g].applyMatrix4(c.makeRotationAxis(s[g],f*g)),a[g].crossVectors(s[g],r[g])}return{tangents:s,normals:r,binormals:a}}clone(){return new this.constructor().copy(this)}copy(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}toJSON(){let e={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return e.arcLengthDivisions=this.arcLengthDivisions,e.type=this.type,e}fromJSON(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}},wr=class extends Un{constructor(e=0,t=0,n=1,s=1,r=0,a=Math.PI*2,o=!1,c=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=e,this.aY=t,this.xRadius=n,this.yRadius=s,this.aStartAngle=r,this.aEndAngle=a,this.aClockwise=o,this.aRotation=c}getPoint(e,t=new Z){let n=t,s=Math.PI*2,r=this.aEndAngle-this.aStartAngle,a=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=s;for(;r>s;)r-=s;r<Number.EPSILON&&(a?r=0:r=s),this.aClockwise===!0&&!a&&(r===s?r=-s:r=r-s);let o=this.aStartAngle+e*r,c=this.aX+this.xRadius*Math.cos(o),l=this.aY+this.yRadius*Math.sin(o);if(this.aRotation!==0){let h=Math.cos(this.aRotation),d=Math.sin(this.aRotation),u=c-this.aX,f=l-this.aY;c=u*h-f*d+this.aX,l=u*d+f*h+this.aY}return n.set(c,l)}copy(e){return super.copy(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}toJSON(){let e=super.toJSON();return e.aX=this.aX,e.aY=this.aY,e.xRadius=this.xRadius,e.yRadius=this.yRadius,e.aStartAngle=this.aStartAngle,e.aEndAngle=this.aEndAngle,e.aClockwise=this.aClockwise,e.aRotation=this.aRotation,e}fromJSON(e){return super.fromJSON(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}},Al=class extends wr{constructor(e,t,n,s,r,a){super(e,t,n,n,s,r,a),this.isArcCurve=!0,this.type="ArcCurve"}};function Au(){let i=0,e=0,t=0,n=0;function s(r,a,o,c){i=r,e=o,t=-3*r+3*a-2*o-c,n=2*r-2*a+o+c}return{initCatmullRom:function(r,a,o,c,l){s(a,o,l*(o-r),l*(c-a))},initNonuniformCatmullRom:function(r,a,o,c,l,h,d){let u=(a-r)/l-(o-r)/(l+h)+(o-a)/h,f=(o-a)/h-(c-a)/(h+d)+(c-o)/d;u*=h,f*=h,s(a,o,u,f)},calc:function(r){let a=r*r,o=a*r;return i+e*r+t*a+n*o}}}var hf=new R,uf=new R,Zh=new Au,$h=new Au,Jh=new Au,Rl=class extends Un{constructor(e=[],t=!1,n="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=e,this.closed=t,this.curveType=n,this.tension=s}getPoint(e,t=new R){let n=t,s=this.points,r=s.length,a=(r-(this.closed?0:1))*e,o=Math.floor(a),c=a-o;this.closed?o+=o>0?0:(Math.floor(Math.abs(o)/r)+1)*r:c===0&&o===r-1&&(o=r-2,c=1);let l,h;this.closed||o>0?l=s[(o-1)%r]:(uf.subVectors(s[0],s[1]).add(s[0]),l=uf);let d=s[o%r],u=s[(o+1)%r];if(this.closed||o+2<r?h=s[(o+2)%r]:(hf.subVectors(s[r-1],s[r-2]).add(s[r-1]),h=hf),this.curveType==="centripetal"||this.curveType==="chordal"){let f=this.curveType==="chordal"?.5:.25,g=Math.pow(l.distanceToSquared(d),f),_=Math.pow(d.distanceToSquared(u),f),p=Math.pow(u.distanceToSquared(h),f);_<1e-4&&(_=1),g<1e-4&&(g=_),p<1e-4&&(p=_),Zh.initNonuniformCatmullRom(l.x,d.x,u.x,h.x,g,_,p),$h.initNonuniformCatmullRom(l.y,d.y,u.y,h.y,g,_,p),Jh.initNonuniformCatmullRom(l.z,d.z,u.z,h.z,g,_,p)}else this.curveType==="catmullrom"&&(Zh.initCatmullRom(l.x,d.x,u.x,h.x,this.tension),$h.initCatmullRom(l.y,d.y,u.y,h.y,this.tension),Jh.initCatmullRom(l.z,d.z,u.z,h.z,this.tension));return n.set(Zh.calc(c),$h.calc(c),Jh.calc(c)),n}copy(e){super.copy(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){let s=e.points[t];this.points.push(s.clone())}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}toJSON(){let e=super.toJSON();e.points=[];for(let t=0,n=this.points.length;t<n;t++){let s=this.points[t];e.points.push(s.toArray())}return e.closed=this.closed,e.curveType=this.curveType,e.tension=this.tension,e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){let s=e.points[t];this.points.push(new R().fromArray(s))}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}};function df(i,e,t,n,s){let r=(n-e)*.5,a=(s-t)*.5,o=i*i,c=i*o;return(2*t-2*n+r+a)*c+(-3*t+3*n-2*r-a)*o+r*i+t}function Lg(i,e){let t=1-i;return t*t*e}function Ug(i,e){return 2*(1-i)*i*e}function Ng(i,e){return i*i*e}function sa(i,e,t,n){return Lg(i,e)+Ug(i,t)+Ng(i,n)}function Fg(i,e){let t=1-i;return t*t*t*e}function Og(i,e){let t=1-i;return 3*t*t*i*e}function Bg(i,e){return 3*(1-i)*i*i*e}function zg(i,e){return i*i*i*e}function ra(i,e,t,n,s){return Fg(i,e)+Og(i,t)+Bg(i,n)+zg(i,s)}var Ea=class extends Un{constructor(e=new Z,t=new Z,n=new Z,s=new Z){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=e,this.v1=t,this.v2=n,this.v3=s}getPoint(e,t=new Z){let n=t,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return n.set(ra(e,s.x,r.x,a.x,o.x),ra(e,s.y,r.y,a.y,o.y)),n}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}},Cl=class extends Un{constructor(e=new R,t=new R,n=new R,s=new R){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=e,this.v1=t,this.v2=n,this.v3=s}getPoint(e,t=new R){let n=t,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return n.set(ra(e,s.x,r.x,a.x,o.x),ra(e,s.y,r.y,a.y,o.y),ra(e,s.z,r.z,a.z,o.z)),n}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}},wa=class extends Un{constructor(e=new Z,t=new Z){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=e,this.v2=t}getPoint(e,t=new Z){let n=t;return e===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(e).add(this.v1)),n}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new Z){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Pl=class extends Un{constructor(e=new R,t=new R){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=e,this.v2=t}getPoint(e,t=new R){let n=t;return e===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(e).add(this.v1)),n}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new R){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Ta=class extends Un{constructor(e=new Z,t=new Z,n=new Z){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=e,this.v1=t,this.v2=n}getPoint(e,t=new Z){let n=t,s=this.v0,r=this.v1,a=this.v2;return n.set(sa(e,s.x,r.x,a.x),sa(e,s.y,r.y,a.y)),n}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Il=class extends Un{constructor(e=new R,t=new R,n=new R){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=e,this.v1=t,this.v2=n}getPoint(e,t=new R){let n=t,s=this.v0,r=this.v1,a=this.v2;return n.set(sa(e,s.x,r.x,a.x),sa(e,s.y,r.y,a.y),sa(e,s.z,r.z,a.z)),n}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Aa=class extends Un{constructor(e=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=e}getPoint(e,t=new Z){let n=t,s=this.points,r=(s.length-1)*e,a=Math.floor(r),o=r-a,c=s[a===0?a:a-1],l=s[a],h=s[a>s.length-2?s.length-1:a+1],d=s[a>s.length-3?s.length-1:a+2];return n.set(df(o,c.x,l.x,h.x,d.x),df(o,c.y,l.y,h.y,d.y)),n}copy(e){super.copy(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){let s=e.points[t];this.points.push(s.clone())}return this}toJSON(){let e=super.toJSON();e.points=[];for(let t=0,n=this.points.length;t<n;t++){let s=this.points[t];e.points.push(s.toArray())}return e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){let s=e.points[t];this.points.push(new Z().fromArray(s))}return this}},ru=Object.freeze({__proto__:null,ArcCurve:Al,CatmullRomCurve3:Rl,CubicBezierCurve:Ea,CubicBezierCurve3:Cl,EllipseCurve:wr,LineCurve:wa,LineCurve3:Pl,QuadraticBezierCurve:Ta,QuadraticBezierCurve3:Il,SplineCurve:Aa}),Dl=class extends Un{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(e){this.curves.push(e)}closePath(){let e=this.curves[0].getPoint(0),t=this.curves[this.curves.length-1].getPoint(1);if(!e.equals(t)){let n=e.isVector2===!0?"LineCurve":"LineCurve3";this.curves.push(new ru[n](t,e))}return this}getPoint(e,t){let n=e*this.getLength(),s=this.getCurveLengths(),r=0;for(;r<s.length;){if(s[r]>=n){let a=s[r]-n,o=this.curves[r],c=o.getLength(),l=c===0?0:1-a/c;return o.getPointAt(l,t)}r++}return null}getLength(){let e=this.getCurveLengths();return e[e.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;let e=[],t=0;for(let n=0,s=this.curves.length;n<s;n++)t+=this.curves[n].getLength(),e.push(t);return this.cacheLengths=e,e}getSpacedPoints(e=40){let t=[];for(let n=0;n<=e;n++)t.push(this.getPoint(n/e));return this.autoClose&&t.push(t[0]),t}getPoints(e=12){let t=[],n;for(let s=0,r=this.curves;s<r.length;s++){let a=r[s],o=a.isEllipseCurve?e*2:a.isLineCurve||a.isLineCurve3?1:a.isSplineCurve?e*a.points.length:e,c=a.getPoints(o);for(let l=0;l<c.length;l++){let h=c[l];n&&n.equals(h)||(t.push(h),n=h)}}return this.autoClose&&t.length>1&&!t[t.length-1].equals(t[0])&&t.push(t[0]),t}copy(e){super.copy(e),this.curves=[];for(let t=0,n=e.curves.length;t<n;t++){let s=e.curves[t];this.curves.push(s.clone())}return this.autoClose=e.autoClose,this}toJSON(){let e=super.toJSON();e.autoClose=this.autoClose,e.curves=[];for(let t=0,n=this.curves.length;t<n;t++){let s=this.curves[t];e.curves.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.autoClose=e.autoClose,this.curves=[];for(let t=0,n=e.curves.length;t<n;t++){let s=e.curves[t];this.curves.push(new ru[s.type]().fromJSON(s))}return this}},mi=class extends Dl{constructor(e){super(),this.type="Path",this.currentPoint=new Z,e&&this.setFromPoints(e)}setFromPoints(e){this.moveTo(e[0].x,e[0].y);for(let t=1,n=e.length;t<n;t++)this.lineTo(e[t].x,e[t].y);return this}moveTo(e,t){return this.currentPoint.set(e,t),this}lineTo(e,t){let n=new wa(this.currentPoint.clone(),new Z(e,t));return this.curves.push(n),this.currentPoint.set(e,t),this}quadraticCurveTo(e,t,n,s){let r=new Ta(this.currentPoint.clone(),new Z(e,t),new Z(n,s));return this.curves.push(r),this.currentPoint.set(n,s),this}bezierCurveTo(e,t,n,s,r,a){let o=new Ea(this.currentPoint.clone(),new Z(e,t),new Z(n,s),new Z(r,a));return this.curves.push(o),this.currentPoint.set(r,a),this}splineThru(e){let t=[this.currentPoint.clone()].concat(e),n=new Aa(t);return this.curves.push(n),this.currentPoint.copy(e[e.length-1]),this}arc(e,t,n,s,r,a){let o=this.currentPoint.x,c=this.currentPoint.y;return this.absarc(e+o,t+c,n,s,r,a),this}absarc(e,t,n,s,r,a){return this.absellipse(e,t,n,n,s,r,a),this}ellipse(e,t,n,s,r,a,o,c){let l=this.currentPoint.x,h=this.currentPoint.y;return this.absellipse(e+l,t+h,n,s,r,a,o,c),this}absellipse(e,t,n,s,r,a,o,c){let l=new wr(e,t,n,s,r,a,o,c);if(this.curves.length>0){let d=l.getPoint(0);d.equals(this.currentPoint)||this.lineTo(d.x,d.y)}this.curves.push(l);let h=l.getPoint(1);return this.currentPoint.copy(h),this}copy(e){return super.copy(e),this.currentPoint.copy(e.currentPoint),this}toJSON(){let e=super.toJSON();return e.currentPoint=this.currentPoint.toArray(),e}fromJSON(e){return super.fromJSON(e),this.currentPoint.fromArray(e.currentPoint),this}},gi=class extends mi{constructor(e){super(e),this.uuid=di(),this.type="Shape",this.holes=[]}getPointsHoles(e){let t=[];for(let n=0,s=this.holes.length;n<s;n++)t[n]=this.holes[n].getPoints(e);return t}extractPoints(e){return{shape:this.getPoints(e),holes:this.getPointsHoles(e)}}copy(e){super.copy(e),this.holes=[];for(let t=0,n=e.holes.length;t<n;t++){let s=e.holes[t];this.holes.push(s.clone())}return this}toJSON(){let e=super.toJSON();e.uuid=this.uuid,e.holes=[];for(let t=0,n=this.holes.length;t<n;t++){let s=this.holes[t];e.holes.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.uuid=e.uuid,this.holes=[];for(let t=0,n=e.holes.length;t<n;t++){let s=e.holes[t];this.holes.push(new mi().fromJSON(s))}return this}};function kg(i,e,t=2){let n=e&&e.length,s=n?e[0]*t:i.length,r=rp(i,0,s,t,!0),a=[];if(!r||r.next===r.prev)return a;let o,c,l;if(n&&(r=Xg(i,e,r,t)),i.length>80*t){o=i[0],c=i[1];let h=o,d=c;for(let u=t;u<s;u+=t){let f=i[u],g=i[u+1];f<o&&(o=f),g<c&&(c=g),f>h&&(h=f),g>d&&(d=g)}l=Math.max(h-o,d-c),l=l!==0?32767/l:0}return Ra(r,a,t,o,c,l,0),a}function rp(i,e,t,n,s){let r;if(s===n0(i,e,t,n)>0)for(let a=e;a<t;a+=n)r=ff(a/n|0,i[a],i[a+1],r);else for(let a=t-n;a>=e;a-=n)r=ff(a/n|0,i[a],i[a+1],r);return r&&Tr(r,r.next)&&(Pa(r),r=r.next),r}function Ls(i,e){if(!i)return i;e||(e=i);let t=i,n;do if(n=!1,!t.steiner&&(Tr(t,t.next)||Ct(t.prev,t,t.next)===0)){if(Pa(t),t=e=t.prev,t===t.next)break;n=!0}else t=t.next;while(n||t!==e);return e}function Ra(i,e,t,n,s,r,a){if(!i)return;!a&&r&&Jg(i,n,s,r);let o=i;for(;i.prev!==i.next;){let c=i.prev,l=i.next;if(r?Vg(i,n,s,r):Hg(i)){e.push(c.i,i.i,l.i),Pa(i),i=l.next,o=l.next;continue}if(i=l,i===o){a?a===1?(i=Gg(Ls(i),e),Ra(i,e,t,n,s,r,2)):a===2&&Wg(i,e,t,n,s,r):Ra(Ls(i),e,t,n,s,r,1);break}}}function Hg(i){let e=i.prev,t=i,n=i.next;if(Ct(e,t,n)>=0)return!1;let s=e.x,r=t.x,a=n.x,o=e.y,c=t.y,l=n.y,h=Math.min(s,r,a),d=Math.min(o,c,l),u=Math.max(s,r,a),f=Math.max(o,c,l),g=n.next;for(;g!==e;){if(g.x>=h&&g.x<=u&&g.y>=d&&g.y<=f&&na(s,o,r,c,a,l,g.x,g.y)&&Ct(g.prev,g,g.next)>=0)return!1;g=g.next}return!0}function Vg(i,e,t,n){let s=i.prev,r=i,a=i.next;if(Ct(s,r,a)>=0)return!1;let o=s.x,c=r.x,l=a.x,h=s.y,d=r.y,u=a.y,f=Math.min(o,c,l),g=Math.min(h,d,u),_=Math.max(o,c,l),p=Math.max(h,d,u),m=au(f,g,e,t,n),M=au(_,p,e,t,n),S=i.prevZ,y=i.nextZ;for(;S&&S.z>=m&&y&&y.z<=M;){if(S.x>=f&&S.x<=_&&S.y>=g&&S.y<=p&&S!==s&&S!==a&&na(o,h,c,d,l,u,S.x,S.y)&&Ct(S.prev,S,S.next)>=0||(S=S.prevZ,y.x>=f&&y.x<=_&&y.y>=g&&y.y<=p&&y!==s&&y!==a&&na(o,h,c,d,l,u,y.x,y.y)&&Ct(y.prev,y,y.next)>=0))return!1;y=y.nextZ}for(;S&&S.z>=m;){if(S.x>=f&&S.x<=_&&S.y>=g&&S.y<=p&&S!==s&&S!==a&&na(o,h,c,d,l,u,S.x,S.y)&&Ct(S.prev,S,S.next)>=0)return!1;S=S.prevZ}for(;y&&y.z<=M;){if(y.x>=f&&y.x<=_&&y.y>=g&&y.y<=p&&y!==s&&y!==a&&na(o,h,c,d,l,u,y.x,y.y)&&Ct(y.prev,y,y.next)>=0)return!1;y=y.nextZ}return!0}function Gg(i,e){let t=i;do{let n=t.prev,s=t.next.next;!Tr(n,s)&&op(n,t,t.next,s)&&Ca(n,s)&&Ca(s,n)&&(e.push(n.i,t.i,s.i),Pa(t),Pa(t.next),t=i=s),t=t.next}while(t!==i);return Ls(t)}function Wg(i,e,t,n,s,r){let a=i;do{let o=a.next.next;for(;o!==a.prev;){if(a.i!==o.i&&Qg(a,o)){let c=lp(a,o);a=Ls(a,a.next),c=Ls(c,c.next),Ra(a,e,t,n,s,r,0),Ra(c,e,t,n,s,r,0);return}o=o.next}a=a.next}while(a!==i)}function Xg(i,e,t,n){let s=[];for(let r=0,a=e.length;r<a;r++){let o=e[r]*n,c=r<a-1?e[r+1]*n:i.length,l=rp(i,o,c,n,!1);l===l.next&&(l.steiner=!0),s.push(Kg(l))}s.sort(qg);for(let r=0;r<s.length;r++)t=Yg(s[r],t);return t}function qg(i,e){let t=i.x-e.x;if(t===0&&(t=i.y-e.y,t===0)){let n=(i.next.y-i.y)/(i.next.x-i.x),s=(e.next.y-e.y)/(e.next.x-e.x);t=n-s}return t}function Yg(i,e){let t=Zg(i,e);if(!t)return e;let n=lp(t,i);return Ls(n,n.next),Ls(t,t.next)}function Zg(i,e){let t=e,n=i.x,s=i.y,r=-1/0,a;if(Tr(i,t))return t;do{if(Tr(i,t.next))return t.next;if(s<=t.y&&s>=t.next.y&&t.next.y!==t.y){let d=t.x+(s-t.y)*(t.next.x-t.x)/(t.next.y-t.y);if(d<=n&&d>r&&(r=d,a=t.x<t.next.x?t:t.next,d===n))return a}t=t.next}while(t!==e);if(!a)return null;let o=a,c=a.x,l=a.y,h=1/0;t=a;do{if(n>=t.x&&t.x>=c&&n!==t.x&&ap(s<l?n:r,s,c,l,s<l?r:n,s,t.x,t.y)){let d=Math.abs(s-t.y)/(n-t.x);Ca(t,i)&&(d<h||d===h&&(t.x>a.x||t.x===a.x&&$g(a,t)))&&(a=t,h=d)}t=t.next}while(t!==o);return a}function $g(i,e){return Ct(i.prev,i,e.prev)<0&&Ct(e.next,i,i.next)<0}function Jg(i,e,t,n){let s=i;do s.z===0&&(s.z=au(s.x,s.y,e,t,n)),s.prevZ=s.prev,s.nextZ=s.next,s=s.next;while(s!==i);s.prevZ.nextZ=null,s.prevZ=null,jg(s)}function jg(i){let e,t=1;do{let n=i,s;i=null;let r=null;for(e=0;n;){e++;let a=n,o=0;for(let l=0;l<t&&(o++,a=a.nextZ,!!a);l++);let c=t;for(;o>0||c>0&&a;)o!==0&&(c===0||!a||n.z<=a.z)?(s=n,n=n.nextZ,o--):(s=a,a=a.nextZ,c--),r?r.nextZ=s:i=s,s.prevZ=r,r=s;n=a}r.nextZ=null,t*=2}while(e>1);return i}function au(i,e,t,n,s){return i=(i-t)*s|0,e=(e-n)*s|0,i=(i|i<<8)&16711935,i=(i|i<<4)&252645135,i=(i|i<<2)&858993459,i=(i|i<<1)&1431655765,e=(e|e<<8)&16711935,e=(e|e<<4)&252645135,e=(e|e<<2)&858993459,e=(e|e<<1)&1431655765,i|e<<1}function Kg(i){let e=i,t=i;do(e.x<t.x||e.x===t.x&&e.y<t.y)&&(t=e),e=e.next;while(e!==i);return t}function ap(i,e,t,n,s,r,a,o){return(s-a)*(e-o)>=(i-a)*(r-o)&&(i-a)*(n-o)>=(t-a)*(e-o)&&(t-a)*(r-o)>=(s-a)*(n-o)}function na(i,e,t,n,s,r,a,o){return!(i===a&&e===o)&&ap(i,e,t,n,s,r,a,o)}function Qg(i,e){return i.next.i!==e.i&&i.prev.i!==e.i&&!e0(i,e)&&(Ca(i,e)&&Ca(e,i)&&t0(i,e)&&(Ct(i.prev,i,e.prev)||Ct(i,e.prev,e))||Tr(i,e)&&Ct(i.prev,i,i.next)>0&&Ct(e.prev,e,e.next)>0)}function Ct(i,e,t){return(e.y-i.y)*(t.x-e.x)-(e.x-i.x)*(t.y-e.y)}function Tr(i,e){return i.x===e.x&&i.y===e.y}function op(i,e,t,n){let s=tl(Ct(i,e,t)),r=tl(Ct(i,e,n)),a=tl(Ct(t,n,i)),o=tl(Ct(t,n,e));return!!(s!==r&&a!==o||s===0&&el(i,t,e)||r===0&&el(i,n,e)||a===0&&el(t,i,n)||o===0&&el(t,e,n))}function el(i,e,t){return e.x<=Math.max(i.x,t.x)&&e.x>=Math.min(i.x,t.x)&&e.y<=Math.max(i.y,t.y)&&e.y>=Math.min(i.y,t.y)}function tl(i){return i>0?1:i<0?-1:0}function e0(i,e){let t=i;do{if(t.i!==i.i&&t.next.i!==i.i&&t.i!==e.i&&t.next.i!==e.i&&op(t,t.next,i,e))return!0;t=t.next}while(t!==i);return!1}function Ca(i,e){return Ct(i.prev,i,i.next)<0?Ct(i,e,i.next)>=0&&Ct(i,i.prev,e)>=0:Ct(i,e,i.prev)<0||Ct(i,i.next,e)<0}function t0(i,e){let t=i,n=!1,s=(i.x+e.x)/2,r=(i.y+e.y)/2;do t.y>r!=t.next.y>r&&t.next.y!==t.y&&s<(t.next.x-t.x)*(r-t.y)/(t.next.y-t.y)+t.x&&(n=!n),t=t.next;while(t!==i);return n}function lp(i,e){let t=ou(i.i,i.x,i.y),n=ou(e.i,e.x,e.y),s=i.next,r=e.prev;return i.next=e,e.prev=i,t.next=s,s.prev=t,n.next=t,t.prev=n,r.next=n,n.prev=r,n}function ff(i,e,t,n){let s=ou(i,e,t);return n?(s.next=n.next,s.prev=n,n.next.prev=s,n.next=s):(s.prev=s,s.next=s),s}function Pa(i){i.next.prev=i.prev,i.prev.next=i.next,i.prevZ&&(i.prevZ.nextZ=i.nextZ),i.nextZ&&(i.nextZ.prevZ=i.prevZ)}function ou(i,e,t){return{i,x:e,y:t,prev:null,next:null,z:0,prevZ:null,nextZ:null,steiner:!1}}function n0(i,e,t,n){let s=0;for(let r=e,a=t-n;r<t;r+=n)s+=(i[a]-i[r])*(i[r+1]+i[a+1]),a=r;return s}var lu=class{static triangulate(e,t,n=2){return kg(e,t,n)}},Rs=class i{static area(e){let t=e.length,n=0;for(let s=t-1,r=0;r<t;s=r++)n+=e[s].x*e[r].y-e[r].x*e[s].y;return n*.5}static isClockWise(e){return i.area(e)<0}static triangulateShape(e,t){let n=[],s=[],r=[];pf(e),mf(n,e);let a=e.length;t.forEach(pf);for(let c=0;c<t.length;c++)s.push(a),a+=t[c].length,mf(n,t[c]);let o=lu.triangulate(n,s);for(let c=0;c<o.length;c+=3)r.push(o.slice(c,c+3));return r}};function pf(i){let e=i.length;e>2&&i[e-1].equals(i[0])&&i.pop()}function mf(i,e){for(let t=0;t<e.length;t++)i.push(e[t].x),i.push(e[t].y)}var Fi=class i extends ut{constructor(e=new gi([new Z(.5,.5),new Z(-.5,.5),new Z(-.5,-.5),new Z(.5,-.5)]),t={}){super(),this.type="ExtrudeGeometry",this.parameters={shapes:e,options:t},e=Array.isArray(e)?e:[e];let n=this,s=[],r=[];for(let o=0,c=e.length;o<c;o++){let l=e[o];a(l)}this.setAttribute("position",new rt(s,3)),this.setAttribute("uv",new rt(r,2)),this.computeVertexNormals();function a(o){let c=[],l=t.curveSegments!==void 0?t.curveSegments:12,h=t.steps!==void 0?t.steps:1,d=t.depth!==void 0?t.depth:1,u=t.bevelEnabled!==void 0?t.bevelEnabled:!0,f=t.bevelThickness!==void 0?t.bevelThickness:.2,g=t.bevelSize!==void 0?t.bevelSize:f-.1,_=t.bevelOffset!==void 0?t.bevelOffset:0,p=t.bevelSegments!==void 0?t.bevelSegments:3,m=t.extrudePath,M=t.UVGenerator!==void 0?t.UVGenerator:i0,S,y=!1,T,b,P,x;if(m){S=m.getSpacedPoints(h),y=!0,u=!1;let O=m.isCatmullRomCurve3?m.closed:!1;T=m.computeFrenetFrames(h,O),b=new R,P=new R,x=new R}u||(p=0,f=0,g=0,_=0);let E=o.extractPoints(l),C=E.shape,I=E.holes;if(!Rs.isClockWise(C)){C=C.reverse();for(let O=0,H=I.length;O<H;O++){let Q=I[O];Rs.isClockWise(Q)&&(I[O]=Q.reverse())}}function X(O){let Q=10000000000000001e-36,G=O[0];for(let V=1;V<=O.length;V++){let se=V%O.length,ce=O[se],fe=ce.x-G.x,me=ce.y-G.y,D=fe*fe+me*me,Me=Math.max(Math.abs(ce.x),Math.abs(ce.y),Math.abs(G.x),Math.abs(G.y)),Ve=Q*Me*Me;if(D<=Ve){O.splice(se,1),V--;continue}G=ce}}X(C),I.forEach(X);let q=I.length,F=C;for(let O=0;O<q;O++){let H=I[O];C=C.concat(H)}function Y(O,H,Q){return H||$e("ExtrudeGeometry: vec does not exist"),O.clone().addScaledVector(H,Q)}let W=C.length;function ie(O,H,Q){let G,V,se,ce=O.x-H.x,fe=O.y-H.y,me=Q.x-O.x,D=Q.y-O.y,Me=ce*ce+fe*fe,Ve=ce*D-fe*me;if(Math.abs(Ve)>Number.EPSILON){let A=Math.sqrt(Me),v=Math.sqrt(me*me+D*D),U=H.x-fe/A,B=H.y+ce/A,k=Q.x-D/v,pe=Q.y+me/v,_e=((k-U)*D-(pe-B)*me)/(ce*D-fe*me);G=U+ce*_e-O.x,V=B+fe*_e-O.y;let te=G*G+V*V;if(te<=2)return new Z(G,V);se=Math.sqrt(te/2)}else{let A=!1;ce>Number.EPSILON?me>Number.EPSILON&&(A=!0):ce<-Number.EPSILON?me<-Number.EPSILON&&(A=!0):Math.sign(fe)===Math.sign(D)&&(A=!0),A?(G=-fe,V=ce,se=Math.sqrt(Me)):(G=ce,V=fe,se=Math.sqrt(Me/2))}return new Z(G/se,V/se)}let ne=[];for(let O=0,H=F.length,Q=H-1,G=O+1;O<H;O++,Q++,G++)Q===H&&(Q=0),G===H&&(G=0),ne[O]=ie(F[O],F[Q],F[G]);let ge=[],ue,xe=ne.concat();for(let O=0,H=q;O<H;O++){let Q=I[O];ue=[];for(let G=0,V=Q.length,se=V-1,ce=G+1;G<V;G++,se++,ce++)se===V&&(se=0),ce===V&&(ce=0),ue[G]=ie(Q[G],Q[se],Q[ce]);ge.push(ue),xe=xe.concat(ue)}let Ne;if(p===0)Ne=Rs.triangulateShape(F,I);else{let O=[],H=[];for(let Q=0;Q<p;Q++){let G=Q/p,V=f*Math.cos(G*Math.PI/2),se=g*Math.sin(G*Math.PI/2)+_;for(let ce=0,fe=F.length;ce<fe;ce++){let me=Y(F[ce],ne[ce],se);Te(me.x,me.y,-V),G===0&&O.push(me)}for(let ce=0,fe=q;ce<fe;ce++){let me=I[ce];ue=ge[ce];let D=[];for(let Me=0,Ve=me.length;Me<Ve;Me++){let A=Y(me[Me],ue[Me],se);Te(A.x,A.y,-V),G===0&&D.push(A)}G===0&&H.push(D)}}Ne=Rs.triangulateShape(O,H)}let it=Ne.length,Xe=g+_;for(let O=0;O<W;O++){let H=u?Y(C[O],xe[O],Xe):C[O];y?(P.copy(T.normals[0]).multiplyScalar(H.x),b.copy(T.binormals[0]).multiplyScalar(H.y),x.copy(S[0]).add(P).add(b),Te(x.x,x.y,x.z)):Te(H.x,H.y,0)}for(let O=1;O<=h;O++)for(let H=0;H<W;H++){let Q=u?Y(C[H],xe[H],Xe):C[H];y?(P.copy(T.normals[O]).multiplyScalar(Q.x),b.copy(T.binormals[O]).multiplyScalar(Q.y),x.copy(S[O]).add(P).add(b),Te(x.x,x.y,x.z)):Te(Q.x,Q.y,d/h*O)}for(let O=p-1;O>=0;O--){let H=O/p,Q=f*Math.cos(H*Math.PI/2),G=g*Math.sin(H*Math.PI/2)+_;for(let V=0,se=F.length;V<se;V++){let ce=Y(F[V],ne[V],G);Te(ce.x,ce.y,d+Q)}for(let V=0,se=I.length;V<se;V++){let ce=I[V];ue=ge[V];for(let fe=0,me=ce.length;fe<me;fe++){let D=Y(ce[fe],ue[fe],G);y?Te(D.x,D.y+S[h-1].y,S[h-1].x+Q):Te(D.x,D.y,d+Q)}}}j(),he();function j(){let O=s.length/3;if(u){let H=0,Q=W*H;for(let G=0;G<it;G++){let V=Ne[G];Fe(V[2]+Q,V[1]+Q,V[0]+Q)}H=h+p*2,Q=W*H;for(let G=0;G<it;G++){let V=Ne[G];Fe(V[0]+Q,V[1]+Q,V[2]+Q)}}else{for(let H=0;H<it;H++){let Q=Ne[H];Fe(Q[2],Q[1],Q[0])}for(let H=0;H<it;H++){let Q=Ne[H];Fe(Q[0]+W*h,Q[1]+W*h,Q[2]+W*h)}}n.addGroup(O,s.length/3-O,0)}function he(){let O=s.length/3,H=0;le(F,H),H+=F.length;for(let Q=0,G=I.length;Q<G;Q++){let V=I[Q];le(V,H),H+=V.length}n.addGroup(O,s.length/3-O,1)}function le(O,H){let Q=O.length;for(;--Q>=0;){let G=Q,V=Q-1;V<0&&(V=O.length-1);for(let se=0,ce=h+p*2;se<ce;se++){let fe=W*se,me=W*(se+1),D=H+G+fe,Me=H+V+fe,Ve=H+V+me,A=H+G+me;ke(D,Me,Ve,A)}}}function Te(O,H,Q){c.push(O),c.push(H),c.push(Q)}function Fe(O,H,Q){oe(O),oe(H),oe(Q);let G=s.length/3,V=M.generateTopUV(n,s,G-3,G-2,G-1);ee(V[0]),ee(V[1]),ee(V[2])}function ke(O,H,Q,G){oe(O),oe(H),oe(G),oe(H),oe(Q),oe(G);let V=s.length/3,se=M.generateSideWallUV(n,s,V-6,V-3,V-2,V-1);ee(se[0]),ee(se[1]),ee(se[3]),ee(se[1]),ee(se[2]),ee(se[3])}function oe(O){s.push(c[O*3+0]),s.push(c[O*3+1]),s.push(c[O*3+2])}function ee(O){r.push(O.x),r.push(O.y)}}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}toJSON(){let e=super.toJSON(),t=this.parameters.shapes,n=this.parameters.options;return s0(t,n,e)}static fromJSON(e,t){let n=[];for(let r=0,a=e.shapes.length;r<a;r++){let o=t[e.shapes[r]];n.push(o)}let s=e.options.extrudePath;return s!==void 0&&(e.options.extrudePath=new ru[s.type]().fromJSON(s)),new i(n,e.options)}},i0={generateTopUV:function(i,e,t,n,s){let r=e[t*3],a=e[t*3+1],o=e[n*3],c=e[n*3+1],l=e[s*3],h=e[s*3+1];return[new Z(r,a),new Z(o,c),new Z(l,h)]},generateSideWallUV:function(i,e,t,n,s,r){let a=e[t*3],o=e[t*3+1],c=e[t*3+2],l=e[n*3],h=e[n*3+1],d=e[n*3+2],u=e[s*3],f=e[s*3+1],g=e[s*3+2],_=e[r*3],p=e[r*3+1],m=e[r*3+2];return Math.abs(o-h)<Math.abs(a-l)?[new Z(a,1-c),new Z(l,1-d),new Z(u,1-g),new Z(_,1-m)]:[new Z(o,1-c),new Z(h,1-d),new Z(f,1-g),new Z(p,1-m)]}};function s0(i,e,t){if(t.shapes=[],Array.isArray(i))for(let n=0,s=i.length;n<s;n++){let r=i[n];t.shapes.push(r.uuid)}else t.shapes.push(i.uuid);return t.options=Object.assign({},e),e.extrudePath!==void 0&&(t.options.extrudePath=e.extrudePath.toJSON()),t}var ei=class i extends Ma{constructor(e=1,t=0){let n=(1+Math.sqrt(5))/2,s=[-1,n,0,1,n,0,-1,-n,0,1,-n,0,0,-1,n,0,1,n,0,-1,-n,0,1,-n,n,0,-1,n,0,1,-n,0,-1,-n,0,1],r=[0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1];super(s,r,e,t),this.type="IcosahedronGeometry",this.parameters={radius:e,detail:t}}static fromJSON(e){return new i(e.radius,e.detail)}};var Hn=class i extends ut{constructor(e=1,t=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:s};let r=e/2,a=t/2,o=Math.floor(n),c=Math.floor(s),l=o+1,h=c+1,d=e/o,u=t/c,f=[],g=[],_=[],p=[];for(let m=0;m<h;m++){let M=m*u-a;for(let S=0;S<l;S++){let y=S*d-r;g.push(y,-M,0),_.push(0,0,1),p.push(S/o),p.push(1-m/c)}}for(let m=0;m<c;m++)for(let M=0;M<o;M++){let S=M+l*m,y=M+l*(m+1),T=M+1+l*(m+1),b=M+1+l*m;f.push(S,y,b),f.push(y,T,b)}this.setIndex(f),this.setAttribute("position",new rt(g,3)),this.setAttribute("normal",new rt(_,3)),this.setAttribute("uv",new rt(p,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new i(e.width,e.height,e.widthSegments,e.heightSegments)}},Ia=class i extends ut{constructor(e=.5,t=1,n=32,s=1,r=0,a=Math.PI*2){super(),this.type="RingGeometry",this.parameters={innerRadius:e,outerRadius:t,thetaSegments:n,phiSegments:s,thetaStart:r,thetaLength:a},n=Math.max(3,n),s=Math.max(1,s);let o=[],c=[],l=[],h=[],d=e,u=(t-e)/s,f=new R,g=new Z;for(let _=0;_<=s;_++){for(let p=0;p<=n;p++){let m=r+p/n*a;f.x=d*Math.cos(m),f.y=d*Math.sin(m),c.push(f.x,f.y,f.z),l.push(0,0,1),g.x=(f.x/t+1)/2,g.y=(f.y/t+1)/2,h.push(g.x,g.y)}d+=u}for(let _=0;_<s;_++){let p=_*(n+1);for(let m=0;m<n;m++){let M=m+p,S=M,y=M+n+1,T=M+n+2,b=M+1;o.push(S,y,b),o.push(y,T,b)}}this.setIndex(o),this.setAttribute("position",new rt(c,3)),this.setAttribute("normal",new rt(l,3)),this.setAttribute("uv",new rt(h,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new i(e.innerRadius,e.outerRadius,e.thetaSegments,e.phiSegments,e.thetaStart,e.thetaLength)}};var Da=class extends ut{constructor(e=null){if(super(),this.type="WireframeGeometry",this.parameters={geometry:e},e!==null){let t=[],n=new Set,s=new R,r=new R;if(e.index!==null){let a=e.attributes.position,o=e.index,c=e.groups;c.length===0&&(c=[{start:0,count:o.count,materialIndex:0}]);for(let l=0,h=c.length;l<h;++l){let d=c[l],u=d.start,f=d.count;for(let g=u,_=u+f;g<_;g+=3)for(let p=0;p<3;p++){let m=o.getX(g+p),M=o.getX(g+(p+1)%3);s.fromBufferAttribute(a,m),r.fromBufferAttribute(a,M),gf(s,r,n)===!0&&(t.push(s.x,s.y,s.z),t.push(r.x,r.y,r.z))}}}else{let a=e.attributes.position;for(let o=0,c=a.count/3;o<c;o++)for(let l=0;l<3;l++){let h=3*o+l,d=3*o+(l+1)%3;s.fromBufferAttribute(a,h),r.fromBufferAttribute(a,d),gf(s,r,n)===!0&&(t.push(s.x,s.y,s.z),t.push(r.x,r.y,r.z))}}this.setAttribute("position",new rt(t,3))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}};function gf(i,e,t){let n=`${i.x},${i.y},${i.z}-${e.x},${e.y},${e.z}`,s=`${e.x},${e.y},${e.z}-${i.x},${i.y},${i.z}`;return t.has(n)===!0||t.has(s)===!0?!1:(t.add(n),t.add(s),!0)}var La=class extends Mn{constructor(e){super(),this.isShadowMaterial=!0,this.type="ShadowMaterial",this.color=new Pe(0),this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.fog=e.fog,this}};function Bs(i){let e={};for(let t in i){e[t]={};for(let n in i[t]){let s=i[t][n];if(_f(s))s.isRenderTargetTexture?(Ze("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=s.clone();else if(Array.isArray(s))if(_f(s[0])){let r=[];for(let a=0,o=s.length;a<o;a++)r[a]=s[a].clone();e[t][n]=r}else e[t][n]=s.slice();else e[t][n]=s}}return e}function un(i){let e={};for(let t=0;t<i.length;t++){let n=Bs(i[t]);for(let s in n)e[s]=n[s]}return e}function _f(i){return i&&(i.isColor||i.isMatrix3||i.isMatrix4||i.isVector2||i.isVector3||i.isVector4||i.isTexture||i.isQuaternion)}function r0(i){let e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function Ru(i){let e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:ht.workingColorSpace}var gn={clone:Bs,merge:un},a0=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,o0=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,bt=class extends Mn{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=a0,this.fragmentShader=o0,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Bs(e.uniforms),this.uniformsGroups=r0(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){let t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(let s in this.uniforms){let a=this.uniforms[s].value;a&&a.isTexture?t.uniforms[s]={type:"t",value:a.toJSON(e).uuid}:a&&a.isColor?t.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?t.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?t.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?t.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?t.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?t.uniforms[s]={type:"m4",value:a.toArray()}:t.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;let n={};for(let s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}fromJSON(e,t){if(super.fromJSON(e,t),e.uniforms!==void 0)for(let n in e.uniforms){let s=e.uniforms[n];switch(this.uniforms[n]={},s.type){case"t":this.uniforms[n].value=t[s.value]||null;break;case"c":this.uniforms[n].value=new Pe().setHex(s.value);break;case"v2":this.uniforms[n].value=new Z().fromArray(s.value);break;case"v3":this.uniforms[n].value=new R().fromArray(s.value);break;case"v4":this.uniforms[n].value=new mt().fromArray(s.value);break;case"m3":this.uniforms[n].value=new Qe().fromArray(s.value);break;case"m4":this.uniforms[n].value=new st().fromArray(s.value);break;default:this.uniforms[n].value=s.value}}if(e.defines!==void 0&&(this.defines=e.defines),e.vertexShader!==void 0&&(this.vertexShader=e.vertexShader),e.fragmentShader!==void 0&&(this.fragmentShader=e.fragmentShader),e.glslVersion!==void 0&&(this.glslVersion=e.glslVersion),e.extensions!==void 0)for(let n in e.extensions)this.extensions[n]=e.extensions[n];return e.lights!==void 0&&(this.lights=e.lights),e.clipping!==void 0&&(this.clipping=e.clipping),this}},Ar=class extends bt{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}},Ke=class extends Mn{constructor(e){super(),this.isMeshStandardMaterial=!0,this.type="MeshStandardMaterial",this.defines={STANDARD:""},this.color=new Pe(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Pe(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Lr,this.normalScale=new Z(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Dn,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.defines={STANDARD:""},this.color.copy(e.color),this.roughness=e.roughness,this.metalness=e.metalness,this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.roughnessMap=e.roughnessMap,this.metalnessMap=e.metalnessMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.envMapIntensity=e.envMapIntensity,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}},Ua=class extends Ke{constructor(e){super(),this.isMeshPhysicalMaterial=!0,this.defines={STANDARD:"",PHYSICAL:""},this.type="MeshPhysicalMaterial",this.anisotropyRotation=0,this.anisotropyMap=null,this.clearcoatMap=null,this.clearcoatRoughness=0,this.clearcoatRoughnessMap=null,this.clearcoatNormalScale=new Z(1,1),this.clearcoatNormalMap=null,this.ior=1.5,Object.defineProperty(this,"reflectivity",{get:function(){return je(2.5*(this.ior-1)/(this.ior+1),0,1)},set:function(t){this.ior=(1+.4*t)/(1-.4*t)}}),this.iridescenceMap=null,this.iridescenceIOR=1.3,this.iridescenceThicknessRange=[100,400],this.iridescenceThicknessMap=null,this.sheenColor=new Pe(0),this.sheenColorMap=null,this.sheenRoughness=1,this.sheenRoughnessMap=null,this.transmissionMap=null,this.thickness=0,this.thicknessMap=null,this.attenuationDistance=1/0,this.attenuationColor=new Pe(1,1,1),this.specularIntensity=1,this.specularIntensityMap=null,this.specularColor=new Pe(1,1,1),this.specularColorMap=null,this._anisotropy=0,this._clearcoat=0,this._dispersion=0,this._iridescence=0,this._sheen=0,this._transmission=0,this.setValues(e)}get anisotropy(){return this._anisotropy}set anisotropy(e){this._anisotropy>0!=e>0&&this.version++,this._anisotropy=e}get clearcoat(){return this._clearcoat}set clearcoat(e){this._clearcoat>0!=e>0&&this.version++,this._clearcoat=e}get iridescence(){return this._iridescence}set iridescence(e){this._iridescence>0!=e>0&&this.version++,this._iridescence=e}get dispersion(){return this._dispersion}set dispersion(e){this._dispersion>0!=e>0&&this.version++,this._dispersion=e}get sheen(){return this._sheen}set sheen(e){this._sheen>0!=e>0&&this.version++,this._sheen=e}get transmission(){return this._transmission}set transmission(e){this._transmission>0!=e>0&&this.version++,this._transmission=e}copy(e){return super.copy(e),this.defines={STANDARD:"",PHYSICAL:""},this.anisotropy=e.anisotropy,this.anisotropyRotation=e.anisotropyRotation,this.anisotropyMap=e.anisotropyMap,this.clearcoat=e.clearcoat,this.clearcoatMap=e.clearcoatMap,this.clearcoatRoughness=e.clearcoatRoughness,this.clearcoatRoughnessMap=e.clearcoatRoughnessMap,this.clearcoatNormalMap=e.clearcoatNormalMap,this.clearcoatNormalScale.copy(e.clearcoatNormalScale),this.dispersion=e.dispersion,this.ior=e.ior,this.iridescence=e.iridescence,this.iridescenceMap=e.iridescenceMap,this.iridescenceIOR=e.iridescenceIOR,this.iridescenceThicknessRange=[...e.iridescenceThicknessRange],this.iridescenceThicknessMap=e.iridescenceThicknessMap,this.sheen=e.sheen,this.sheenColor.copy(e.sheenColor),this.sheenColorMap=e.sheenColorMap,this.sheenRoughness=e.sheenRoughness,this.sheenRoughnessMap=e.sheenRoughnessMap,this.transmission=e.transmission,this.transmissionMap=e.transmissionMap,this.thickness=e.thickness,this.thicknessMap=e.thicknessMap,this.attenuationDistance=e.attenuationDistance,this.attenuationColor.copy(e.attenuationColor),this.specularIntensity=e.specularIntensity,this.specularIntensityMap=e.specularIntensityMap,this.specularColor.copy(e.specularColor),this.specularColorMap=e.specularColorMap,this}};var Na=class extends Mn{constructor(e){super(),this.isMeshNormalMaterial=!0,this.type="MeshNormalMaterial",this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Lr,this.normalScale=new Z(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.flatShading=!1,this.setValues(e)}copy(e){return super.copy(e),this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.flatShading=e.flatShading,this}},Fa=class extends Mn{constructor(e){super(),this.isMeshLambertMaterial=!0,this.type="MeshLambertMaterial",this.color=new Pe(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Pe(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Lr,this.normalScale=new Z(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Dn,this.combine=Jl,this.reflectivity=1,this.envMapIntensity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.envMapIntensity=e.envMapIntensity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}},Ll=class extends Mn{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Xf,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}},Ul=class extends Mn{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}};function nl(i,e){return!i||i.constructor===e?i:typeof e.BYTES_PER_ELEMENT=="number"?new e(i):Array.prototype.slice.call(i)}var is=class{constructor(e,t,n,s){this.parameterPositions=e,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new t.constructor(n),this.sampleValues=t,this.valueSize=n,this.settings=null,this.DefaultSettings_={}}evaluate(e){let t=this.parameterPositions,n=this._cachedIndex,s=t[n],r=t[n-1];n:{e:{let a;t:{i:if(!(e<s)){for(let o=n+2;;){if(s===void 0){if(e<r)break i;return n=t.length,this._cachedIndex=n,this.copySampleValue_(n-1)}if(n===o)break;if(r=s,s=t[++n],e<s)break e}a=t.length;break t}if(!(e>=r)){let o=t[1];e<o&&(n=2,r=o);for(let c=n-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(n===c)break;if(s=r,r=t[--n-1],e>=r)break e}a=n,n=0;break t}break n}for(;n<a;){let o=n+a>>>1;e<t[o]?a=o:n=o+1}if(s=t[n],r=t[n-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return n=t.length,this._cachedIndex=n,this.copySampleValue_(n-1)}this._cachedIndex=n,this.intervalChanged_(n,r,s)}return this.interpolate_(n,r,e,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(e){let t=this.resultBuffer,n=this.sampleValues,s=this.valueSize,r=e*s;for(let a=0;a!==s;++a)t[a]=n[r+a];return t}interpolate_(){throw new Error("THREE.Interpolant: Call to abstract method.")}intervalChanged_(){}},Nl=class extends is{constructor(e,t,n,s){super(e,t,n,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:eu,endingEnd:eu}}intervalChanged_(e,t,n){let s=this.parameterPositions,r=e-2,a=e+1,o=s[r],c=s[a];if(o===void 0)switch(this.getSettings_().endingStart){case tu:r=e,o=2*t-n;break;case nu:r=s.length-2,o=t+s[r]-s[r+1];break;default:r=e,o=n}if(c===void 0)switch(this.getSettings_().endingEnd){case tu:a=e,c=2*n-t;break;case nu:a=1,c=n+s[1]-s[0];break;default:a=e-1,c=t}let l=(n-t)*.5,h=this.valueSize;this._weightPrev=l/(t-o),this._weightNext=l/(c-n),this._offsetPrev=r*h,this._offsetNext=a*h}interpolate_(e,t,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=e*o,l=c-o,h=this._offsetPrev,d=this._offsetNext,u=this._weightPrev,f=this._weightNext,g=(n-t)/(s-t),_=g*g,p=_*g,m=-u*p+2*u*_-u*g,M=(1+u)*p+(-1.5-2*u)*_+(-.5+u)*g+1,S=(-1-f)*p+(1.5+f)*_+.5*g,y=f*p-f*_;for(let T=0;T!==o;++T)r[T]=m*a[h+T]+M*a[l+T]+S*a[c+T]+y*a[d+T];return r}},Fl=class extends is{constructor(e,t,n,s){super(e,t,n,s)}interpolate_(e,t,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=e*o,l=c-o,h=(n-t)/(s-t),d=1-h;for(let u=0;u!==o;++u)r[u]=a[l+u]*d+a[c+u]*h;return r}},Ol=class extends is{constructor(e,t,n,s){super(e,t,n,s)}interpolate_(e){return this.copySampleValue_(e-1)}},Bl=class extends is{interpolate_(e,t,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=e*o,l=c-o,h=this.inTangents,d=this.outTangents;if(!h||!d){let g=(n-t)/(s-t),_=1-g;for(let p=0;p!==o;++p)r[p]=a[l+p]*_+a[c+p]*g;return r}let u=o*2,f=e-1;for(let g=0;g!==o;++g){let _=a[l+g],p=a[c+g],m=f*u+g*2,M=d[m],S=d[m+1],y=e*u+g*2,T=h[y],b=h[y+1],P=(n-t)/(s-t),x,E,C,I,L;for(let X=0;X<8;X++){x=P*P,E=x*P,C=1-P,I=C*C,L=I*C;let F=L*t+3*I*P*M+3*C*x*T+E*s-n;if(Math.abs(F)<1e-10)break;let Y=3*I*(M-t)+6*C*P*(T-M)+3*x*(s-T);if(Math.abs(Y)<1e-10)break;P=P-F/Y,P=Math.max(0,Math.min(1,P))}r[g]=L*_+3*I*P*S+3*C*x*b+E*p}return r}},Nn=class{constructor(e,t,n,s){if(e===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(t===void 0||t.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+e);this.name=e,this.times=nl(t,this.TimeBufferType),this.values=nl(n,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(e){let t=e.constructor,n;if(t.toJSON!==this.toJSON)n=t.toJSON(e);else{n={name:e.name,times:nl(e.times,Array),values:nl(e.values,Array)};let s=e.getInterpolation();s!==e.DefaultInterpolation&&(n.interpolation=s)}return n.type=e.ValueTypeName,n}InterpolantFactoryMethodDiscrete(e){return new Ol(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodLinear(e){return new Fl(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodSmooth(e){return new Nl(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodBezier(e){let t=new Bl(this.times,this.values,this.getValueSize(),e);return this.settings&&(t.inTangents=this.settings.inTangents,t.outTangents=this.settings.outTangents),t}setInterpolation(e){let t;switch(e){case aa:t=this.InterpolantFactoryMethodDiscrete;break;case _l:t=this.InterpolantFactoryMethodLinear;break;case al:t=this.InterpolantFactoryMethodSmooth;break;case Qh:t=this.InterpolantFactoryMethodBezier;break}if(t===void 0){let n="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(e!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(n);return Ze("KeyframeTrack:",n),this}return this.createInterpolant=t,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return aa;case this.InterpolantFactoryMethodLinear:return _l;case this.InterpolantFactoryMethodSmooth:return al;case this.InterpolantFactoryMethodBezier:return Qh}}getValueSize(){return this.values.length/this.times.length}shift(e){if(e!==0){let t=this.times;for(let n=0,s=t.length;n!==s;++n)t[n]+=e}return this}scale(e){if(e!==1){let t=this.times;for(let n=0,s=t.length;n!==s;++n)t[n]*=e}return this}trim(e,t){let n=this.times,s=n.length,r=0,a=s-1;for(;r!==s&&n[r]<e;)++r;for(;a!==-1&&n[a]>t;)--a;if(++a,r!==0||a!==s){r>=a&&(a=Math.max(a,1),r=a-1);let o=this.getValueSize();this.times=n.slice(r,a),this.values=this.values.slice(r*o,a*o)}return this}validate(){let e=!0,t=this.getValueSize();t-Math.floor(t)!==0&&($e("KeyframeTrack: Invalid value size in track.",this),e=!1);let n=this.times,s=this.values,r=n.length;r===0&&($e("KeyframeTrack: Track is empty.",this),e=!1);let a=null;for(let o=0;o!==r;o++){let c=n[o];if(typeof c=="number"&&isNaN(c)){$e("KeyframeTrack: Time is not a valid number.",this,o,c),e=!1;break}if(a!==null&&a>c){$e("KeyframeTrack: Out of order keys.",this,o,c,a),e=!1;break}a=c}if(s!==void 0&&jm(s))for(let o=0,c=s.length;o!==c;++o){let l=s[o];if(isNaN(l)){$e("KeyframeTrack: Value is not a valid number.",this,o,l),e=!1;break}}return e}optimize(){let e=this.times.slice(),t=this.values.slice(),n=this.getValueSize(),s=this.getInterpolation()===al,r=e.length-1,a=1;for(let o=1;o<r;++o){let c=!1,l=e[o],h=e[o+1];if(l!==h&&(o!==1||l!==e[0]))if(s)c=!0;else{let d=o*n,u=d-n,f=d+n;for(let g=0;g!==n;++g){let _=t[d+g];if(_!==t[u+g]||_!==t[f+g]){c=!0;break}}}if(c){if(o!==a){e[a]=e[o];let d=o*n,u=a*n;for(let f=0;f!==n;++f)t[u+f]=t[d+f]}++a}}if(r>0){e[a]=e[r];for(let o=r*n,c=a*n,l=0;l!==n;++l)t[c+l]=t[o+l];++a}return a!==e.length?(this.times=e.slice(0,a),this.values=t.slice(0,a*n)):(this.times=e,this.values=t),this}clone(){let e=this.times.slice(),t=this.values.slice(),n=this.constructor,s=new n(this.name,e,t);return s.createInterpolant=this.createInterpolant,s}};Nn.prototype.ValueTypeName="";Nn.prototype.TimeBufferType=Float32Array;Nn.prototype.ValueBufferType=Float32Array;Nn.prototype.DefaultInterpolation=_l;var ss=class extends Nn{constructor(e,t,n){super(e,t,n)}};ss.prototype.ValueTypeName="bool";ss.prototype.ValueBufferType=Array;ss.prototype.DefaultInterpolation=aa;ss.prototype.InterpolantFactoryMethodLinear=void 0;ss.prototype.InterpolantFactoryMethodSmooth=void 0;var zl=class extends Nn{constructor(e,t,n,s){super(e,t,n,s)}};zl.prototype.ValueTypeName="color";var kl=class extends Nn{constructor(e,t,n,s){super(e,t,n,s)}};kl.prototype.ValueTypeName="number";var Hl=class extends is{constructor(e,t,n,s){super(e,t,n,s)}interpolate_(e,t,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=(n-t)/(s-t),l=e*o;for(let h=l+o;l!==h;l+=4)In.slerpFlat(r,0,a,l-o,a,l,c);return r}},Oa=class extends Nn{constructor(e,t,n,s){super(e,t,n,s)}InterpolantFactoryMethodLinear(e){return new Hl(this.times,this.values,this.getValueSize(),e)}};Oa.prototype.ValueTypeName="quaternion";Oa.prototype.InterpolantFactoryMethodSmooth=void 0;var rs=class extends Nn{constructor(e,t,n){super(e,t,n)}};rs.prototype.ValueTypeName="string";rs.prototype.ValueBufferType=Array;rs.prototype.DefaultInterpolation=aa;rs.prototype.InterpolantFactoryMethodLinear=void 0;rs.prototype.InterpolantFactoryMethodSmooth=void 0;var Vl=class extends Nn{constructor(e,t,n,s){super(e,t,n,s)}};Vl.prototype.ValueTypeName="vector";var Gl=class{constructor(e,t,n){let s=this,r=!1,a=0,o=0,c,l=[];this.onStart=void 0,this.onLoad=e,this.onProgress=t,this.onError=n,this._abortController=null,this.itemStart=function(h){o++,r===!1&&s.onStart!==void 0&&s.onStart(h,a,o),r=!0},this.itemEnd=function(h){a++,s.onProgress!==void 0&&s.onProgress(h,a,o),a===o&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return h=h.normalize("NFC"),c?c(h):h},this.setURLModifier=function(h){return c=h,this},this.addHandler=function(h,d){return l.push(h,d),this},this.removeHandler=function(h){let d=l.indexOf(h);return d!==-1&&l.splice(d,2),this},this.getHandler=function(h){for(let d=0,u=l.length;d<u;d+=2){let f=l[d],g=l[d+1];if(f.global&&(f.lastIndex=0),f.test(h))return g}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}},cp=new Gl,Wl=class{constructor(e){this.manager=e!==void 0?e:cp,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(e,t){let n=this;return new Promise(function(s,r){n.load(e,s,t,r)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}};Wl.DEFAULT_MATERIAL_NAME="__DEFAULT";var Rr=class extends ft{constructor(e,t=1){super(),this.isLight=!0,this.type="Light",this.color=new Pe(e),this.intensity=t}dispose(){this.dispatchEvent({type:"dispose"})}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){let t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,t}},Ba=class extends Rr{constructor(e,t,n){super(e,n),this.isHemisphereLight=!0,this.type="HemisphereLight",this.position.copy(ft.DEFAULT_UP),this.updateMatrix(),this.groundColor=new Pe(t)}copy(e,t){return super.copy(e,t),this.groundColor.copy(e.groundColor),this}toJSON(e){let t=super.toJSON(e);return t.object.groundColor=this.groundColor.getHex(),t}},jh=new st,xf=new R,vf=new R,Xl=class{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.biasNode=null,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new Z(512,512),this.mapType=hn,this.map=null,this.mapPass=null,this.matrix=new st,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new br,this._frameExtents=new Z(1,1),this._viewportCount=1,this._viewports=[new mt(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){let t=this.camera,n=this.matrix;xf.setFromMatrixPosition(e.matrixWorld),t.position.copy(xf),vf.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(vf),t.updateMatrixWorld(),jh.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix(jh,t.coordinateSystem,t.reversedDepth),t.coordinateSystem===gr||t.reversedDepth?n.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(jh)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this.biasNode=e.biasNode,this}clone(){return new this.constructor().copy(this)}toJSON(){let e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}},il=new R,sl=new In,ci=new R,za=class extends ft{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new st,this.projectionMatrix=new st,this.projectionMatrixInverse=new st,this.coordinateSystem=$n,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(il,sl,ci),ci.x===1&&ci.y===1&&ci.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(il,sl,ci.set(1,1,1)).invert()}updateWorldMatrix(e,t,n=!1){super.updateWorldMatrix(e,t,n),this.matrixWorld.decompose(il,sl,ci),ci.x===1&&ci.y===1&&ci.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(il,sl,ci.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},ts=new R,yf=new Z,Mf=new Z,Qt=class extends za{constructor(e=50,t=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){let t=.5*this.getFilmHeight()/e;this.fov=xr*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){let e=Math.tan(pr*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return xr*2*Math.atan(Math.tan(pr*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){ts.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(ts.x,ts.y).multiplyScalar(-e/ts.z),ts.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(ts.x,ts.y).multiplyScalar(-e/ts.z)}getViewSize(e,t){return this.getViewBounds(e,yf,Mf),t.subVectors(Mf,yf)}setViewOffset(e,t,n,s,r,a){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=this.near,t=e*Math.tan(pr*.5*this.fov)/this.zoom,n=2*t,s=this.aspect*n,r=-.5*s,a=this.view;if(this.view!==null&&this.view.enabled){let c=a.fullWidth,l=a.fullHeight;r+=a.offsetX*s/c,t-=a.offsetY*n/l,s*=a.width/c,n*=a.height/l}let o=this.filmOffset;o!==0&&(r+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}};var cu=class extends Xl{constructor(){super(new Qt(90,1,.5,500)),this.isPointLightShadow=!0}},ka=class extends Rr{constructor(e,t,n=0,s=2){super(e,t),this.isPointLight=!0,this.type="PointLight",this.distance=n,this.decay=s,this.shadow=new cu}get power(){return this.intensity*4*Math.PI}set power(e){this.intensity=e/(4*Math.PI)}dispose(){super.dispose(),this.shadow.dispose()}copy(e,t){return super.copy(e,t),this.distance=e.distance,this.decay=e.decay,this.shadow=e.shadow.clone(),this}toJSON(e){let t=super.toJSON(e);return t.object.distance=this.distance,t.object.decay=this.decay,t.object.shadow=this.shadow.toJSON(),t}},as=class extends za{constructor(e=-1,t=1,n=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=n-e,a=n+e,o=s+t,c=s-t;if(this.view!==null&&this.view.enabled){let l=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=l*this.view.offsetX,a=r+l*this.view.width,o-=h*this.view.offsetY,c=o-h*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,c,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}},hu=class extends Xl{constructor(){super(new as(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}},Cr=class extends Rr{constructor(e,t){super(e,t),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(ft.DEFAULT_UP),this.updateMatrix(),this.target=new ft,this.shadow=new hu}dispose(){super.dispose(),this.shadow.dispose()}copy(e){return super.copy(e),this.target=e.target.clone(),this.shadow=e.shadow.clone(),this}toJSON(e){let t=super.toJSON(e);return t.object.shadow=this.shadow.toJSON(),t.object.target=this.target.uuid,t}};var Ha=class extends ut{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){let e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}};var hr=-90,ur=1,ql=class extends ft{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new Qt(hr,ur,e,t);s.layers=this.layers,this.add(s);let r=new Qt(hr,ur,e,t);r.layers=this.layers,this.add(r);let a=new Qt(hr,ur,e,t);a.layers=this.layers,this.add(a);let o=new Qt(hr,ur,e,t);o.layers=this.layers,this.add(o);let c=new Qt(hr,ur,e,t);c.layers=this.layers,this.add(c);let l=new Qt(hr,ur,e,t);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){let e=this.coordinateSystem,t=this.children.concat(),[n,s,r,a,o,c]=t;for(let l of t)this.remove(l);if(e===$n)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(e===gr)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(let l of t)this.add(l),l.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();let{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());let[r,a,o,c,l,h]=this.children,d=e.getRenderTarget(),u=e.getActiveCubeFace(),f=e.getActiveMipmapLevel(),g=e.xr.enabled;e.xr.enabled=!1;let _=n.texture.generateMipmaps;n.texture.generateMipmaps=!1;let p=!1;e.isWebGLRenderer===!0?p=e.state.buffers.depth.getReversed():p=e.reversedDepthBuffer,e.setRenderTarget(n,0,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,r),e.setRenderTarget(n,1,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,a),e.setRenderTarget(n,2,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,o),e.setRenderTarget(n,3,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,c),e.setRenderTarget(n,4,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,l),n.texture.generateMipmaps=_,e.setRenderTarget(n,5,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,h),e.setRenderTarget(d,u,f),e.xr.enabled=g,n.texture.needsPMREMUpdate=!0}},Yl=class extends Qt{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}},Va=class{constructor(){this._previousTime=0,this._currentTime=0,this._startTime=performance.now(),this._delta=0,this._elapsed=0,this._timescale=1,this._document=null,this._pageVisibilityHandler=null}connect(e){this._document=e,e.hidden!==void 0&&(this._pageVisibilityHandler=l0.bind(this),e.addEventListener("visibilitychange",this._pageVisibilityHandler,!1))}disconnect(){this._pageVisibilityHandler!==null&&(this._document.removeEventListener("visibilitychange",this._pageVisibilityHandler),this._pageVisibilityHandler=null),this._document=null}getDelta(){return this._delta/1e3}getElapsed(){return this._elapsed/1e3}getTimescale(){return this._timescale}setTimescale(e){return this._timescale=e,this}reset(){return this._currentTime=performance.now()-this._startTime,this}dispose(){this.disconnect()}update(e){return this._pageVisibilityHandler!==null&&this._document.hidden===!0?this._delta=0:(this._previousTime=this._currentTime,this._currentTime=(e!==void 0?e:performance.now())-this._startTime,this._delta=(this._currentTime-this._previousTime)*this._timescale,this._elapsed+=this._delta),this}};function l0(){this._document.hidden===!1&&this.reset()}var Cu="\\[\\]\\.:\\/",c0=new RegExp("["+Cu+"]","g"),Pu="[^"+Cu+"]",h0="[^"+Cu.replace("\\.","")+"]",u0=/((?:WC+[\/:])*)/.source.replace("WC",Pu),d0=/(WCOD+)?/.source.replace("WCOD",h0),f0=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",Pu),p0=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",Pu),m0=new RegExp("^"+u0+d0+f0+p0+"$"),g0=["material","materials","bones","map"],uu=class{constructor(e,t,n){let s=n||Et.parseTrackName(t);this._targetGroup=e,this._bindings=e.subscribe_(t,s)}getValue(e,t){this.bind();let n=this._targetGroup.nCachedObjects_,s=this._bindings[n];s!==void 0&&s.getValue(e,t)}setValue(e,t){let n=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=n.length;s!==r;++s)n[s].setValue(e,t)}bind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,n=e.length;t!==n;++t)e[t].bind()}unbind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,n=e.length;t!==n;++t)e[t].unbind()}},Et=class i{constructor(e,t,n){this.path=t,this.parsedPath=n||i.parseTrackName(t),this.node=i.findNode(e,this.parsedPath.nodeName),this.rootNode=e,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(e,t,n){return e&&e.isAnimationObjectGroup?new i.Composite(e,t,n):new i(e,t,n)}static sanitizeNodeName(e){return e.replace(/\s/g,"_").replace(c0,"")}static parseTrackName(e){let t=m0.exec(e);if(t===null)throw new Error("THREE.PropertyBinding: Cannot parse trackName: "+e);let n={nodeName:t[2],objectName:t[3],objectIndex:t[4],propertyName:t[5],propertyIndex:t[6]},s=n.nodeName&&n.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=n.nodeName.substring(s+1);g0.indexOf(r)!==-1&&(n.nodeName=n.nodeName.substring(0,s),n.objectName=r)}if(n.propertyName===null||n.propertyName.length===0)throw new Error("THREE.PropertyBinding: can not parse propertyName from trackName: "+e);return n}static findNode(e,t){if(t===void 0||t===""||t==="."||t===-1||t===e.name||t===e.uuid)return e;if(e.skeleton){let n=e.skeleton.getBoneByName(t);if(n!==void 0)return n}if(e.children){let n=function(r){for(let a=0;a<r.length;a++){let o=r[a];if(o.name===t||o.uuid===t)return o;let c=n(o.children);if(c)return c}return null},s=n(e.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,t){e[t]=this.targetObject[this.propertyName]}_getValue_array(e,t){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)e[t++]=n[s]}_getValue_arrayElement(e,t){e[t]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,t){this.resolvedProperty.toArray(e,t)}_setValue_direct(e,t){this.targetObject[this.propertyName]=e[t]}_setValue_direct_setNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,t){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)n[s]=e[t++]}_setValue_array_setNeedsUpdate(e,t){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)n[s]=e[t++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,t){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)n[s]=e[t++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,t){this.resolvedProperty[this.propertyIndex]=e[t]}_setValue_arrayElement_setNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,t){this.resolvedProperty.fromArray(e,t)}_setValue_fromArray_setNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,t){this.bind(),this.getValue(e,t)}_setValue_unbound(e,t){this.bind(),this.setValue(e,t)}bind(){let e=this.node,t=this.parsedPath,n=t.objectName,s=t.propertyName,r=t.propertyIndex;if(e||(e=i.findNode(this.rootNode,t.nodeName),this.node=e),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!e){Ze("PropertyBinding: No target node found for track: "+this.path+".");return}if(n){let l=t.objectIndex;switch(n){case"materials":if(!e.material){$e("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.materials){$e("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}e=e.material.materials;break;case"bones":if(!e.skeleton){$e("PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}e=e.skeleton.bones;for(let h=0;h<e.length;h++)if(e[h].name===l){l=h;break}break;case"map":if("map"in e){e=e.map;break}if(!e.material){$e("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.map){$e("PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}e=e.material.map;break;default:if(e[n]===void 0){$e("PropertyBinding: Can not bind to objectName of node undefined.",this);return}e=e[n]}if(l!==void 0){if(e[l]===void 0){$e("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,e);return}e=e[l]}}let a=e[s];if(a===void 0){let l=t.nodeName;$e("PropertyBinding: Trying to update property for track: "+l+"."+s+" but it wasn't found.",e);return}let o=this.Versioning.None;this.targetObject=e,e.isMaterial===!0?o=this.Versioning.NeedsUpdate:e.isObject3D===!0&&(o=this.Versioning.MatrixWorldNeedsUpdate);let c=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!e.geometry){$e("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!e.geometry.morphAttributes){$e("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}e.morphTargetDictionary[r]!==void 0&&(r=e.morphTargetDictionary[r])}c=this.BindingType.ArrayElement,this.resolvedProperty=a,this.propertyIndex=r}else a.fromArray!==void 0&&a.toArray!==void 0?(c=this.BindingType.HasFromToArray,this.resolvedProperty=a):Array.isArray(a)?(c=this.BindingType.EntireArray,this.resolvedProperty=a):this.propertyName=s;this.getValue=this.GetterByBindingType[c],this.setValue=this.SetterByBindingTypeAndVersioning[c][o]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};Et.Composite=uu;Et.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};Et.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};Et.prototype.GetterByBindingType=[Et.prototype._getValue_direct,Et.prototype._getValue_array,Et.prototype._getValue_arrayElement,Et.prototype._getValue_toArray];Et.prototype.SetterByBindingTypeAndVersioning=[[Et.prototype._setValue_direct,Et.prototype._setValue_direct_setNeedsUpdate,Et.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_array,Et.prototype._setValue_array_setNeedsUpdate,Et.prototype._setValue_array_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_arrayElement,Et.prototype._setValue_arrayElement_setNeedsUpdate,Et.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_fromArray,Et.prototype._setValue_fromArray_setNeedsUpdate,Et.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var ES=new Float32Array(1);var os=class extends ma{constructor(e,t,n=1){super(e,t),this.isInstancedInterleavedBuffer=!0,this.meshPerAttribute=n}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}clone(e){let t=super.clone(e);return t.meshPerAttribute=this.meshPerAttribute,t}toJSON(e){let t=super.toJSON(e);return t.isInstancedInterleavedBuffer=!0,t.meshPerAttribute=this.meshPerAttribute,t}};var Sf=new st,Ga=class{constructor(e,t,n=0,s=1/0){this.ray=new Di(e,t),this.near=n,this.far=s,this.camera=null,this.layers=new yr,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(e,t){this.ray.set(e,t)}setFromCamera(e,t){t.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t):t.isOrthographicCamera?(this.ray.origin.set(e.x,e.y,t.projectionMatrix.elements[14]).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t):$e("Raycaster: Unsupported camera type: "+t.type)}setFromXRController(e){return Sf.identity().extractRotation(e.matrixWorld),this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(Sf),this}intersectObject(e,t=!0,n=[]){return du(e,this,n,t),n.sort(bf),n}intersectObjects(e,t=!0,n=[]){for(let s=0,r=e.length;s<r;s++)du(e[s],this,n,t);return n.sort(bf),n}};function bf(i,e){return i.distance-e.distance}function du(i,e,t,n){let s=!0;if(i.layers.test(e.layers)&&i.raycast(e,t)===!1&&(s=!1),s===!0&&n===!0){let r=i.children;for(let a=0,o=r.length;a<o;a++)du(r[a],e,t,!0)}}var Pr=class{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=je(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(je(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}};var Fu=class Fu{constructor(e,t,n,s){this.elements=[1,0,0,1],e!==void 0&&this.set(e,t,n,s)}identity(){return this.set(1,0,0,1),this}fromArray(e,t=0){for(let n=0;n<4;n++)this.elements[n]=e[n+t];return this}set(e,t,n,s){let r=this.elements;return r[0]=e,r[2]=t,r[1]=n,r[3]=s,this}};Fu.prototype.isMatrix2=!0;var fu=Fu;var Ef=new R,rl=new R,dr=new R,fr=new R,Kh=new R,_0=new R,x0=new R,Wa=class{constructor(e=new R,t=new R){this.start=e,this.end=t}set(e,t){return this.start.copy(e),this.end.copy(t),this}copy(e){return this.start.copy(e.start),this.end.copy(e.end),this}getCenter(e){return e.addVectors(this.start,this.end).multiplyScalar(.5)}delta(e){return e.subVectors(this.end,this.start)}distanceSq(){return this.start.distanceToSquared(this.end)}distance(){return this.start.distanceTo(this.end)}at(e,t){return this.delta(t).multiplyScalar(e).add(this.start)}closestPointToPointParameter(e,t){Ef.subVectors(e,this.start),rl.subVectors(this.end,this.start);let n=rl.dot(rl);if(n===0)return 0;let r=rl.dot(Ef)/n;return t&&(r=je(r,0,1)),r}closestPointToPoint(e,t,n){let s=this.closestPointToPointParameter(e,t);return this.delta(n).multiplyScalar(s).add(this.start)}distanceSqToLine3(e,t=_0,n=x0){let s=10000000000000001e-32,r,a,o=this.start,c=e.start,l=this.end,h=e.end;dr.subVectors(l,o),fr.subVectors(h,c),Kh.subVectors(o,c);let d=dr.dot(dr),u=fr.dot(fr),f=fr.dot(Kh);if(d<=s&&u<=s)return t.copy(o),n.copy(c),t.sub(n),t.dot(t);if(d<=s)r=0,a=f/u,a=je(a,0,1);else{let g=dr.dot(Kh);if(u<=s)a=0,r=je(-g/d,0,1);else{let _=dr.dot(fr),p=d*u-_*_;p!==0?r=je((_*f-g*u)/p,0,1):r=0,a=(_*r+f)/u,a<0?(a=0,r=je(-g/d,0,1)):a>1&&(a=1,r=je((_-g)/d,0,1))}}return t.copy(o).addScaledVector(dr,r),n.copy(c).addScaledVector(fr,a),t.distanceToSquared(n)}applyMatrix4(e){return this.start.applyMatrix4(e),this.end.applyMatrix4(e),this}equals(e){return e.start.equals(this.start)&&e.end.equals(this.end)}clone(){return new this.constructor().copy(this)}};var Xa=class extends jn{constructor(e,t=null){super(),this.object=e,this.domElement=t,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(e){if(e===void 0){Ze("Controls: connect() now requires an element.");return}this.domElement!==null&&this.disconnect(),this.domElement=e}disconnect(){}dispose(){}update(){}};function Iu(i,e,t,n){let s=v0(n);switch(t){case bu:return i*e;case ic:return i*e/s.components*s.byteLength;case sc:return i*e/s.components*s.byteLength;case ps:return i*e*2/s.components*s.byteLength;case rc:return i*e*2/s.components*s.byteLength;case Eu:return i*e*3/s.components*s.byteLength;case bn:return i*e*4/s.components*s.byteLength;case ac:return i*e*4/s.components*s.byteLength;case to:case no:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case io:case so:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case lc:case hc:return Math.max(i,16)*Math.max(e,8)/4;case oc:case cc:return Math.max(i,8)*Math.max(e,8)/2;case uc:case dc:case pc:case mc:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case fc:case ro:case gc:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case _c:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case xc:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case vc:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case yc:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case Mc:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case Sc:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case bc:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case Ec:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case wc:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case Tc:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case Ac:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case Rc:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case Cc:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case Pc:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case Ic:case Dc:case Lc:return Math.ceil(i/4)*Math.ceil(e/4)*16;case Uc:case Nc:return Math.ceil(i/4)*Math.ceil(e/4)*8;case ao:case Fc:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function v0(i){switch(i){case hn:case vu:return{byteLength:1,components:1};case Dr:case yu:case nn:return{byteLength:2,components:1};case tc:case nc:return{byteLength:2,components:4};case ni:case ec:case Vn:return{byteLength:4,components:1};case Mu:case Su:return{byteLength:4,components:3}}throw new Error(`THREE.TextureUtils: Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"185"}}));typeof window<"u"&&(window.__THREE__?Ze("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="185");/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function Dp(){let i=null,e=!1,t=null,n=null;function s(r,a){t(r,a),n=i.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&i!==null&&(n=i.requestAnimationFrame(s),e=!0)},stop:function(){i!==null&&i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){i=r}}}function M0(i){let e=new WeakMap;function t(o,c){let l=o.array,h=o.usage,d=l.byteLength,u=i.createBuffer();i.bindBuffer(c,u),i.bufferData(c,l,h),o.onUploadCallback();let f;if(l instanceof Float32Array)f=i.FLOAT;else if(typeof Float16Array<"u"&&l instanceof Float16Array)f=i.HALF_FLOAT;else if(l instanceof Uint16Array)o.isFloat16BufferAttribute?f=i.HALF_FLOAT:f=i.UNSIGNED_SHORT;else if(l instanceof Int16Array)f=i.SHORT;else if(l instanceof Uint32Array)f=i.UNSIGNED_INT;else if(l instanceof Int32Array)f=i.INT;else if(l instanceof Int8Array)f=i.BYTE;else if(l instanceof Uint8Array)f=i.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)f=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:u,type:f,bytesPerElement:l.BYTES_PER_ELEMENT,version:o.version,size:d}}function n(o,c,l){let h=c.array,d=c.updateRanges;if(i.bindBuffer(l,o),d.length===0)i.bufferSubData(l,0,h);else{d.sort((f,g)=>f.start-g.start);let u=0;for(let f=1;f<d.length;f++){let g=d[u],_=d[f];_.start<=g.start+g.count+1?g.count=Math.max(g.count,_.start+_.count-g.start):(++u,d[u]=_)}d.length=u+1;for(let f=0,g=d.length;f<g;f++){let _=d[f];i.bufferSubData(l,_.start*h.BYTES_PER_ELEMENT,h,_.start,_.count)}c.clearUpdateRanges()}c.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),e.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);let c=e.get(o);c&&(i.deleteBuffer(c.buffer),e.delete(o))}function a(o,c){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){let h=e.get(o);(!h||h.version<o.version)&&e.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}let l=e.get(o);if(l===void 0)e.set(o,t(o,c));else if(l.version<o.version){if(l.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(l.buffer,o,c),l.version=o.version}}return{get:s,remove:r,update:a}}var S0=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,b0=`#ifdef USE_ALPHAHASH
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
#endif`,E0=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,w0=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,T0=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,A0=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,R0=`#ifdef USE_AOMAP
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
#endif`,C0=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,P0=`#ifdef USE_BATCHING
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
#endif`,I0=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,D0=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,L0=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,U0=`float G_BlinnPhong_Implicit( ) {
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
} // validated`,N0=`#ifdef USE_IRIDESCENCE
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
#endif`,F0=`#ifdef USE_BUMPMAP
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
#endif`,O0=`#if NUM_CLIPPING_PLANES > 0
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
#endif`,B0=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,z0=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,k0=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,H0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,V0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,G0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,W0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
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
#endif`,X0=`#define PI 3.141592653589793
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
} // validated`,q0=`#ifdef ENVMAP_TYPE_CUBE_UV
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
#endif`,Y0=`vec3 transformedNormal = objectNormal;
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
#endif`,Z0=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,$0=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,J0=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,j0=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,K0="gl_FragColor = linearToOutputTexel( gl_FragColor );",Q0=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,e_=`#ifdef USE_ENVMAP
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
#endif`,t_=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,n_=`#ifdef USE_ENVMAP
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
#endif`,i_=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,s_=`#ifdef USE_ENVMAP
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
#endif`,r_=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,a_=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,o_=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,l_=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,c_=`#ifdef USE_GRADIENTMAP
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
}`,h_=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,u_=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,d_=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,f_=`uniform bool receiveShadow;
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
#include <lightprobes_pars_fragment>`,p_=`#ifdef USE_ENVMAP
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
#endif`,m_=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,g_=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,__=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,x_=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,v_=`PhysicalMaterial material;
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
#endif`,y_=`uniform sampler2D dfgLUT;
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
}`,M_=`
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
#endif`,S_=`#if defined( RE_IndirectDiffuse )
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
#endif`,b_=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,E_=`#ifdef USE_LIGHT_PROBES_GRID
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
#endif`,w_=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,T_=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,A_=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,R_=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,C_=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,P_=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,I_=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
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
#endif`,D_=`#if defined( USE_POINTS_UV )
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
#endif`,L_=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,U_=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,N_=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,F_=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,O_=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,B_=`#ifdef USE_MORPHTARGETS
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
#endif`,z_=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,k_=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
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
vec3 nonPerturbedNormal = normal;`,H_=`#ifdef USE_NORMALMAP_OBJECTSPACE
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
#endif`,V_=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,G_=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,W_=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,X_=`#ifdef USE_NORMALMAP
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
#endif`,q_=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Y_=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Z_=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,$_=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,J_=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,j_=`vec3 packNormalToRGB( const in vec3 normal ) {
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
}`,K_=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Q_=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,ex=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,tx=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,nx=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,ix=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,sx=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,rx=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,ax=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
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
#endif`,ox=`float getShadowMask() {
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
}`,lx=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,cx=`#ifdef USE_SKINNING
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
#endif`,hx=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,ux=`#ifdef USE_SKINNING
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
#endif`,dx=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,fx=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,px=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,mx=`#ifndef saturate
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
vec3 CustomToneMapping( vec3 color ) { return color; }`,gx=`#ifdef USE_TRANSMISSION
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
#endif`,_x=`#ifdef USE_TRANSMISSION
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
#endif`,xx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,vx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,yx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,Mx=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,Sx=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,bx=`uniform sampler2D t2D;
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
}`,Ex=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,wx=`#ifdef ENVMAP_TYPE_CUBE
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
}`,Tx=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Ax=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Rx=`#include <common>
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
}`,Cx=`#if DEPTH_PACKING == 3200
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
}`,Px=`#define DISTANCE
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
}`,Ix=`#define DISTANCE
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
}`,Dx=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Lx=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ux=`uniform float scale;
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
}`,Nx=`uniform vec3 diffuse;
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
}`,Fx=`#include <common>
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
}`,Ox=`uniform vec3 diffuse;
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
}`,Bx=`#define LAMBERT
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
}`,zx=`#define LAMBERT
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
}`,kx=`#define MATCAP
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
}`,Hx=`#define MATCAP
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
}`,Vx=`#define NORMAL
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
}`,Gx=`#define NORMAL
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
}`,Wx=`#define PHONG
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
}`,Xx=`#define PHONG
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
}`,qx=`#define STANDARD
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
}`,Yx=`#define STANDARD
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
}`,Zx=`#define TOON
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
}`,$x=`#define TOON
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
}`,Jx=`uniform float size;
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
}`,jx=`uniform vec3 diffuse;
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
}`,Kx=`#include <common>
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
}`,Qx=`uniform vec3 color;
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
}`,ev=`uniform float rotation;
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
}`,tv=`uniform vec3 diffuse;
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
}`,ot={alphahash_fragment:S0,alphahash_pars_fragment:b0,alphamap_fragment:E0,alphamap_pars_fragment:w0,alphatest_fragment:T0,alphatest_pars_fragment:A0,aomap_fragment:R0,aomap_pars_fragment:C0,batching_pars_vertex:P0,batching_vertex:I0,begin_vertex:D0,beginnormal_vertex:L0,bsdfs:U0,iridescence_fragment:N0,bumpmap_pars_fragment:F0,clipping_planes_fragment:O0,clipping_planes_pars_fragment:B0,clipping_planes_pars_vertex:z0,clipping_planes_vertex:k0,color_fragment:H0,color_pars_fragment:V0,color_pars_vertex:G0,color_vertex:W0,common:X0,cube_uv_reflection_fragment:q0,defaultnormal_vertex:Y0,displacementmap_pars_vertex:Z0,displacementmap_vertex:$0,emissivemap_fragment:J0,emissivemap_pars_fragment:j0,colorspace_fragment:K0,colorspace_pars_fragment:Q0,envmap_fragment:e_,envmap_common_pars_fragment:t_,envmap_pars_fragment:n_,envmap_pars_vertex:i_,envmap_physical_pars_fragment:p_,envmap_vertex:s_,fog_vertex:r_,fog_pars_vertex:a_,fog_fragment:o_,fog_pars_fragment:l_,gradientmap_pars_fragment:c_,lightmap_pars_fragment:h_,lights_lambert_fragment:u_,lights_lambert_pars_fragment:d_,lights_pars_begin:f_,lights_toon_fragment:m_,lights_toon_pars_fragment:g_,lights_phong_fragment:__,lights_phong_pars_fragment:x_,lights_physical_fragment:v_,lights_physical_pars_fragment:y_,lights_fragment_begin:M_,lights_fragment_maps:S_,lights_fragment_end:b_,lightprobes_pars_fragment:E_,logdepthbuf_fragment:w_,logdepthbuf_pars_fragment:T_,logdepthbuf_pars_vertex:A_,logdepthbuf_vertex:R_,map_fragment:C_,map_pars_fragment:P_,map_particle_fragment:I_,map_particle_pars_fragment:D_,metalnessmap_fragment:L_,metalnessmap_pars_fragment:U_,morphinstance_vertex:N_,morphcolor_vertex:F_,morphnormal_vertex:O_,morphtarget_pars_vertex:B_,morphtarget_vertex:z_,normal_fragment_begin:k_,normal_fragment_maps:H_,normal_pars_fragment:V_,normal_pars_vertex:G_,normal_vertex:W_,normalmap_pars_fragment:X_,clearcoat_normal_fragment_begin:q_,clearcoat_normal_fragment_maps:Y_,clearcoat_pars_fragment:Z_,iridescence_pars_fragment:$_,opaque_fragment:J_,packing:j_,premultiplied_alpha_fragment:K_,project_vertex:Q_,dithering_fragment:ex,dithering_pars_fragment:tx,roughnessmap_fragment:nx,roughnessmap_pars_fragment:ix,shadowmap_pars_fragment:sx,shadowmap_pars_vertex:rx,shadowmap_vertex:ax,shadowmask_pars_fragment:ox,skinbase_vertex:lx,skinning_pars_vertex:cx,skinning_vertex:hx,skinnormal_vertex:ux,specularmap_fragment:dx,specularmap_pars_fragment:fx,tonemapping_fragment:px,tonemapping_pars_fragment:mx,transmission_fragment:gx,transmission_pars_fragment:_x,uv_pars_fragment:xx,uv_pars_vertex:vx,uv_vertex:yx,worldpos_vertex:Mx,background_vert:Sx,background_frag:bx,backgroundCube_vert:Ex,backgroundCube_frag:wx,cube_vert:Tx,cube_frag:Ax,depth_vert:Rx,depth_frag:Cx,distance_vert:Px,distance_frag:Ix,equirect_vert:Dx,equirect_frag:Lx,linedashed_vert:Ux,linedashed_frag:Nx,meshbasic_vert:Fx,meshbasic_frag:Ox,meshlambert_vert:Bx,meshlambert_frag:zx,meshmatcap_vert:kx,meshmatcap_frag:Hx,meshnormal_vert:Vx,meshnormal_frag:Gx,meshphong_vert:Wx,meshphong_frag:Xx,meshphysical_vert:qx,meshphysical_frag:Yx,meshtoon_vert:Zx,meshtoon_frag:$x,points_vert:Jx,points_frag:jx,shadow_vert:Kx,shadow_frag:Qx,sprite_vert:ev,sprite_frag:tv},be={common:{diffuse:{value:new Pe(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Qe}},envmap:{envMap:{value:null},envMapRotation:{value:new Qe},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Qe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Qe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Qe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Qe},normalScale:{value:new Z(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Qe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Qe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Qe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Qe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Pe(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:null},probesMin:{value:new R},probesMax:{value:new R},probesResolution:{value:new R}},points:{diffuse:{value:new Pe(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0},uvTransform:{value:new Qe}},sprite:{diffuse:{value:new Pe(16777215)},opacity:{value:1},center:{value:new Z(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}}},_n={basic:{uniforms:un([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.fog]),vertexShader:ot.meshbasic_vert,fragmentShader:ot.meshbasic_frag},lambert:{uniforms:un([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.fog,be.lights,{emissive:{value:new Pe(0)},envMapIntensity:{value:1}}]),vertexShader:ot.meshlambert_vert,fragmentShader:ot.meshlambert_frag},phong:{uniforms:un([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.fog,be.lights,{emissive:{value:new Pe(0)},specular:{value:new Pe(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:ot.meshphong_vert,fragmentShader:ot.meshphong_frag},standard:{uniforms:un([be.common,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.roughnessmap,be.metalnessmap,be.fog,be.lights,{emissive:{value:new Pe(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:ot.meshphysical_vert,fragmentShader:ot.meshphysical_frag},toon:{uniforms:un([be.common,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.gradientmap,be.fog,be.lights,{emissive:{value:new Pe(0)}}]),vertexShader:ot.meshtoon_vert,fragmentShader:ot.meshtoon_frag},matcap:{uniforms:un([be.common,be.bumpmap,be.normalmap,be.displacementmap,be.fog,{matcap:{value:null}}]),vertexShader:ot.meshmatcap_vert,fragmentShader:ot.meshmatcap_frag},points:{uniforms:un([be.points,be.fog]),vertexShader:ot.points_vert,fragmentShader:ot.points_frag},dashed:{uniforms:un([be.common,be.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:ot.linedashed_vert,fragmentShader:ot.linedashed_frag},depth:{uniforms:un([be.common,be.displacementmap]),vertexShader:ot.depth_vert,fragmentShader:ot.depth_frag},normal:{uniforms:un([be.common,be.bumpmap,be.normalmap,be.displacementmap,{opacity:{value:1}}]),vertexShader:ot.meshnormal_vert,fragmentShader:ot.meshnormal_frag},sprite:{uniforms:un([be.sprite,be.fog]),vertexShader:ot.sprite_vert,fragmentShader:ot.sprite_frag},background:{uniforms:{uvTransform:{value:new Qe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:ot.background_vert,fragmentShader:ot.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Qe}},vertexShader:ot.backgroundCube_vert,fragmentShader:ot.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:ot.cube_vert,fragmentShader:ot.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:ot.equirect_vert,fragmentShader:ot.equirect_frag},distance:{uniforms:un([be.common,be.displacementmap,{referencePosition:{value:new R},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:ot.distance_vert,fragmentShader:ot.distance_frag},shadow:{uniforms:un([be.lights,be.fog,{color:{value:new Pe(0)},opacity:{value:1}}]),vertexShader:ot.shadow_vert,fragmentShader:ot.shadow_frag}};_n.physical={uniforms:un([_n.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Qe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Qe},clearcoatNormalScale:{value:new Z(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Qe},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Qe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Qe},sheen:{value:0},sheenColor:{value:new Pe(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Qe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Qe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Qe},transmissionSamplerSize:{value:new Z},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Qe},attenuationDistance:{value:0},attenuationColor:{value:new Pe(0)},specularColor:{value:new Pe(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Qe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Qe},anisotropyVector:{value:new Z},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Qe}}]),vertexShader:ot.meshphysical_vert,fragmentShader:ot.meshphysical_frag};var zc={r:0,b:0,g:0},nv=new st,Lp=new Qe;Lp.set(-1,0,0,0,1,0,0,0,1);function iv(i,e,t,n,s,r){let a=new Pe(0),o=s===!0?0:1,c,l,h=null,d=0,u=null;function f(M){let S=M.isScene===!0?M.background:null;if(S&&S.isTexture){let y=M.backgroundBlurriness>0;S=e.get(S,y)}return S}function g(M){let S=!1,y=f(M);y===null?p(a,o):y&&y.isColor&&(p(y,1),S=!0);let T=i.xr.getEnvironmentBlendMode();T==="additive"?t.buffers.color.setClear(0,0,0,1,r):T==="alpha-blend"&&t.buffers.color.setClear(0,0,0,0,r),(i.autoClear||S)&&(t.buffers.depth.setTest(!0),t.buffers.depth.setMask(!0),t.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function _(M,S){let y=f(S);y&&(y.isCubeTexture||y.mapping===Qa)?(l===void 0&&(l=new et(new Bt(1,1,1),new bt({name:"BackgroundCubeMaterial",uniforms:Bs(_n.backgroundCube.uniforms),vertexShader:_n.backgroundCube.vertexShader,fragmentShader:_n.backgroundCube.fragmentShader,side:tn,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),l.geometry.deleteAttribute("uv"),l.onBeforeRender=function(T,b,P){this.matrixWorld.copyPosition(P.matrixWorld)},Object.defineProperty(l.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),n.update(l)),l.material.uniforms.envMap.value=y,l.material.uniforms.backgroundBlurriness.value=S.backgroundBlurriness,l.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,l.material.uniforms.backgroundRotation.value.setFromMatrix4(nv.makeRotationFromEuler(S.backgroundRotation)).transpose(),y.isCubeTexture&&y.isRenderTargetTexture===!1&&l.material.uniforms.backgroundRotation.value.premultiply(Lp),l.material.toneMapped=ht.getTransfer(y.colorSpace)!==pt,(h!==y||d!==y.version||u!==i.toneMapping)&&(l.material.needsUpdate=!0,h=y,d=y.version,u=i.toneMapping),l.layers.enableAll(),M.unshift(l,l.geometry,l.material,0,0,null)):y&&y.isTexture&&(c===void 0&&(c=new et(new Hn(2,2),new bt({name:"BackgroundMaterial",uniforms:Bs(_n.background.uniforms),vertexShader:_n.background.vertexShader,fragmentShader:_n.background.fragmentShader,side:Jn,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),n.update(c)),c.material.uniforms.t2D.value=y,c.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,c.material.toneMapped=ht.getTransfer(y.colorSpace)!==pt,y.matrixAutoUpdate===!0&&y.updateMatrix(),c.material.uniforms.uvTransform.value.copy(y.matrix),(h!==y||d!==y.version||u!==i.toneMapping)&&(c.material.needsUpdate=!0,h=y,d=y.version,u=i.toneMapping),c.layers.enableAll(),M.unshift(c,c.geometry,c.material,0,0,null))}function p(M,S){M.getRGB(zc,Ru(i)),t.buffers.color.setClear(zc.r,zc.g,zc.b,S,r)}function m(){l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(M,S=1){a.set(M),o=S,p(a,o)},getClearAlpha:function(){return o},setClearAlpha:function(M){o=M,p(a,o)},render:g,addToRenderList:_,dispose:m}}function sv(i,e){let t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=u(null),r=s,a=!1;function o(I,L,X,q,F){let Y=!1,W=d(I,q,X,L);r!==W&&(r=W,l(r.object)),Y=f(I,q,X,F),Y&&g(I,q,X,F),F!==null&&e.update(F,i.ELEMENT_ARRAY_BUFFER),(Y||a)&&(a=!1,y(I,L,X,q),F!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(F).buffer))}function c(){return i.createVertexArray()}function l(I){return i.bindVertexArray(I)}function h(I){return i.deleteVertexArray(I)}function d(I,L,X,q){let F=q.wireframe===!0,Y=n[L.id];Y===void 0&&(Y={},n[L.id]=Y);let W=I.isInstancedMesh===!0?I.id:0,ie=Y[W];ie===void 0&&(ie={},Y[W]=ie);let ne=ie[X.id];ne===void 0&&(ne={},ie[X.id]=ne);let ge=ne[F];return ge===void 0&&(ge=u(c()),ne[F]=ge),ge}function u(I){let L=[],X=[],q=[];for(let F=0;F<t;F++)L[F]=0,X[F]=0,q[F]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:L,enabledAttributes:X,attributeDivisors:q,object:I,attributes:{},index:null}}function f(I,L,X,q){let F=r.attributes,Y=L.attributes,W=0,ie=X.getAttributes();for(let ne in ie)if(ie[ne].location>=0){let ue=F[ne],xe=Y[ne];if(xe===void 0&&(ne==="instanceMatrix"&&I.instanceMatrix&&(xe=I.instanceMatrix),ne==="instanceColor"&&I.instanceColor&&(xe=I.instanceColor)),ue===void 0||ue.attribute!==xe||xe&&ue.data!==xe.data)return!0;W++}return r.attributesNum!==W||r.index!==q}function g(I,L,X,q){let F={},Y=L.attributes,W=0,ie=X.getAttributes();for(let ne in ie)if(ie[ne].location>=0){let ue=Y[ne];ue===void 0&&(ne==="instanceMatrix"&&I.instanceMatrix&&(ue=I.instanceMatrix),ne==="instanceColor"&&I.instanceColor&&(ue=I.instanceColor));let xe={};xe.attribute=ue,ue&&ue.data&&(xe.data=ue.data),F[ne]=xe,W++}r.attributes=F,r.attributesNum=W,r.index=q}function _(){let I=r.newAttributes;for(let L=0,X=I.length;L<X;L++)I[L]=0}function p(I){m(I,0)}function m(I,L){let X=r.newAttributes,q=r.enabledAttributes,F=r.attributeDivisors;X[I]=1,q[I]===0&&(i.enableVertexAttribArray(I),q[I]=1),F[I]!==L&&(i.vertexAttribDivisor(I,L),F[I]=L)}function M(){let I=r.newAttributes,L=r.enabledAttributes;for(let X=0,q=L.length;X<q;X++)L[X]!==I[X]&&(i.disableVertexAttribArray(X),L[X]=0)}function S(I,L,X,q,F,Y,W){W===!0?i.vertexAttribIPointer(I,L,X,F,Y):i.vertexAttribPointer(I,L,X,q,F,Y)}function y(I,L,X,q){_();let F=q.attributes,Y=X.getAttributes(),W=L.defaultAttributeValues;for(let ie in Y){let ne=Y[ie];if(ne.location>=0){let ge=F[ie];if(ge===void 0&&(ie==="instanceMatrix"&&I.instanceMatrix&&(ge=I.instanceMatrix),ie==="instanceColor"&&I.instanceColor&&(ge=I.instanceColor)),ge!==void 0){let ue=ge.normalized,xe=ge.itemSize,Ne=e.get(ge);if(Ne===void 0)continue;let it=Ne.buffer,Xe=Ne.type,j=Ne.bytesPerElement,he=Xe===i.INT||Xe===i.UNSIGNED_INT||ge.gpuType===ec;if(ge.isInterleavedBufferAttribute){let le=ge.data,Te=le.stride,Fe=ge.offset;if(le.isInstancedInterleavedBuffer){for(let ke=0;ke<ne.locationSize;ke++)m(ne.location+ke,le.meshPerAttribute);I.isInstancedMesh!==!0&&q._maxInstanceCount===void 0&&(q._maxInstanceCount=le.meshPerAttribute*le.count)}else for(let ke=0;ke<ne.locationSize;ke++)p(ne.location+ke);i.bindBuffer(i.ARRAY_BUFFER,it);for(let ke=0;ke<ne.locationSize;ke++)S(ne.location+ke,xe/ne.locationSize,Xe,ue,Te*j,(Fe+xe/ne.locationSize*ke)*j,he)}else{if(ge.isInstancedBufferAttribute){for(let le=0;le<ne.locationSize;le++)m(ne.location+le,ge.meshPerAttribute);I.isInstancedMesh!==!0&&q._maxInstanceCount===void 0&&(q._maxInstanceCount=ge.meshPerAttribute*ge.count)}else for(let le=0;le<ne.locationSize;le++)p(ne.location+le);i.bindBuffer(i.ARRAY_BUFFER,it);for(let le=0;le<ne.locationSize;le++)S(ne.location+le,xe/ne.locationSize,Xe,ue,xe*j,xe/ne.locationSize*le*j,he)}}else if(W!==void 0){let ue=W[ie];if(ue!==void 0)switch(ue.length){case 2:i.vertexAttrib2fv(ne.location,ue);break;case 3:i.vertexAttrib3fv(ne.location,ue);break;case 4:i.vertexAttrib4fv(ne.location,ue);break;default:i.vertexAttrib1fv(ne.location,ue)}}}}M()}function T(){E();for(let I in n){let L=n[I];for(let X in L){let q=L[X];for(let F in q){let Y=q[F];for(let W in Y)h(Y[W].object),delete Y[W];delete q[F]}}delete n[I]}}function b(I){if(n[I.id]===void 0)return;let L=n[I.id];for(let X in L){let q=L[X];for(let F in q){let Y=q[F];for(let W in Y)h(Y[W].object),delete Y[W];delete q[F]}}delete n[I.id]}function P(I){for(let L in n){let X=n[L];for(let q in X){let F=X[q];if(F[I.id]===void 0)continue;let Y=F[I.id];for(let W in Y)h(Y[W].object),delete Y[W];delete F[I.id]}}}function x(I){for(let L in n){let X=n[L],q=I.isInstancedMesh===!0?I.id:0,F=X[q];if(F!==void 0){for(let Y in F){let W=F[Y];for(let ie in W)h(W[ie].object),delete W[ie];delete F[Y]}delete X[q],Object.keys(X).length===0&&delete n[L]}}}function E(){C(),a=!0,r!==s&&(r=s,l(r.object))}function C(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:E,resetDefaultState:C,dispose:T,releaseStatesOfGeometry:b,releaseStatesOfObject:x,releaseStatesOfProgram:P,initAttributes:_,enableAttribute:p,disableUnusedAttributes:M}}function rv(i,e,t){let n;function s(c){n=c}function r(c,l){i.drawArrays(n,c,l),t.update(l,n,1)}function a(c,l,h){h!==0&&(i.drawArraysInstanced(n,c,l,h),t.update(l,n,h))}function o(c,l,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,l,0,h);let u=0;for(let f=0;f<h;f++)u+=l[f];t.update(u,n,1)}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o}function av(i,e,t,n){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){let P=e.get("EXT_texture_filter_anisotropic");s=i.getParameter(P.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(P){return!(P!==bn&&n.convert(P)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(P){let x=P===nn&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(P!==hn&&n.convert(P)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&P!==Vn&&!x)}function c(P){if(P==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";P="mediump"}return P==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=t.precision!==void 0?t.precision:"highp",h=c(l);h!==l&&(Ze("WebGLRenderer:",l,"not supported, using",h,"instead."),l=h);let d=t.logarithmicDepthBuffer===!0,u=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control");t.reversedDepthBuffer===!0&&u===!1&&Ze("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.");let f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),g=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),_=i.getParameter(i.MAX_TEXTURE_SIZE),p=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),M=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),S=i.getParameter(i.MAX_VARYING_VECTORS),y=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),T=i.getParameter(i.MAX_SAMPLES),b=i.getParameter(i.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:c,textureFormatReadable:a,textureTypeReadable:o,precision:l,logarithmicDepthBuffer:d,reversedDepthBuffer:u,maxTextures:f,maxVertexTextures:g,maxTextureSize:_,maxCubemapSize:p,maxAttributes:m,maxVertexUniforms:M,maxVaryings:S,maxFragmentUniforms:y,maxSamples:T,samples:b}}function ov(i){let e=this,t=null,n=0,s=!1,r=!1,a=new zn,o=new Qe,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(d,u){let f=d.length!==0||u||n!==0||s;return s=u,n=d.length,f},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(d,u){t=h(d,u,0)},this.setState=function(d,u,f){let g=d.clippingPlanes,_=d.clipIntersection,p=d.clipShadows,m=i.get(d);if(!s||g===null||g.length===0||r&&!p)r?h(null):l();else{let M=r?0:n,S=M*4,y=m.clippingState||null;c.value=y,y=h(g,u,S,f);for(let T=0;T!==S;++T)y[T]=t[T];m.clippingState=y,this.numIntersection=_?this.numPlanes:0,this.numPlanes+=M}};function l(){c.value!==t&&(c.value=t,c.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function h(d,u,f,g){let _=d!==null?d.length:0,p=null;if(_!==0){if(p=c.value,g!==!0||p===null){let m=f+_*4,M=u.matrixWorldInverse;o.getNormalMatrix(M),(p===null||p.length<m)&&(p=new Float32Array(m));for(let S=0,y=f;S!==_;++S,y+=4)a.copy(d[S]).applyMatrix4(M,o),a.normal.toArray(p,y),p[y+3]=a.constant}c.value=p,c.needsUpdate=!0}return e.numPlanes=_,e.numIntersection=0,p}}var ms=4,hp=[.125,.215,.35,.446,.526,.582],zs=20,lv=256,oo=new as,up=new Pe,Ou=null,Bu=0,zu=0,ku=!1,cv=new R,Fr=class{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,s=100,r={}){let{size:a=256,position:o=cv}=r;Ou=this._renderer.getRenderTarget(),Bu=this._renderer.getActiveCubeFace(),zu=this._renderer.getActiveMipmapLevel(),ku=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(a);let c=this._allocateTargets();return c.depthBuffer=!0,this._sceneToCubeUV(e,n,s,c,o),t>0&&this._blur(c,0,0,t),this._applyPMREM(c),this._cleanup(c),c}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=pp(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=fp(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Ou,Bu,zu),this._renderer.xr.enabled=ku,e.scissorTest=!1,Ur(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===us||e.mapping===Os?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Ou=this._renderer.getRenderTarget(),Bu=this._renderer.getActiveCubeFace(),zu=this._renderer.getActiveMipmapLevel(),ku=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;let n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){let e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:en,minFilter:en,generateMipmaps:!1,type:nn,format:bn,colorSpace:oa,depthBuffer:!1},s=dp(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=dp(e,t,n);let{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=hv(r)),this._blurMaterial=dv(r,e,t),this._ggxMaterial=uv(r,e,t)}return s}_compileMaterial(e){let t=new et(new ut,e);this._renderer.compile(t,oo)}_sceneToCubeUV(e,t,n,s,r){let c=new Qt(90,1,t,n),l=[1,-1,1,1,1,1],h=[1,1,1,-1,-1,-1],d=this._renderer,u=d.autoClear,f=d.toneMapping;d.getClearColor(up),d.toneMapping=ti,d.autoClear=!1,d.state.buffers.depth.getReversed()&&(d.setRenderTarget(s),d.clearDepth(),d.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new et(new Bt,new Li({name:"PMREM.Background",side:tn,depthWrite:!1,depthTest:!1})));let _=this._backgroundBox,p=_.material,m=!1,M=e.background;M?M.isColor&&(p.color.copy(M),e.background=null,m=!0):(p.color.copy(up),m=!0);for(let S=0;S<6;S++){let y=S%3;y===0?(c.up.set(0,l[S],0),c.position.set(r.x,r.y,r.z),c.lookAt(r.x+h[S],r.y,r.z)):y===1?(c.up.set(0,0,l[S]),c.position.set(r.x,r.y,r.z),c.lookAt(r.x,r.y+h[S],r.z)):(c.up.set(0,l[S],0),c.position.set(r.x,r.y,r.z),c.lookAt(r.x,r.y,r.z+h[S]));let T=this._cubeSize;Ur(s,y*T,S>2?T:0,T,T),d.setRenderTarget(s),m&&d.render(_,c),d.render(e,c)}d.toneMapping=f,d.autoClear=u,e.background=M}_textureToCubeUV(e,t){let n=this._renderer,s=e.mapping===us||e.mapping===Os;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=pp()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=fp());let r=s?this._cubemapMaterial:this._equirectMaterial,a=this._lodMeshes[0];a.material=r;let o=r.uniforms;o.envMap.value=e;let c=this._cubeSize;Ur(t,0,0,3*c,2*c),n.setRenderTarget(t),n.render(a,oo)}_applyPMREM(e){let t=this._renderer,n=t.autoClear;t.autoClear=!1;let s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=n}_applyGGXFilter(e,t,n){let s=this._renderer,r=this._pingPongRenderTarget,a=this._ggxMaterial,o=this._lodMeshes[n];o.material=a;let c=a.uniforms,l=n/(this._lodMeshes.length-1),h=t/(this._lodMeshes.length-1),d=Math.sqrt(l*l-h*h),u=0+l*1.25,f=d*u,{_lodMax:g}=this,_=this._sizeLods[n],p=3*_*(n>g-ms?n-g+ms:0),m=4*(this._cubeSize-_);c.envMap.value=e.texture,c.roughness.value=f,c.mipInt.value=g-t,Ur(r,p,m,3*_,2*_),s.setRenderTarget(r),s.render(o,oo),c.envMap.value=r.texture,c.roughness.value=0,c.mipInt.value=g-n,Ur(e,p,m,3*_,2*_),s.setRenderTarget(e),s.render(o,oo)}_blur(e,t,n,s,r){let a=this._pingPongRenderTarget;this._halfBlur(e,a,t,n,s,"latitudinal",r),this._halfBlur(a,e,n,n,s,"longitudinal",r)}_halfBlur(e,t,n,s,r,a,o){let c=this._renderer,l=this._blurMaterial;a!=="latitudinal"&&a!=="longitudinal"&&$e("blur direction must be either latitudinal or longitudinal!");let h=3,d=this._lodMeshes[s];d.material=l;let u=l.uniforms,f=this._sizeLods[n]-1,g=isFinite(r)?Math.PI/(2*f):2*Math.PI/(2*zs-1),_=r/g,p=isFinite(r)?1+Math.floor(h*_):zs;p>zs&&Ze(`sigmaRadians, ${r}, is too large and will clip, as it requested ${p} samples when the maximum is set to ${zs}`);let m=[],M=0;for(let P=0;P<zs;++P){let x=P/_,E=Math.exp(-x*x/2);m.push(E),P===0?M+=E:P<p&&(M+=2*E)}for(let P=0;P<m.length;P++)m[P]=m[P]/M;u.envMap.value=e.texture,u.samples.value=p,u.weights.value=m,u.latitudinal.value=a==="latitudinal",o&&(u.poleAxis.value=o);let{_lodMax:S}=this;u.dTheta.value=g,u.mipInt.value=S-n;let y=this._sizeLods[s],T=3*y*(s>S-ms?s-S+ms:0),b=4*(this._cubeSize-y);Ur(t,T,b,3*y,2*y),c.setRenderTarget(t),c.render(d,oo)}};function hv(i){let e=[],t=[],n=[],s=i,r=i-ms+1+hp.length;for(let a=0;a<r;a++){let o=Math.pow(2,s);e.push(o);let c=1/o;a>i-ms?c=hp[a-i+ms-1]:a===0&&(c=0),t.push(c);let l=1/(o-2),h=-l,d=1+l,u=[h,h,d,h,d,d,h,h,d,d,h,d],f=6,g=6,_=3,p=2,m=1,M=new Float32Array(_*g*f),S=new Float32Array(p*g*f),y=new Float32Array(m*g*f);for(let b=0;b<f;b++){let P=b%3*2/3-1,x=b>2?0:-1,E=[P,x,0,P+2/3,x,0,P+2/3,x+1,0,P,x,0,P+2/3,x+1,0,P,x+1,0];M.set(E,_*g*b),S.set(u,p*g*b);let C=[b,b,b,b,b,b];y.set(C,m*g*b)}let T=new ut;T.setAttribute("position",new Ut(M,_)),T.setAttribute("uv",new Ut(S,p)),T.setAttribute("faceIndex",new Ut(y,m)),n.push(new et(T,null)),s>ms&&s--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function dp(i,e,t){let n=new Ht(i,e,t);return n.texture.mapping=Qa,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Ur(i,e,t,n,s){i.viewport.set(e,t,n,s),i.scissor.set(e,t,n,s)}function uv(i,e,t){return new bt({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:lv,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Gc(),fragmentShader:`

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
		`,blending:zt,depthTest:!1,depthWrite:!1})}function dv(i,e,t){let n=new Float32Array(zs),s=new R(0,1,0);return new bt({name:"SphericalGaussianBlur",defines:{n:zs,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Gc(),fragmentShader:`

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
		`,blending:zt,depthTest:!1,depthWrite:!1})}function fp(){return new bt({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Gc(),fragmentShader:`

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
		`,blending:zt,depthTest:!1,depthWrite:!1})}function pp(){return new bt({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Gc(),fragmentShader:`

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
	`}var Hc=class extends Ht{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;let n={width:e,height:e,depth:1},s=[n,n,n,n,n,n];this.texture=new va(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;let n={uniforms:{tEquirect:{value:null}},vertexShader:`

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
			`},s=new Bt(5,5,5),r=new bt({name:"CubemapFromEquirect",uniforms:Bs(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:tn,blending:zt});r.uniforms.tEquirect.value=t;let a=new et(s,r),o=t.minFilter;return t.minFilter===ds&&(t.minFilter=en),new ql(1,10,this).update(e,a),t.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,t=!0,n=!0,s=!0){let r=e.getRenderTarget();for(let a=0;a<6;a++)e.setRenderTarget(this,a),e.clear(t,n,s);e.setRenderTarget(r)}};function fv(i){let e=new WeakMap,t=new WeakMap,n=null;function s(u,f=!1){return u==null?null:f?a(u):r(u)}function r(u){if(u&&u.isTexture){let f=u.mapping;if(f===jl||f===Kl)if(e.has(u)){let g=e.get(u).texture;return o(g,u.mapping)}else{let g=u.image;if(g&&g.height>0){let _=new Hc(g.height);return _.fromEquirectangularTexture(i,u),e.set(u,_),u.addEventListener("dispose",l),o(_.texture,u.mapping)}else return null}}return u}function a(u){if(u&&u.isTexture){let f=u.mapping,g=f===jl||f===Kl,_=f===us||f===Os;if(g||_){let p=t.get(u),m=p!==void 0?p.texture.pmremVersion:0;if(u.isRenderTargetTexture&&u.pmremVersion!==m)return n===null&&(n=new Fr(i)),p=g?n.fromEquirectangular(u,p):n.fromCubemap(u,p),p.texture.pmremVersion=u.pmremVersion,t.set(u,p),p.texture;if(p!==void 0)return p.texture;{let M=u.image;return g&&M&&M.height>0||_&&M&&c(M)?(n===null&&(n=new Fr(i)),p=g?n.fromEquirectangular(u):n.fromCubemap(u),p.texture.pmremVersion=u.pmremVersion,t.set(u,p),u.addEventListener("dispose",h),p.texture):null}}}return u}function o(u,f){return f===jl?u.mapping=us:f===Kl&&(u.mapping=Os),u}function c(u){let f=0,g=6;for(let _=0;_<g;_++)u[_]!==void 0&&f++;return f===g}function l(u){let f=u.target;f.removeEventListener("dispose",l);let g=e.get(f);g!==void 0&&(e.delete(f),g.dispose())}function h(u){let f=u.target;f.removeEventListener("dispose",h);let g=t.get(f);g!==void 0&&(t.delete(f),g.dispose())}function d(){e=new WeakMap,t=new WeakMap,n!==null&&(n.dispose(),n=null)}return{get:s,dispose:d}}function pv(i){let e={};function t(n){if(e[n]!==void 0)return e[n];let s=i.getExtension(n);return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){let s=t(n);return s===null&&Cs("WebGLRenderer: "+n+" extension not supported."),s}}}function mv(i,e,t,n){let s={},r=new WeakMap;function a(d){let u=d.target;u.index!==null&&e.remove(u.index);for(let g in u.attributes)e.remove(u.attributes[g]);u.removeEventListener("dispose",a),delete s[u.id];let f=r.get(u);f&&(e.remove(f),r.delete(u)),n.releaseStatesOfGeometry(u),u.isInstancedBufferGeometry===!0&&delete u._maxInstanceCount,t.memory.geometries--}function o(d,u){return s[u.id]===!0||(u.addEventListener("dispose",a),s[u.id]=!0,t.memory.geometries++),u}function c(d){let u=d.attributes;for(let f in u)e.update(u[f],i.ARRAY_BUFFER)}function l(d){let u=[],f=d.index,g=d.attributes.position,_=0;if(g===void 0)return;if(f!==null){let M=f.array;_=f.version;for(let S=0,y=M.length;S<y;S+=3){let T=M[S+0],b=M[S+1],P=M[S+2];u.push(T,b,b,P,P,T)}}else{let M=g.array;_=g.version;for(let S=0,y=M.length/3-1;S<y;S+=3){let T=S+0,b=S+1,P=S+2;u.push(T,b,b,P,P,T)}}let p=new(g.count>=65535?pa:fa)(u,1);p.version=_;let m=r.get(d);m&&e.remove(m),r.set(d,p)}function h(d){let u=r.get(d);if(u){let f=d.index;f!==null&&u.version<f.version&&l(d)}else l(d);return r.get(d)}return{get:o,update:c,getWireframeAttribute:h}}function gv(i,e,t){let n;function s(d){n=d}let r,a;function o(d){r=d.type,a=d.bytesPerElement}function c(d,u){i.drawElements(n,u,r,d*a),t.update(u,n,1)}function l(d,u,f){f!==0&&(i.drawElementsInstanced(n,u,r,d*a,f),t.update(u,n,f))}function h(d,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,u,0,r,d,0,f);let _=0;for(let p=0;p<f;p++)_+=u[p];t.update(_,n,1)}this.setMode=s,this.setIndex=o,this.render=c,this.renderInstances=l,this.renderMultiDraw=h}function _v(i){let e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,a,o){switch(t.calls++,a){case i.TRIANGLES:t.triangles+=o*(r/3);break;case i.LINES:t.lines+=o*(r/2);break;case i.LINE_STRIP:t.lines+=o*(r-1);break;case i.LINE_LOOP:t.lines+=o*r;break;case i.POINTS:t.points+=o*r;break;default:$e("WebGLInfo: Unknown draw mode:",a);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:n}}function xv(i,e,t){let n=new WeakMap,s=new mt;function r(a,o,c){let l=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,d=h!==void 0?h.length:0,u=n.get(o);if(u===void 0||u.count!==d){let E=function(){P.dispose(),n.delete(o),o.removeEventListener("dispose",E)};u!==void 0&&u.texture.dispose();let f=o.morphAttributes.position!==void 0,g=o.morphAttributes.normal!==void 0,_=o.morphAttributes.color!==void 0,p=o.morphAttributes.position||[],m=o.morphAttributes.normal||[],M=o.morphAttributes.color||[],S=0;f===!0&&(S=1),g===!0&&(S=2),_===!0&&(S=3);let y=o.attributes.position.count*S,T=1;y>e.maxTextureSize&&(T=Math.ceil(y/e.maxTextureSize),y=e.maxTextureSize);let b=new Float32Array(y*T*4*d),P=new ua(b,y,T,d);P.type=Vn,P.needsUpdate=!0;let x=S*4;for(let C=0;C<d;C++){let I=p[C],L=m[C],X=M[C],q=y*T*4*C;for(let F=0;F<I.count;F++){let Y=F*x;f===!0&&(s.fromBufferAttribute(I,F),b[q+Y+0]=s.x,b[q+Y+1]=s.y,b[q+Y+2]=s.z,b[q+Y+3]=0),g===!0&&(s.fromBufferAttribute(L,F),b[q+Y+4]=s.x,b[q+Y+5]=s.y,b[q+Y+6]=s.z,b[q+Y+7]=0),_===!0&&(s.fromBufferAttribute(X,F),b[q+Y+8]=s.x,b[q+Y+9]=s.y,b[q+Y+10]=s.z,b[q+Y+11]=X.itemSize===4?s.w:1)}}u={count:d,texture:P,size:new Z(y,T)},n.set(o,u),o.addEventListener("dispose",E)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)c.getUniforms().setValue(i,"morphTexture",a.morphTexture,t);else{let f=0;for(let _=0;_<l.length;_++)f+=l[_];let g=o.morphTargetsRelative?1:1-f;c.getUniforms().setValue(i,"morphTargetBaseInfluence",g),c.getUniforms().setValue(i,"morphTargetInfluences",l)}c.getUniforms().setValue(i,"morphTargetsTexture",u.texture,t),c.getUniforms().setValue(i,"morphTargetsTextureSize",u.size)}return{update:r}}function vv(i,e,t,n,s){let r=new WeakMap;function a(l){let h=s.render.frame,d=l.geometry,u=e.get(l,d);if(r.get(u)!==h&&(e.update(u),r.set(u,h)),l.isInstancedMesh&&(l.hasEventListener("dispose",c)===!1&&l.addEventListener("dispose",c),r.get(l)!==h&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),r.set(l,h))),l.isSkinnedMesh){let f=l.skeleton;r.get(f)!==h&&(f.update(),r.set(f,h))}return u}function o(){r=new WeakMap}function c(l){let h=l.target;h.removeEventListener("dispose",c),n.releaseStatesOfObject(h),t.remove(h.instanceMatrix),h.instanceColor!==null&&t.remove(h.instanceColor)}return{update:a,dispose:o}}var yv={[Za]:"LINEAR_TONE_MAPPING",[$a]:"REINHARD_TONE_MAPPING",[Ja]:"CINEON_TONE_MAPPING",[hs]:"ACES_FILMIC_TONE_MAPPING",[Ka]:"AGX_TONE_MAPPING",[Fs]:"NEUTRAL_TONE_MAPPING",[ja]:"CUSTOM_TONE_MAPPING"};function Mv(i,e,t,n,s,r){let a=new Ht(e,t,{type:i,depthBuffer:s,stencilBuffer:r,samples:n?4:0,depthTexture:s?new Kn(e,t):void 0}),o=new Ht(e,t,{type:nn,depthBuffer:!1,stencilBuffer:!1}),c=new ut;c.setAttribute("position",new rt([-1,3,0,-1,-1,0,3,-1,0],3)),c.setAttribute("uv",new rt([0,2,0,0,2,0],2));let l=new Ar({uniforms:{tDiffuse:{value:null}},vertexShader:`
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
			}`,depthTest:!1,depthWrite:!1}),h=new et(c,l),d=new as(-1,1,1,-1,0,1),u=null,f=null,g=!1,_,p=null,m=[],M=!1;this.setSize=function(S,y){a.setSize(S,y),o.setSize(S,y);for(let T=0;T<m.length;T++){let b=m[T];b.setSize&&b.setSize(S,y)}},this.setEffects=function(S){m=S,M=m.length>0&&m[0].isRenderPass===!0;let y=a.width,T=a.height;for(let b=0;b<m.length;b++){let P=m[b];P.setSize&&P.setSize(y,T)}},this.begin=function(S,y){if(g||S.toneMapping===ti&&m.length===0)return!1;if(p=y,y!==null){let T=y.width,b=y.height;(a.width!==T||a.height!==b)&&this.setSize(T,b)}return M===!1&&S.setRenderTarget(a),_=S.toneMapping,S.toneMapping=ti,!0},this.hasRenderPass=function(){return M},this.end=function(S,y){S.toneMapping=_,g=!0;let T=a,b=o;for(let P=0;P<m.length;P++){let x=m[P];if(x.enabled!==!1&&(x.render(S,b,T,y),x.needsSwap!==!1)){let E=T;T=b,b=E}}if(u!==S.outputColorSpace||f!==S.toneMapping){u=S.outputColorSpace,f=S.toneMapping,l.defines={},ht.getTransfer(u)===pt&&(l.defines.SRGB_TRANSFER="");let P=yv[f];P&&(l.defines[P]=""),l.needsUpdate=!0}l.uniforms.tDiffuse.value=T.texture,S.setRenderTarget(p),S.render(h,d),p=null,g=!1},this.isCompositing=function(){return g},this.dispose=function(){a.depthTexture&&a.depthTexture.dispose(),a.dispose(),o.dispose(),c.dispose(),l.dispose()}}var Up=new pn,Gu=new Kn(1,1),Np=new ua,Fp=new Ml,Op=new va,mp=[],gp=[],_p=new Float32Array(16),xp=new Float32Array(9),vp=new Float32Array(4);function Or(i,e,t){let n=i[0];if(n<=0||n>0)return i;let s=e*t,r=mp[s];if(r===void 0&&(r=new Float32Array(s),mp[s]=r),e!==0){n.toArray(r,0);for(let a=1,o=0;a!==e;++a)o+=t,i[a].toArray(r,o)}return r}function Wt(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function Xt(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function Wc(i,e){let t=gp[e];t===void 0&&(t=new Int32Array(e),gp[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function Sv(i,e){let t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function bv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Wt(t,e))return;i.uniform2fv(this.addr,e),Xt(t,e)}}function Ev(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Wt(t,e))return;i.uniform3fv(this.addr,e),Xt(t,e)}}function wv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Wt(t,e))return;i.uniform4fv(this.addr,e),Xt(t,e)}}function Tv(i,e){let t=this.cache,n=e.elements;if(n===void 0){if(Wt(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),Xt(t,e)}else{if(Wt(t,n))return;vp.set(n),i.uniformMatrix2fv(this.addr,!1,vp),Xt(t,n)}}function Av(i,e){let t=this.cache,n=e.elements;if(n===void 0){if(Wt(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),Xt(t,e)}else{if(Wt(t,n))return;xp.set(n),i.uniformMatrix3fv(this.addr,!1,xp),Xt(t,n)}}function Rv(i,e){let t=this.cache,n=e.elements;if(n===void 0){if(Wt(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),Xt(t,e)}else{if(Wt(t,n))return;_p.set(n),i.uniformMatrix4fv(this.addr,!1,_p),Xt(t,n)}}function Cv(i,e){let t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function Pv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Wt(t,e))return;i.uniform2iv(this.addr,e),Xt(t,e)}}function Iv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Wt(t,e))return;i.uniform3iv(this.addr,e),Xt(t,e)}}function Dv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Wt(t,e))return;i.uniform4iv(this.addr,e),Xt(t,e)}}function Lv(i,e){let t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function Uv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Wt(t,e))return;i.uniform2uiv(this.addr,e),Xt(t,e)}}function Nv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Wt(t,e))return;i.uniform3uiv(this.addr,e),Xt(t,e)}}function Fv(i,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Wt(t,e))return;i.uniform4uiv(this.addr,e),Xt(t,e)}}function Ov(i,e,t){let n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(Gu.compareFunction=t.isReversedDepthBuffer()?Bc:Oc,r=Gu):r=Up,t.setTexture2D(e||r,s)}function Bv(i,e,t){let n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture3D(e||Fp,s)}function zv(i,e,t){let n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTextureCube(e||Op,s)}function kv(i,e,t){let n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture2DArray(e||Np,s)}function Hv(i){switch(i){case 5126:return Sv;case 35664:return bv;case 35665:return Ev;case 35666:return wv;case 35674:return Tv;case 35675:return Av;case 35676:return Rv;case 5124:case 35670:return Cv;case 35667:case 35671:return Pv;case 35668:case 35672:return Iv;case 35669:case 35673:return Dv;case 5125:return Lv;case 36294:return Uv;case 36295:return Nv;case 36296:return Fv;case 35678:case 36198:case 36298:case 36306:case 35682:return Ov;case 35679:case 36299:case 36307:return Bv;case 35680:case 36300:case 36308:case 36293:return zv;case 36289:case 36303:case 36311:case 36292:return kv}}function Vv(i,e){i.uniform1fv(this.addr,e)}function Gv(i,e){let t=Or(e,this.size,2);i.uniform2fv(this.addr,t)}function Wv(i,e){let t=Or(e,this.size,3);i.uniform3fv(this.addr,t)}function Xv(i,e){let t=Or(e,this.size,4);i.uniform4fv(this.addr,t)}function qv(i,e){let t=Or(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function Yv(i,e){let t=Or(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function Zv(i,e){let t=Or(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function $v(i,e){i.uniform1iv(this.addr,e)}function Jv(i,e){i.uniform2iv(this.addr,e)}function jv(i,e){i.uniform3iv(this.addr,e)}function Kv(i,e){i.uniform4iv(this.addr,e)}function Qv(i,e){i.uniform1uiv(this.addr,e)}function ey(i,e){i.uniform2uiv(this.addr,e)}function ty(i,e){i.uniform3uiv(this.addr,e)}function ny(i,e){i.uniform4uiv(this.addr,e)}function iy(i,e,t){let n=this.cache,s=e.length,r=Wc(t,s);Wt(n,r)||(i.uniform1iv(this.addr,r),Xt(n,r));let a;this.type===i.SAMPLER_2D_SHADOW?a=Gu:a=Up;for(let o=0;o!==s;++o)t.setTexture2D(e[o]||a,r[o])}function sy(i,e,t){let n=this.cache,s=e.length,r=Wc(t,s);Wt(n,r)||(i.uniform1iv(this.addr,r),Xt(n,r));for(let a=0;a!==s;++a)t.setTexture3D(e[a]||Fp,r[a])}function ry(i,e,t){let n=this.cache,s=e.length,r=Wc(t,s);Wt(n,r)||(i.uniform1iv(this.addr,r),Xt(n,r));for(let a=0;a!==s;++a)t.setTextureCube(e[a]||Op,r[a])}function ay(i,e,t){let n=this.cache,s=e.length,r=Wc(t,s);Wt(n,r)||(i.uniform1iv(this.addr,r),Xt(n,r));for(let a=0;a!==s;++a)t.setTexture2DArray(e[a]||Np,r[a])}function oy(i){switch(i){case 5126:return Vv;case 35664:return Gv;case 35665:return Wv;case 35666:return Xv;case 35674:return qv;case 35675:return Yv;case 35676:return Zv;case 5124:case 35670:return $v;case 35667:case 35671:return Jv;case 35668:case 35672:return jv;case 35669:case 35673:return Kv;case 5125:return Qv;case 36294:return ey;case 36295:return ty;case 36296:return ny;case 35678:case 36198:case 36298:case 36306:case 35682:return iy;case 35679:case 36299:case 36307:return sy;case 35680:case 36300:case 36308:case 36293:return ry;case 36289:case 36303:case 36311:case 36292:return ay}}var Wu=class{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=Hv(t.type)}},Xu=class{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=oy(t.type)}},qu=class{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){let s=this.seq;for(let r=0,a=s.length;r!==a;++r){let o=s[r];o.setValue(e,t[o.id],n)}}},Hu=/(\w+)(\])?(\[|\.)?/g;function yp(i,e){i.seq.push(e),i.map[e.id]=e}function ly(i,e,t){let n=i.name,s=n.length;for(Hu.lastIndex=0;;){let r=Hu.exec(n),a=Hu.lastIndex,o=r[1],c=r[2]==="]",l=r[3];if(c&&(o=o|0),l===void 0||l==="["&&a+2===s){yp(t,l===void 0?new Wu(o,i,e):new Xu(o,i,e));break}else{let d=t.map[o];d===void 0&&(d=new qu(o),yp(t,d)),t=d}}}var Nr=class{constructor(e,t){this.seq=[],this.map={};let n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let a=0;a<n;++a){let o=e.getActiveUniform(t,a),c=e.getUniformLocation(t,o.name);ly(o,c,this)}let s=[],r=[];for(let a of this.seq)a.type===e.SAMPLER_2D_SHADOW||a.type===e.SAMPLER_CUBE_SHADOW||a.type===e.SAMPLER_2D_ARRAY_SHADOW?s.push(a):r.push(a);s.length>0&&(this.seq=s.concat(r))}setValue(e,t,n,s){let r=this.map[t];r!==void 0&&r.setValue(e,n,s)}setOptional(e,t,n){let s=t[n];s!==void 0&&this.setValue(e,n,s)}static upload(e,t,n,s){for(let r=0,a=t.length;r!==a;++r){let o=t[r],c=n[o.id];c.needsUpdate!==!1&&o.setValue(e,c.value,s)}}static seqWithValue(e,t){let n=[];for(let s=0,r=e.length;s!==r;++s){let a=e[s];a.id in t&&n.push(a)}return n}};function Mp(i,e,t){let n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}var cy=37297,hy=0;function uy(i,e){let t=i.split(`
`),n=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let a=s;a<r;a++){let o=a+1;n.push(`${o===e?">":" "} ${o}: ${t[a]}`)}return n.join(`
`)}var Sp=new Qe;function dy(i){ht._getMatrix(Sp,ht.workingColorSpace,i);let e=`mat3( ${Sp.elements.map(t=>t.toFixed(4))} )`;switch(ht.getTransfer(i)){case la:return[e,"LinearTransferOETF"];case pt:return[e,"sRGBTransferOETF"];default:return Ze("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function bp(i,e,t){let n=i.getShaderParameter(e,i.COMPILE_STATUS),r=(i.getShaderInfoLog(e)||"").trim();if(n&&r==="")return"";let a=/ERROR: 0:(\d+)/.exec(r);if(a){let o=parseInt(a[1]);return t.toUpperCase()+`

`+r+`

`+uy(i.getShaderSource(e),o)}else return r}function fy(i,e){let t=dy(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}var py={[Za]:"Linear",[$a]:"Reinhard",[Ja]:"Cineon",[hs]:"ACESFilmic",[Ka]:"AgX",[Fs]:"Neutral",[ja]:"Custom"};function my(i,e){let t=py[e];return t===void 0?(Ze("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+i+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}var kc=new R;function gy(){ht.getLuminanceCoefficients(kc);let i=kc.x.toFixed(4),e=kc.y.toFixed(4),t=kc.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function _y(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(co).join(`
`)}function xy(i){let e=[];for(let t in i){let n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function vy(i,e){let t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){let r=i.getActiveAttrib(e,s),a=r.name,o=1;r.type===i.FLOAT_MAT2&&(o=2),r.type===i.FLOAT_MAT3&&(o=3),r.type===i.FLOAT_MAT4&&(o=4),t[a]={type:r.type,location:i.getAttribLocation(e,a),locationSize:o}}return t}function co(i){return i!==""}function Ep(i,e){let t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function wp(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}var yy=/^[ \t]*#include +<([\w\d./]+)>/gm;function Yu(i){return i.replace(yy,Sy)}var My=new Map;function Sy(i,e){let t=ot[e];if(t===void 0){let n=My.get(e);if(n!==void 0)t=ot[n],Ze('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("THREE.WebGLProgram: Can not resolve #include <"+e+">")}return Yu(t)}var by=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Tp(i){return i.replace(by,Ey)}function Ey(i,e,t,n){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Ap(i){let e=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?e+=`
#define HIGH_PRECISION`:i.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}var wy={[Us]:"SHADOWMAP_TYPE_PCF",[Ir]:"SHADOWMAP_TYPE_VSM"};function Ty(i){return wy[i.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}var Ay={[us]:"ENVMAP_TYPE_CUBE",[Os]:"ENVMAP_TYPE_CUBE",[Qa]:"ENVMAP_TYPE_CUBE_UV"};function Ry(i){return i.envMap===!1?"ENVMAP_TYPE_CUBE":Ay[i.envMapMode]||"ENVMAP_TYPE_CUBE"}var Cy={[Os]:"ENVMAP_MODE_REFRACTION"};function Py(i){return i.envMap===!1?"ENVMAP_MODE_REFLECTION":Cy[i.envMapMode]||"ENVMAP_MODE_REFLECTION"}var Iy={[Jl]:"ENVMAP_BLENDING_MULTIPLY",[Vf]:"ENVMAP_BLENDING_MIX",[Gf]:"ENVMAP_BLENDING_ADD"};function Dy(i){return i.envMap===!1?"ENVMAP_BLENDING_NONE":Iy[i.combine]||"ENVMAP_BLENDING_NONE"}function Ly(i){let e=i.envMapCubeUVHeight;if(e===null)return null;let t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function Uy(i,e,t,n){let s=i.getContext(),r=t.defines,a=t.vertexShader,o=t.fragmentShader,c=Ty(t),l=Ry(t),h=Py(t),d=Dy(t),u=Ly(t),f=_y(t),g=xy(r),_=s.createProgram(),p,m,M=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(p=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(co).join(`
`),p.length>0&&(p+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(co).join(`
`),m.length>0&&(m+=`
`)):(p=[Ap(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+h:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexNormals?"#define HAS_NORMAL":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(co).join(`
`),m=[Ap(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+l:"",t.envMap?"#define "+h:"",t.envMap?"#define "+d:"",u?"#define CUBEUV_TEXEL_WIDTH "+u.texelWidth:"",u?"#define CUBEUV_TEXEL_HEIGHT "+u.texelHeight:"",u?"#define CUBEUV_MAX_MIP "+u.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor?"#define USE_COLOR":"",t.vertexAlphas||t.batchingColor?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.numLightProbeGrids>0?"#define USE_LIGHT_PROBES_GRID":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==ti?"#define TONE_MAPPING":"",t.toneMapping!==ti?ot.tonemapping_pars_fragment:"",t.toneMapping!==ti?my("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",ot.colorspace_pars_fragment,fy("linearToOutputTexel",t.outputColorSpace),gy(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(co).join(`
`)),a=Yu(a),a=Ep(a,t),a=wp(a,t),o=Yu(o),o=Ep(o,t),o=wp(o,t),a=Tp(a),o=Tp(o),t.isRawShaderMaterial!==!0&&(M=`#version 300 es
`,p=[f,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+p,m=["#define varying in",t.glslVersion===wu?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===wu?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);let S=M+p+a,y=M+m+o,T=Mp(s,s.VERTEX_SHADER,S),b=Mp(s,s.FRAGMENT_SHADER,y);s.attachShader(_,T),s.attachShader(_,b),t.index0AttributeName!==void 0?s.bindAttribLocation(_,0,t.index0AttributeName):t.hasPositionAttribute===!0&&s.bindAttribLocation(_,0,"position"),s.linkProgram(_);function P(I){if(i.debug.checkShaderErrors){let L=s.getProgramInfoLog(_)||"",X=s.getShaderInfoLog(T)||"",q=s.getShaderInfoLog(b)||"",F=L.trim(),Y=X.trim(),W=q.trim(),ie=!0,ne=!0;if(s.getProgramParameter(_,s.LINK_STATUS)===!1)if(ie=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,_,T,b);else{let ge=bp(s,T,"vertex"),ue=bp(s,b,"fragment");$e("WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(_,s.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+F+`
`+ge+`
`+ue)}else F!==""?Ze("WebGLProgram: Program Info Log:",F):(Y===""||W==="")&&(ne=!1);ne&&(I.diagnostics={runnable:ie,programLog:F,vertexShader:{log:Y,prefix:p},fragmentShader:{log:W,prefix:m}})}s.deleteShader(T),s.deleteShader(b),x=new Nr(s,_),E=vy(s,_)}let x;this.getUniforms=function(){return x===void 0&&P(this),x};let E;this.getAttributes=function(){return E===void 0&&P(this),E};let C=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return C===!1&&(C=s.getProgramParameter(_,cy)),C},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(_),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=hy++,this.cacheKey=e,this.usedTimes=1,this.program=_,this.vertexShader=T,this.fragmentShader=b,this}var Ny=0,Zu=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e,t,n){let s=this._getShaderCacheForMaterial(e);return s.has(t)===!1&&(s.add(t),t.usedTimes++),s.has(n)===!1&&(s.add(n),n.usedTimes++),this}remove(e){let t=this.materialCache.get(e);for(let n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderStage(e){return this._getShaderStage(e.vertexShader)}getFragmentShaderStage(e){return this._getShaderStage(e.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){let t=this.materialCache,n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){let t=this.shaderCache,n=t.get(e);return n===void 0&&(n=new $u(e),t.set(e,n)),n}},$u=class{constructor(e){this.id=Ny++,this.code=e,this.usedTimes=0}};function Fy(i){return i===ps||i===ro||i===ao}function Oy(i,e,t,n,s,r){let a=new yr,o=new Zu,c=new Set,l=[],h=new Map,d=n.logarithmicDepthBuffer,u=n.precision,f={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function g(x){return c.add(x),x===0?"uv":`uv${x}`}function _(x,E,C,I,L,X){let q=I.fog,F=L.geometry,Y=x.isMeshStandardMaterial||x.isMeshLambertMaterial||x.isMeshPhongMaterial?I.environment:null,W=x.isMeshStandardMaterial||x.isMeshLambertMaterial&&!x.envMap||x.isMeshPhongMaterial&&!x.envMap,ie=e.get(x.envMap||Y,W),ne=ie&&ie.mapping===Qa?ie.image.height:null,ge=f[x.type];x.precision!==null&&(u=n.getMaxPrecision(x.precision),u!==x.precision&&Ze("WebGLProgram.getParameters:",x.precision,"not supported, using",u,"instead."));let ue=F.morphAttributes.position||F.morphAttributes.normal||F.morphAttributes.color,xe=ue!==void 0?ue.length:0,Ne=0;F.morphAttributes.position!==void 0&&(Ne=1),F.morphAttributes.normal!==void 0&&(Ne=2),F.morphAttributes.color!==void 0&&(Ne=3);let it,Xe,j,he;if(ge){let Oe=_n[ge];it=Oe.vertexShader,Xe=Oe.fragmentShader}else{it=x.vertexShader,Xe=x.fragmentShader;let Oe=o.getVertexShaderStage(x),It=o.getFragmentShaderStage(x);o.update(x,Oe,It),j=Oe.id,he=It.id}let le=i.getRenderTarget(),Te=i.state.buffers.depth.getReversed(),Fe=L.isInstancedMesh===!0,ke=L.isBatchedMesh===!0,oe=!!x.map,ee=!!x.matcap,O=!!ie,H=!!x.aoMap,Q=!!x.lightMap,G=!!x.bumpMap&&x.wireframe===!1,V=!!x.normalMap,se=!!x.displacementMap,ce=!!x.emissiveMap,fe=!!x.metalnessMap,me=!!x.roughnessMap,D=x.anisotropy>0,Me=x.clearcoat>0,Ve=x.dispersion>0,A=x.iridescence>0,v=x.sheen>0,U=x.transmission>0,B=D&&!!x.anisotropyMap,k=Me&&!!x.clearcoatMap,pe=Me&&!!x.clearcoatNormalMap,_e=Me&&!!x.clearcoatRoughnessMap,te=A&&!!x.iridescenceMap,re=A&&!!x.iridescenceThicknessMap,Se=v&&!!x.sheenColorMap,Ie=v&&!!x.sheenRoughnessMap,ve=!!x.specularMap,ye=!!x.specularColorMap,Be=!!x.specularIntensityMap,qe=U&&!!x.transmissionMap,Je=U&&!!x.thicknessMap,N=!!x.gradientMap,Ee=!!x.alphaMap,ae=x.alphaTest>0,we=!!x.alphaHash,Ce=!!x.extensions,de=ti;x.toneMapped&&(le===null||le.isXRRenderTarget===!0)&&(de=i.toneMapping);let He={shaderID:ge,shaderType:x.type,shaderName:x.name,vertexShader:it,fragmentShader:Xe,defines:x.defines,customVertexShaderID:j,customFragmentShaderID:he,isRawShaderMaterial:x.isRawShaderMaterial===!0,glslVersion:x.glslVersion,precision:u,batching:ke,batchingColor:ke&&L._colorsTexture!==null,instancing:Fe,instancingColor:Fe&&L.instanceColor!==null,instancingMorph:Fe&&L.morphTexture!==null,outputColorSpace:le===null?i.outputColorSpace:le.isXRRenderTarget===!0?le.texture.colorSpace:ht.workingColorSpace,alphaToCoverage:!!x.alphaToCoverage,map:oe,matcap:ee,envMap:O,envMapMode:O&&ie.mapping,envMapCubeUVHeight:ne,aoMap:H,lightMap:Q,bumpMap:G,normalMap:V,displacementMap:se,emissiveMap:ce,normalMapObjectSpace:V&&x.normalMapType===qf,normalMapTangentSpace:V&&x.normalMapType===Lr,packedNormalMap:V&&x.normalMapType===Lr&&Fy(x.normalMap.format),metalnessMap:fe,roughnessMap:me,anisotropy:D,anisotropyMap:B,clearcoat:Me,clearcoatMap:k,clearcoatNormalMap:pe,clearcoatRoughnessMap:_e,dispersion:Ve,iridescence:A,iridescenceMap:te,iridescenceThicknessMap:re,sheen:v,sheenColorMap:Se,sheenRoughnessMap:Ie,specularMap:ve,specularColorMap:ye,specularIntensityMap:Be,transmission:U,transmissionMap:qe,thicknessMap:Je,gradientMap:N,opaque:x.transparent===!1&&x.blending===Ps&&x.alphaToCoverage===!1,alphaMap:Ee,alphaTest:ae,alphaHash:we,combine:x.combine,mapUv:oe&&g(x.map.channel),aoMapUv:H&&g(x.aoMap.channel),lightMapUv:Q&&g(x.lightMap.channel),bumpMapUv:G&&g(x.bumpMap.channel),normalMapUv:V&&g(x.normalMap.channel),displacementMapUv:se&&g(x.displacementMap.channel),emissiveMapUv:ce&&g(x.emissiveMap.channel),metalnessMapUv:fe&&g(x.metalnessMap.channel),roughnessMapUv:me&&g(x.roughnessMap.channel),anisotropyMapUv:B&&g(x.anisotropyMap.channel),clearcoatMapUv:k&&g(x.clearcoatMap.channel),clearcoatNormalMapUv:pe&&g(x.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:_e&&g(x.clearcoatRoughnessMap.channel),iridescenceMapUv:te&&g(x.iridescenceMap.channel),iridescenceThicknessMapUv:re&&g(x.iridescenceThicknessMap.channel),sheenColorMapUv:Se&&g(x.sheenColorMap.channel),sheenRoughnessMapUv:Ie&&g(x.sheenRoughnessMap.channel),specularMapUv:ve&&g(x.specularMap.channel),specularColorMapUv:ye&&g(x.specularColorMap.channel),specularIntensityMapUv:Be&&g(x.specularIntensityMap.channel),transmissionMapUv:qe&&g(x.transmissionMap.channel),thicknessMapUv:Je&&g(x.thicknessMap.channel),alphaMapUv:Ee&&g(x.alphaMap.channel),vertexTangents:!!F.attributes.tangent&&(V||D),vertexNormals:!!F.attributes.normal,vertexColors:x.vertexColors,vertexAlphas:x.vertexColors===!0&&!!F.attributes.color&&F.attributes.color.itemSize===4,pointsUvs:L.isPoints===!0&&!!F.attributes.uv&&(oe||Ee),fog:!!q,useFog:x.fog===!0,fogExp2:!!q&&q.isFogExp2,flatShading:x.wireframe===!1&&(x.flatShading===!0||F.attributes.normal===void 0&&V===!1&&(x.isMeshLambertMaterial||x.isMeshPhongMaterial||x.isMeshStandardMaterial||x.isMeshPhysicalMaterial)),sizeAttenuation:x.sizeAttenuation===!0,logarithmicDepthBuffer:d,reversedDepthBuffer:Te,skinning:L.isSkinnedMesh===!0,hasPositionAttribute:F.attributes.position!==void 0,morphTargets:F.morphAttributes.position!==void 0,morphNormals:F.morphAttributes.normal!==void 0,morphColors:F.morphAttributes.color!==void 0,morphTargetsCount:xe,morphTextureStride:Ne,numDirLights:E.directional.length,numPointLights:E.point.length,numSpotLights:E.spot.length,numSpotLightMaps:E.spotLightMap.length,numRectAreaLights:E.rectArea.length,numHemiLights:E.hemi.length,numDirLightShadows:E.directionalShadowMap.length,numPointLightShadows:E.pointShadowMap.length,numSpotLightShadows:E.spotShadowMap.length,numSpotLightShadowsWithMaps:E.numSpotLightShadowsWithMaps,numLightProbes:E.numLightProbes,numLightProbeGrids:X.length,numClippingPlanes:r.numPlanes,numClipIntersection:r.numIntersection,dithering:x.dithering,shadowMapEnabled:i.shadowMap.enabled&&C.length>0,shadowMapType:i.shadowMap.type,toneMapping:de,decodeVideoTexture:oe&&x.map.isVideoTexture===!0&&ht.getTransfer(x.map.colorSpace)===pt,decodeVideoTextureEmissive:ce&&x.emissiveMap.isVideoTexture===!0&&ht.getTransfer(x.emissiveMap.colorSpace)===pt,premultipliedAlpha:x.premultipliedAlpha,doubleSided:x.side===Sn,flipSided:x.side===tn,useDepthPacking:x.depthPacking>=0,depthPacking:x.depthPacking||0,index0AttributeName:x.index0AttributeName,extensionClipCullDistance:Ce&&x.extensions.clipCullDistance===!0&&t.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Ce&&x.extensions.multiDraw===!0||ke)&&t.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:t.has("KHR_parallel_shader_compile"),customProgramCacheKey:x.customProgramCacheKey()};return He.vertexUv1s=c.has(1),He.vertexUv2s=c.has(2),He.vertexUv3s=c.has(3),c.clear(),He}function p(x){let E=[];if(x.shaderID?E.push(x.shaderID):(E.push(x.customVertexShaderID),E.push(x.customFragmentShaderID)),x.defines!==void 0)for(let C in x.defines)E.push(C),E.push(x.defines[C]);return x.isRawShaderMaterial===!1&&(m(E,x),M(E,x),E.push(i.outputColorSpace)),E.push(x.customProgramCacheKey),E.join()}function m(x,E){x.push(E.precision),x.push(E.outputColorSpace),x.push(E.envMapMode),x.push(E.envMapCubeUVHeight),x.push(E.mapUv),x.push(E.alphaMapUv),x.push(E.lightMapUv),x.push(E.aoMapUv),x.push(E.bumpMapUv),x.push(E.normalMapUv),x.push(E.displacementMapUv),x.push(E.emissiveMapUv),x.push(E.metalnessMapUv),x.push(E.roughnessMapUv),x.push(E.anisotropyMapUv),x.push(E.clearcoatMapUv),x.push(E.clearcoatNormalMapUv),x.push(E.clearcoatRoughnessMapUv),x.push(E.iridescenceMapUv),x.push(E.iridescenceThicknessMapUv),x.push(E.sheenColorMapUv),x.push(E.sheenRoughnessMapUv),x.push(E.specularMapUv),x.push(E.specularColorMapUv),x.push(E.specularIntensityMapUv),x.push(E.transmissionMapUv),x.push(E.thicknessMapUv),x.push(E.combine),x.push(E.fogExp2),x.push(E.sizeAttenuation),x.push(E.morphTargetsCount),x.push(E.morphAttributeCount),x.push(E.numDirLights),x.push(E.numPointLights),x.push(E.numSpotLights),x.push(E.numSpotLightMaps),x.push(E.numHemiLights),x.push(E.numRectAreaLights),x.push(E.numDirLightShadows),x.push(E.numPointLightShadows),x.push(E.numSpotLightShadows),x.push(E.numSpotLightShadowsWithMaps),x.push(E.numLightProbes),x.push(E.shadowMapType),x.push(E.toneMapping),x.push(E.numClippingPlanes),x.push(E.numClipIntersection),x.push(E.depthPacking)}function M(x,E){a.disableAll(),E.instancing&&a.enable(0),E.instancingColor&&a.enable(1),E.instancingMorph&&a.enable(2),E.matcap&&a.enable(3),E.envMap&&a.enable(4),E.normalMapObjectSpace&&a.enable(5),E.normalMapTangentSpace&&a.enable(6),E.clearcoat&&a.enable(7),E.iridescence&&a.enable(8),E.alphaTest&&a.enable(9),E.vertexColors&&a.enable(10),E.vertexAlphas&&a.enable(11),E.vertexUv1s&&a.enable(12),E.vertexUv2s&&a.enable(13),E.vertexUv3s&&a.enable(14),E.vertexTangents&&a.enable(15),E.anisotropy&&a.enable(16),E.alphaHash&&a.enable(17),E.batching&&a.enable(18),E.dispersion&&a.enable(19),E.batchingColor&&a.enable(20),E.gradientMap&&a.enable(21),E.packedNormalMap&&a.enable(22),E.vertexNormals&&a.enable(23),x.push(a.mask),a.disableAll(),E.fog&&a.enable(0),E.useFog&&a.enable(1),E.flatShading&&a.enable(2),E.logarithmicDepthBuffer&&a.enable(3),E.reversedDepthBuffer&&a.enable(4),E.skinning&&a.enable(5),E.morphTargets&&a.enable(6),E.morphNormals&&a.enable(7),E.morphColors&&a.enable(8),E.premultipliedAlpha&&a.enable(9),E.shadowMapEnabled&&a.enable(10),E.doubleSided&&a.enable(11),E.flipSided&&a.enable(12),E.useDepthPacking&&a.enable(13),E.dithering&&a.enable(14),E.transmission&&a.enable(15),E.sheen&&a.enable(16),E.opaque&&a.enable(17),E.pointsUvs&&a.enable(18),E.decodeVideoTexture&&a.enable(19),E.decodeVideoTextureEmissive&&a.enable(20),E.alphaToCoverage&&a.enable(21),E.numLightProbeGrids>0&&a.enable(22),E.hasPositionAttribute&&a.enable(23),x.push(a.mask)}function S(x){let E=f[x.type],C;if(E){let I=_n[E];C=gn.clone(I.uniforms)}else C=x.uniforms;return C}function y(x,E){let C=h.get(E);return C!==void 0?++C.usedTimes:(C=new Uy(i,E,x,s),l.push(C),h.set(E,C)),C}function T(x){if(--x.usedTimes===0){let E=l.indexOf(x);l[E]=l[l.length-1],l.pop(),h.delete(x.cacheKey),x.destroy()}}function b(x){o.remove(x)}function P(){o.dispose()}return{getParameters:_,getProgramCacheKey:p,getUniforms:S,acquireProgram:y,releaseProgram:T,releaseShaderCache:b,programs:l,dispose:P}}function By(){let i=new WeakMap;function e(a){return i.has(a)}function t(a){let o=i.get(a);return o===void 0&&(o={},i.set(a,o)),o}function n(a){i.delete(a)}function s(a,o,c){i.get(a)[o]=c}function r(){i=new WeakMap}return{has:e,get:t,remove:n,update:s,dispose:r}}function zy(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.materialVariant!==e.materialVariant?i.materialVariant-e.materialVariant:i.z!==e.z?i.z-e.z:i.id-e.id}function Rp(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function Cp(){let i=[],e=0,t=[],n=[],s=[];function r(){e=0,t.length=0,n.length=0,s.length=0}function a(u){let f=0;return u.isInstancedMesh&&(f+=2),u.isSkinnedMesh&&(f+=1),f}function o(u,f,g,_,p,m){let M=i[e];return M===void 0?(M={id:u.id,object:u,geometry:f,material:g,materialVariant:a(u),groupOrder:_,renderOrder:u.renderOrder,z:p,group:m},i[e]=M):(M.id=u.id,M.object=u,M.geometry=f,M.material=g,M.materialVariant=a(u),M.groupOrder=_,M.renderOrder=u.renderOrder,M.z=p,M.group=m),e++,M}function c(u,f,g,_,p,m){let M=o(u,f,g,_,p,m);g.transmission>0?n.push(M):g.transparent===!0?s.push(M):t.push(M)}function l(u,f,g,_,p,m){let M=o(u,f,g,_,p,m);g.transmission>0?n.unshift(M):g.transparent===!0?s.unshift(M):t.unshift(M)}function h(u,f,g){t.length>1&&t.sort(u||zy),n.length>1&&n.sort(f||Rp),s.length>1&&s.sort(f||Rp),g&&(t.reverse(),n.reverse(),s.reverse())}function d(){for(let u=e,f=i.length;u<f;u++){let g=i[u];if(g.id===null)break;g.id=null,g.object=null,g.geometry=null,g.material=null,g.group=null}}return{opaque:t,transmissive:n,transparent:s,init:r,push:c,unshift:l,finish:d,sort:h}}function ky(){let i=new WeakMap;function e(n,s){let r=i.get(n),a;return r===void 0?(a=new Cp,i.set(n,[a])):s>=r.length?(a=new Cp,r.push(a)):a=r[s],a}function t(){i=new WeakMap}return{get:e,dispose:t}}function Hy(){let i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new R,color:new Pe};break;case"SpotLight":t={position:new R,direction:new R,color:new Pe,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new R,color:new Pe,distance:0,decay:0};break;case"HemisphereLight":t={direction:new R,skyColor:new Pe,groundColor:new Pe};break;case"RectAreaLight":t={color:new Pe,position:new R,halfWidth:new R,halfHeight:new R};break}return i[e.id]=t,t}}}function Vy(){let i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Z};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Z};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Z,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}var Gy=0;function Wy(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function Xy(i){let e=new Hy,t=Vy(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)n.probe.push(new R);let s=new R,r=new st,a=new st;function o(l){let h=0,d=0,u=0;for(let E=0;E<9;E++)n.probe[E].set(0,0,0);let f=0,g=0,_=0,p=0,m=0,M=0,S=0,y=0,T=0,b=0,P=0;l.sort(Wy);for(let E=0,C=l.length;E<C;E++){let I=l[E],L=I.color,X=I.intensity,q=I.distance,F=null;if(I.shadow&&I.shadow.map&&(I.shadow.map.texture.format===ps?F=I.shadow.map.texture:F=I.shadow.map.depthTexture||I.shadow.map.texture),I.isAmbientLight)h+=L.r*X,d+=L.g*X,u+=L.b*X;else if(I.isLightProbe){for(let Y=0;Y<9;Y++)n.probe[Y].addScaledVector(I.sh.coefficients[Y],X);P++}else if(I.isDirectionalLight){let Y=e.get(I);if(Y.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){let W=I.shadow,ie=t.get(I);ie.shadowIntensity=W.intensity,ie.shadowBias=W.bias,ie.shadowNormalBias=W.normalBias,ie.shadowRadius=W.radius,ie.shadowMapSize=W.mapSize,n.directionalShadow[f]=ie,n.directionalShadowMap[f]=F,n.directionalShadowMatrix[f]=I.shadow.matrix,M++}n.directional[f]=Y,f++}else if(I.isSpotLight){let Y=e.get(I);Y.position.setFromMatrixPosition(I.matrixWorld),Y.color.copy(L).multiplyScalar(X),Y.distance=q,Y.coneCos=Math.cos(I.angle),Y.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),Y.decay=I.decay,n.spot[_]=Y;let W=I.shadow;if(I.map&&(n.spotLightMap[T]=I.map,T++,W.updateMatrices(I),I.castShadow&&b++),n.spotLightMatrix[_]=W.matrix,I.castShadow){let ie=t.get(I);ie.shadowIntensity=W.intensity,ie.shadowBias=W.bias,ie.shadowNormalBias=W.normalBias,ie.shadowRadius=W.radius,ie.shadowMapSize=W.mapSize,n.spotShadow[_]=ie,n.spotShadowMap[_]=F,y++}_++}else if(I.isRectAreaLight){let Y=e.get(I);Y.color.copy(L).multiplyScalar(X),Y.halfWidth.set(I.width*.5,0,0),Y.halfHeight.set(0,I.height*.5,0),n.rectArea[p]=Y,p++}else if(I.isPointLight){let Y=e.get(I);if(Y.color.copy(I.color).multiplyScalar(I.intensity),Y.distance=I.distance,Y.decay=I.decay,I.castShadow){let W=I.shadow,ie=t.get(I);ie.shadowIntensity=W.intensity,ie.shadowBias=W.bias,ie.shadowNormalBias=W.normalBias,ie.shadowRadius=W.radius,ie.shadowMapSize=W.mapSize,ie.shadowCameraNear=W.camera.near,ie.shadowCameraFar=W.camera.far,n.pointShadow[g]=ie,n.pointShadowMap[g]=F,n.pointShadowMatrix[g]=I.shadow.matrix,S++}n.point[g]=Y,g++}else if(I.isHemisphereLight){let Y=e.get(I);Y.skyColor.copy(I.color).multiplyScalar(X),Y.groundColor.copy(I.groundColor).multiplyScalar(X),n.hemi[m]=Y,m++}}p>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=be.LTC_FLOAT_1,n.rectAreaLTC2=be.LTC_FLOAT_2):(n.rectAreaLTC1=be.LTC_HALF_1,n.rectAreaLTC2=be.LTC_HALF_2)),n.ambient[0]=h,n.ambient[1]=d,n.ambient[2]=u;let x=n.hash;(x.directionalLength!==f||x.pointLength!==g||x.spotLength!==_||x.rectAreaLength!==p||x.hemiLength!==m||x.numDirectionalShadows!==M||x.numPointShadows!==S||x.numSpotShadows!==y||x.numSpotMaps!==T||x.numLightProbes!==P)&&(n.directional.length=f,n.spot.length=_,n.rectArea.length=p,n.point.length=g,n.hemi.length=m,n.directionalShadow.length=M,n.directionalShadowMap.length=M,n.pointShadow.length=S,n.pointShadowMap.length=S,n.spotShadow.length=y,n.spotShadowMap.length=y,n.directionalShadowMatrix.length=M,n.pointShadowMatrix.length=S,n.spotLightMatrix.length=y+T-b,n.spotLightMap.length=T,n.numSpotLightShadowsWithMaps=b,n.numLightProbes=P,x.directionalLength=f,x.pointLength=g,x.spotLength=_,x.rectAreaLength=p,x.hemiLength=m,x.numDirectionalShadows=M,x.numPointShadows=S,x.numSpotShadows=y,x.numSpotMaps=T,x.numLightProbes=P,n.version=Gy++)}function c(l,h){let d=0,u=0,f=0,g=0,_=0,p=h.matrixWorldInverse;for(let m=0,M=l.length;m<M;m++){let S=l[m];if(S.isDirectionalLight){let y=n.directional[d];y.direction.setFromMatrixPosition(S.matrixWorld),s.setFromMatrixPosition(S.target.matrixWorld),y.direction.sub(s),y.direction.transformDirection(p),d++}else if(S.isSpotLight){let y=n.spot[f];y.position.setFromMatrixPosition(S.matrixWorld),y.position.applyMatrix4(p),y.direction.setFromMatrixPosition(S.matrixWorld),s.setFromMatrixPosition(S.target.matrixWorld),y.direction.sub(s),y.direction.transformDirection(p),f++}else if(S.isRectAreaLight){let y=n.rectArea[g];y.position.setFromMatrixPosition(S.matrixWorld),y.position.applyMatrix4(p),a.identity(),r.copy(S.matrixWorld),r.premultiply(p),a.extractRotation(r),y.halfWidth.set(S.width*.5,0,0),y.halfHeight.set(0,S.height*.5,0),y.halfWidth.applyMatrix4(a),y.halfHeight.applyMatrix4(a),g++}else if(S.isPointLight){let y=n.point[u];y.position.setFromMatrixPosition(S.matrixWorld),y.position.applyMatrix4(p),u++}else if(S.isHemisphereLight){let y=n.hemi[_];y.direction.setFromMatrixPosition(S.matrixWorld),y.direction.transformDirection(p),_++}}}return{setup:o,setupView:c,state:n}}function Pp(i){let e=new Xy(i),t=[],n=[],s=[];function r(u){d.camera=u,t.length=0,n.length=0,s.length=0}function a(u){t.push(u)}function o(u){n.push(u)}function c(u){s.push(u)}function l(){e.setup(t)}function h(u){e.setupView(t,u)}let d={lightsArray:t,shadowsArray:n,lightProbeGridArray:s,camera:null,lights:e,transmissionRenderTarget:{},textureUnits:0};return{init:r,state:d,setupLights:l,setupLightsView:h,pushLight:a,pushShadow:o,pushLightProbeGrid:c}}function qy(i){let e=new WeakMap;function t(s,r=0){let a=e.get(s),o;return a===void 0?(o=new Pp(i),e.set(s,[o])):r>=a.length?(o=new Pp(i),a.push(o)):o=a[r],o}function n(){e=new WeakMap}return{get:t,dispose:n}}var Yy=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,Zy=`uniform sampler2D shadow_pass;
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
}`,$y=[new R(1,0,0),new R(-1,0,0),new R(0,1,0),new R(0,-1,0),new R(0,0,1),new R(0,0,-1)],Jy=[new R(0,-1,0),new R(0,-1,0),new R(0,0,1),new R(0,0,-1),new R(0,-1,0),new R(0,-1,0)],Ip=new st,lo=new R,Vu=new R;function jy(i,e,t){let n=new br,s=new Z,r=new Z,a=new mt,o=new Ll,c=new Ul,l={},h=t.maxTextureSize,d={[Jn]:tn,[tn]:Jn,[Sn]:Sn},u=new bt({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Z},radius:{value:4}},vertexShader:Yy,fragmentShader:Zy}),f=u.clone();f.defines.HORIZONTAL_PASS=1;let g=new ut;g.setAttribute("position",new Ut(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let _=new et(g,u),p=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Us;let m=this.type;this.render=function(b,P,x){if(p.enabled===!1||p.autoUpdate===!1&&p.needsUpdate===!1||b.length===0)return;this.type===Af&&(Ze("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),this.type=Us);let E=i.getRenderTarget(),C=i.getActiveCubeFace(),I=i.getActiveMipmapLevel(),L=i.state;L.setBlending(zt),L.buffers.depth.getReversed()===!0?L.buffers.color.setClear(0,0,0,0):L.buffers.color.setClear(1,1,1,1),L.buffers.depth.setTest(!0),L.setScissorTest(!1);let X=m!==this.type;X&&P.traverse(function(q){q.material&&(Array.isArray(q.material)?q.material.forEach(F=>F.needsUpdate=!0):q.material.needsUpdate=!0)});for(let q=0,F=b.length;q<F;q++){let Y=b[q],W=Y.shadow;if(W===void 0){Ze("WebGLShadowMap:",Y,"has no shadow.");continue}if(W.autoUpdate===!1&&W.needsUpdate===!1)continue;s.copy(W.mapSize);let ie=W.getFrameExtents();s.multiply(ie),r.copy(W.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/ie.x),s.x=r.x*ie.x,W.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/ie.y),s.y=r.y*ie.y,W.mapSize.y=r.y));let ne=i.state.buffers.depth.getReversed();if(W.camera._reversedDepth=ne,W.map===null||X===!0){if(W.map!==null&&(W.map.depthTexture!==null&&(W.map.depthTexture.dispose(),W.map.depthTexture=null),W.map.dispose()),this.type===Ir){if(Y.isPointLight){Ze("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}W.map=new Ht(s.x,s.y,{format:ps,type:nn,minFilter:en,magFilter:en,generateMipmaps:!1}),W.map.texture.name=Y.name+".shadowMap",W.map.depthTexture=new Kn(s.x,s.y,Vn),W.map.depthTexture.name=Y.name+".shadowMapDepth",W.map.depthTexture.format=fi,W.map.depthTexture.compareFunction=null,W.map.depthTexture.minFilter=Ot,W.map.depthTexture.magFilter=Ot}else Y.isPointLight?(W.map=new Hc(s.x),W.map.depthTexture=new Tl(s.x,ni)):(W.map=new Ht(s.x,s.y),W.map.depthTexture=new Kn(s.x,s.y,ni)),W.map.depthTexture.name=Y.name+".shadowMap",W.map.depthTexture.format=fi,this.type===Us?(W.map.depthTexture.compareFunction=ne?Bc:Oc,W.map.depthTexture.minFilter=en,W.map.depthTexture.magFilter=en):(W.map.depthTexture.compareFunction=null,W.map.depthTexture.minFilter=Ot,W.map.depthTexture.magFilter=Ot);W.camera.updateProjectionMatrix()}let ge=W.map.isWebGLCubeRenderTarget?6:1;for(let ue=0;ue<ge;ue++){if(W.map.isWebGLCubeRenderTarget)i.setRenderTarget(W.map,ue),i.clear();else{ue===0&&(i.setRenderTarget(W.map),i.clear());let xe=W.getViewport(ue);a.set(r.x*xe.x,r.y*xe.y,r.x*xe.z,r.y*xe.w),L.viewport(a)}if(Y.isPointLight){let xe=W.camera,Ne=W.matrix,it=Y.distance||xe.far;it!==xe.far&&(xe.far=it,xe.updateProjectionMatrix()),lo.setFromMatrixPosition(Y.matrixWorld),xe.position.copy(lo),Vu.copy(xe.position),Vu.add($y[ue]),xe.up.copy(Jy[ue]),xe.lookAt(Vu),xe.updateMatrixWorld(),Ne.makeTranslation(-lo.x,-lo.y,-lo.z),Ip.multiplyMatrices(xe.projectionMatrix,xe.matrixWorldInverse),W._frustum.setFromProjectionMatrix(Ip,xe.coordinateSystem,xe.reversedDepth)}else W.updateMatrices(Y);n=W.getFrustum(),y(P,x,W.camera,Y,this.type)}W.isPointLightShadow!==!0&&this.type===Ir&&M(W,x),W.needsUpdate=!1}m=this.type,p.needsUpdate=!1,i.setRenderTarget(E,C,I)};function M(b,P){let x=e.update(_);u.defines.VSM_SAMPLES!==b.blurSamples&&(u.defines.VSM_SAMPLES=b.blurSamples,f.defines.VSM_SAMPLES=b.blurSamples,u.needsUpdate=!0,f.needsUpdate=!0),b.mapPass===null&&(b.mapPass=new Ht(s.x,s.y,{format:ps,type:nn})),u.uniforms.shadow_pass.value=b.map.depthTexture,u.uniforms.resolution.value=b.mapSize,u.uniforms.radius.value=b.radius,i.setRenderTarget(b.mapPass),i.clear(),i.renderBufferDirect(P,null,x,u,_,null),f.uniforms.shadow_pass.value=b.mapPass.texture,f.uniforms.resolution.value=b.mapSize,f.uniforms.radius.value=b.radius,i.setRenderTarget(b.map),i.clear(),i.renderBufferDirect(P,null,x,f,_,null)}function S(b,P,x,E){let C=null,I=x.isPointLight===!0?b.customDistanceMaterial:b.customDepthMaterial;if(I!==void 0)C=I;else if(C=x.isPointLight===!0?c:o,i.localClippingEnabled&&P.clipShadows===!0&&Array.isArray(P.clippingPlanes)&&P.clippingPlanes.length!==0||P.displacementMap&&P.displacementScale!==0||P.alphaMap&&P.alphaTest>0||P.map&&P.alphaTest>0||P.alphaToCoverage===!0){let L=C.uuid,X=P.uuid,q=l[L];q===void 0&&(q={},l[L]=q);let F=q[X];F===void 0&&(F=C.clone(),q[X]=F,P.addEventListener("dispose",T)),C=F}if(C.visible=P.visible,C.wireframe=P.wireframe,E===Ir?C.side=P.shadowSide!==null?P.shadowSide:P.side:C.side=P.shadowSide!==null?P.shadowSide:d[P.side],C.alphaMap=P.alphaMap,C.alphaTest=P.alphaToCoverage===!0?.5:P.alphaTest,C.map=P.map,C.clipShadows=P.clipShadows,C.clippingPlanes=P.clippingPlanes,C.clipIntersection=P.clipIntersection,C.displacementMap=P.displacementMap,C.displacementScale=P.displacementScale,C.displacementBias=P.displacementBias,C.wireframeLinewidth=P.wireframeLinewidth,C.linewidth=P.linewidth,x.isPointLight===!0&&C.isMeshDistanceMaterial===!0){let L=i.properties.get(C);L.light=x}return C}function y(b,P,x,E,C){if(b.visible===!1)return;if(b.layers.test(P.layers)&&(b.isMesh||b.isLine||b.isPoints)&&(b.castShadow||b.receiveShadow&&C===Ir)&&(!b.frustumCulled||n.intersectsObject(b))){b.modelViewMatrix.multiplyMatrices(x.matrixWorldInverse,b.matrixWorld);let X=e.update(b),q=b.material;if(Array.isArray(q)){let F=X.groups;for(let Y=0,W=F.length;Y<W;Y++){let ie=F[Y],ne=q[ie.materialIndex];if(ne&&ne.visible){let ge=S(b,ne,E,C);b.onBeforeShadow(i,b,P,x,X,ge,ie),i.renderBufferDirect(x,null,X,ge,b,ie),b.onAfterShadow(i,b,P,x,X,ge,ie)}}}else if(q.visible){let F=S(b,q,E,C);b.onBeforeShadow(i,b,P,x,X,F,null),i.renderBufferDirect(x,null,X,F,b,null),b.onAfterShadow(i,b,P,x,X,F,null)}}let L=b.children;for(let X=0,q=L.length;X<q;X++)y(L[X],P,x,E,C)}function T(b){b.target.removeEventListener("dispose",T);for(let x in l){let E=l[x],C=b.target.uuid;C in E&&(E[C].dispose(),delete E[C])}}}function Ky(i,e){function t(){let N=!1,Ee=new mt,ae=null,we=new mt(0,0,0,0);return{setMask:function(Ce){ae!==Ce&&!N&&(i.colorMask(Ce,Ce,Ce,Ce),ae=Ce)},setLocked:function(Ce){N=Ce},setClear:function(Ce,de,He,Oe,It){It===!0&&(Ce*=Oe,de*=Oe,He*=Oe),Ee.set(Ce,de,He,Oe),we.equals(Ee)===!1&&(i.clearColor(Ce,de,He,Oe),we.copy(Ee))},reset:function(){N=!1,ae=null,we.set(-1,0,0,0)}}}function n(){let N=!1,Ee=!1,ae=null,we=null,Ce=null;return{setReversed:function(de){if(Ee!==de){let He=e.get("EXT_clip_control");de?He.clipControlEXT(He.LOWER_LEFT_EXT,He.ZERO_TO_ONE_EXT):He.clipControlEXT(He.LOWER_LEFT_EXT,He.NEGATIVE_ONE_TO_ONE_EXT),Ee=de;let Oe=Ce;Ce=null,this.setClear(Oe)}},getReversed:function(){return Ee},setTest:function(de){de?le(i.DEPTH_TEST):Te(i.DEPTH_TEST)},setMask:function(de){ae!==de&&!N&&(i.depthMask(de),ae=de)},setFunc:function(de){if(Ee&&(de=np[de]),we!==de){switch(de){case cl:i.depthFunc(i.NEVER);break;case hl:i.depthFunc(i.ALWAYS);break;case ul:i.depthFunc(i.LESS);break;case Is:i.depthFunc(i.LEQUAL);break;case dl:i.depthFunc(i.EQUAL);break;case fl:i.depthFunc(i.GEQUAL);break;case pl:i.depthFunc(i.GREATER);break;case ml:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}we=de}},setLocked:function(de){N=de},setClear:function(de){Ce!==de&&(Ce=de,Ee&&(de=1-de),i.clearDepth(de))},reset:function(){N=!1,ae=null,we=null,Ce=null,Ee=!1}}}function s(){let N=!1,Ee=null,ae=null,we=null,Ce=null,de=null,He=null,Oe=null,It=null;return{setTest:function(Mt){N||(Mt?le(i.STENCIL_TEST):Te(i.STENCIL_TEST))},setMask:function(Mt){Ee!==Mt&&!N&&(i.stencilMask(Mt),Ee=Mt)},setFunc:function(Mt,ai,oi){(ae!==Mt||we!==ai||Ce!==oi)&&(i.stencilFunc(Mt,ai,oi),ae=Mt,we=ai,Ce=oi)},setOp:function(Mt,ai,oi){(de!==Mt||He!==ai||Oe!==oi)&&(i.stencilOp(Mt,ai,oi),de=Mt,He=ai,Oe=oi)},setLocked:function(Mt){N=Mt},setClear:function(Mt){It!==Mt&&(i.clearStencil(Mt),It=Mt)},reset:function(){N=!1,Ee=null,ae=null,we=null,Ce=null,de=null,He=null,Oe=null,It=null}}}let r=new t,a=new n,o=new s,c=new WeakMap,l=new WeakMap,h={},d={},u={},f=new WeakMap,g=[],_=null,p=!1,m=null,M=null,S=null,y=null,T=null,b=null,P=null,x=new Pe(0,0,0),E=0,C=!1,I=null,L=null,X=null,q=null,F=null,Y=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS),W=!1,ie=0,ne=i.getParameter(i.VERSION);ne.indexOf("WebGL")!==-1?(ie=parseFloat(/^WebGL (\d)/.exec(ne)[1]),W=ie>=1):ne.indexOf("OpenGL ES")!==-1&&(ie=parseFloat(/^OpenGL ES (\d)/.exec(ne)[1]),W=ie>=2);let ge=null,ue={},xe=i.getParameter(i.SCISSOR_BOX),Ne=i.getParameter(i.VIEWPORT),it=new mt().fromArray(xe),Xe=new mt().fromArray(Ne);function j(N,Ee,ae,we){let Ce=new Uint8Array(4),de=i.createTexture();i.bindTexture(N,de),i.texParameteri(N,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(N,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let He=0;He<ae;He++)N===i.TEXTURE_3D||N===i.TEXTURE_2D_ARRAY?i.texImage3D(Ee,0,i.RGBA,1,1,we,0,i.RGBA,i.UNSIGNED_BYTE,Ce):i.texImage2D(Ee+He,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,Ce);return de}let he={};he[i.TEXTURE_2D]=j(i.TEXTURE_2D,i.TEXTURE_2D,1),he[i.TEXTURE_CUBE_MAP]=j(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),he[i.TEXTURE_2D_ARRAY]=j(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),he[i.TEXTURE_3D]=j(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),le(i.DEPTH_TEST),a.setFunc(Is),G(!1),V(pu),le(i.CULL_FACE),H(zt);function le(N){h[N]!==!0&&(i.enable(N),h[N]=!0)}function Te(N){h[N]!==!1&&(i.disable(N),h[N]=!1)}function Fe(N,Ee){return u[N]!==Ee?(i.bindFramebuffer(N,Ee),u[N]=Ee,N===i.DRAW_FRAMEBUFFER&&(u[i.FRAMEBUFFER]=Ee),N===i.FRAMEBUFFER&&(u[i.DRAW_FRAMEBUFFER]=Ee),!0):!1}function ke(N,Ee){let ae=g,we=!1;if(N){ae=f.get(Ee),ae===void 0&&(ae=[],f.set(Ee,ae));let Ce=N.textures;if(ae.length!==Ce.length||ae[0]!==i.COLOR_ATTACHMENT0){for(let de=0,He=Ce.length;de<He;de++)ae[de]=i.COLOR_ATTACHMENT0+de;ae.length=Ce.length,we=!0}}else ae[0]!==i.BACK&&(ae[0]=i.BACK,we=!0);we&&i.drawBuffers(ae)}function oe(N){return _!==N?(i.useProgram(N),_=N,!0):!1}let ee={[Pn]:i.FUNC_ADD,[Rf]:i.FUNC_SUBTRACT,[Cf]:i.FUNC_REVERSE_SUBTRACT};ee[Pf]=i.MIN,ee[If]=i.MAX;let O={[Ns]:i.ZERO,[Df]:i.ONE,[Lf]:i.SRC_COLOR,[ol]:i.SRC_ALPHA,[Of]:i.SRC_ALPHA_SATURATE,[Ya]:i.DST_COLOR,[qa]:i.DST_ALPHA,[Uf]:i.ONE_MINUS_SRC_COLOR,[ll]:i.ONE_MINUS_SRC_ALPHA,[Ff]:i.ONE_MINUS_DST_COLOR,[Nf]:i.ONE_MINUS_DST_ALPHA,[Bf]:i.CONSTANT_COLOR,[zf]:i.ONE_MINUS_CONSTANT_COLOR,[kf]:i.CONSTANT_ALPHA,[Hf]:i.ONE_MINUS_CONSTANT_ALPHA};function H(N,Ee,ae,we,Ce,de,He,Oe,It,Mt){if(N===zt){p===!0&&(Te(i.BLEND),p=!1);return}if(p===!1&&(le(i.BLEND),p=!0),N!==$l){if(N!==m||Mt!==C){if((M!==Pn||T!==Pn)&&(i.blendEquation(i.FUNC_ADD),M=Pn,T=Pn),Mt)switch(N){case Ps:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case mu:i.blendFunc(i.ONE,i.ONE);break;case gu:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case _u:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:$e("WebGLState: Invalid blending: ",N);break}else switch(N){case Ps:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case mu:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case gu:$e("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case _u:$e("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:$e("WebGLState: Invalid blending: ",N);break}S=null,y=null,b=null,P=null,x.set(0,0,0),E=0,m=N,C=Mt}return}Ce=Ce||Ee,de=de||ae,He=He||we,(Ee!==M||Ce!==T)&&(i.blendEquationSeparate(ee[Ee],ee[Ce]),M=Ee,T=Ce),(ae!==S||we!==y||de!==b||He!==P)&&(i.blendFuncSeparate(O[ae],O[we],O[de],O[He]),S=ae,y=we,b=de,P=He),(Oe.equals(x)===!1||It!==E)&&(i.blendColor(Oe.r,Oe.g,Oe.b,It),x.copy(Oe),E=It),m=N,C=!1}function Q(N,Ee){N.side===Sn?Te(i.CULL_FACE):le(i.CULL_FACE);let ae=N.side===tn;Ee&&(ae=!ae),G(ae),N.blending===Ps&&N.transparent===!1?H(zt):H(N.blending,N.blendEquation,N.blendSrc,N.blendDst,N.blendEquationAlpha,N.blendSrcAlpha,N.blendDstAlpha,N.blendColor,N.blendAlpha,N.premultipliedAlpha),a.setFunc(N.depthFunc),a.setTest(N.depthTest),a.setMask(N.depthWrite),r.setMask(N.colorWrite);let we=N.stencilWrite;o.setTest(we),we&&(o.setMask(N.stencilWriteMask),o.setFunc(N.stencilFunc,N.stencilRef,N.stencilFuncMask),o.setOp(N.stencilFail,N.stencilZFail,N.stencilZPass)),ce(N.polygonOffset,N.polygonOffsetFactor,N.polygonOffsetUnits),N.alphaToCoverage===!0?le(i.SAMPLE_ALPHA_TO_COVERAGE):Te(i.SAMPLE_ALPHA_TO_COVERAGE)}function G(N){I!==N&&(N?i.frontFace(i.CW):i.frontFace(i.CCW),I=N)}function V(N){N!==wf?(le(i.CULL_FACE),N!==L&&(N===pu?i.cullFace(i.BACK):N===Tf?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):Te(i.CULL_FACE),L=N}function se(N){N!==X&&(W&&i.lineWidth(N),X=N)}function ce(N,Ee,ae){N?(le(i.POLYGON_OFFSET_FILL),(q!==Ee||F!==ae)&&(q=Ee,F=ae,a.getReversed()&&(Ee=-Ee),i.polygonOffset(Ee,ae))):Te(i.POLYGON_OFFSET_FILL)}function fe(N){N?le(i.SCISSOR_TEST):Te(i.SCISSOR_TEST)}function me(N){N===void 0&&(N=i.TEXTURE0+Y-1),ge!==N&&(i.activeTexture(N),ge=N)}function D(N,Ee,ae){ae===void 0&&(ge===null?ae=i.TEXTURE0+Y-1:ae=ge);let we=ue[ae];we===void 0&&(we={type:void 0,texture:void 0},ue[ae]=we),(we.type!==N||we.texture!==Ee)&&(ge!==ae&&(i.activeTexture(ae),ge=ae),i.bindTexture(N,Ee||he[N]),we.type=N,we.texture=Ee)}function Me(){let N=ue[ge];N!==void 0&&N.type!==void 0&&(i.bindTexture(N.type,null),N.type=void 0,N.texture=void 0)}function Ve(){try{i.compressedTexImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function A(){try{i.compressedTexImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function v(){try{i.texSubImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function U(){try{i.texSubImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function B(){try{i.compressedTexSubImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function k(){try{i.compressedTexSubImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function pe(){try{i.texStorage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function _e(){try{i.texStorage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function te(){try{i.texImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function re(){try{i.texImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function Se(N){return d[N]!==void 0?d[N]:i.getParameter(N)}function Ie(N,Ee){d[N]!==Ee&&(i.pixelStorei(N,Ee),d[N]=Ee)}function ve(N){it.equals(N)===!1&&(i.scissor(N.x,N.y,N.z,N.w),it.copy(N))}function ye(N){Xe.equals(N)===!1&&(i.viewport(N.x,N.y,N.z,N.w),Xe.copy(N))}function Be(N,Ee){let ae=l.get(Ee);ae===void 0&&(ae=new WeakMap,l.set(Ee,ae));let we=ae.get(N);we===void 0&&(we=i.getUniformBlockIndex(Ee,N.name),ae.set(N,we))}function qe(N,Ee){let we=l.get(Ee).get(N);c.get(Ee)!==we&&(i.uniformBlockBinding(Ee,we,N.__bindingPointIndex),c.set(Ee,we))}function Je(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),a.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),i.pixelStorei(i.PACK_ALIGNMENT,4),i.pixelStorei(i.UNPACK_ALIGNMENT,4),i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,!1),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,i.BROWSER_DEFAULT_WEBGL),i.pixelStorei(i.PACK_ROW_LENGTH,0),i.pixelStorei(i.PACK_SKIP_PIXELS,0),i.pixelStorei(i.PACK_SKIP_ROWS,0),i.pixelStorei(i.UNPACK_ROW_LENGTH,0),i.pixelStorei(i.UNPACK_IMAGE_HEIGHT,0),i.pixelStorei(i.UNPACK_SKIP_PIXELS,0),i.pixelStorei(i.UNPACK_SKIP_ROWS,0),i.pixelStorei(i.UNPACK_SKIP_IMAGES,0),h={},d={},ge=null,ue={},u={},f=new WeakMap,g=[],_=null,p=!1,m=null,M=null,S=null,y=null,T=null,b=null,P=null,x=new Pe(0,0,0),E=0,C=!1,I=null,L=null,X=null,q=null,F=null,it.set(0,0,i.canvas.width,i.canvas.height),Xe.set(0,0,i.canvas.width,i.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:le,disable:Te,bindFramebuffer:Fe,drawBuffers:ke,useProgram:oe,setBlending:H,setMaterial:Q,setFlipSided:G,setCullFace:V,setLineWidth:se,setPolygonOffset:ce,setScissorTest:fe,activeTexture:me,bindTexture:D,unbindTexture:Me,compressedTexImage2D:Ve,compressedTexImage3D:A,texImage2D:te,texImage3D:re,pixelStorei:Ie,getParameter:Se,updateUBOMapping:Be,uniformBlockBinding:qe,texStorage2D:pe,texStorage3D:_e,texSubImage2D:v,texSubImage3D:U,compressedTexSubImage2D:B,compressedTexSubImage3D:k,scissor:ve,viewport:ye,reset:Je}}function Qy(i,e,t,n,s,r,a){let o=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new Z,h=new WeakMap,d=new Set,u,f=new WeakMap,g=!1;try{g=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function _(A,v){return g?new OffscreenCanvas(A,v):ca("canvas")}function p(A,v,U){let B=1,k=Ve(A);if((k.width>U||k.height>U)&&(B=U/Math.max(k.width,k.height)),B<1)if(typeof HTMLImageElement<"u"&&A instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&A instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&A instanceof ImageBitmap||typeof VideoFrame<"u"&&A instanceof VideoFrame){let pe=Math.floor(B*k.width),_e=Math.floor(B*k.height);u===void 0&&(u=_(pe,_e));let te=v?_(pe,_e):u;return te.width=pe,te.height=_e,te.getContext("2d").drawImage(A,0,0,pe,_e),Ze("WebGLRenderer: Texture has been resized from ("+k.width+"x"+k.height+") to ("+pe+"x"+_e+")."),te}else return"data"in A&&Ze("WebGLRenderer: Image in DataTexture is too big ("+k.width+"x"+k.height+")."),A;return A}function m(A){return A.generateMipmaps}function M(A){i.generateMipmap(A)}function S(A){return A.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:A.isWebGL3DRenderTarget?i.TEXTURE_3D:A.isWebGLArrayRenderTarget||A.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function y(A,v,U,B,k,pe=!1){if(A!==null){if(i[A]!==void 0)return i[A];Ze("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+A+"'")}let _e;B&&(_e=e.get("EXT_texture_norm16"),_e||Ze("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let te=v;if(v===i.RED&&(U===i.FLOAT&&(te=i.R32F),U===i.HALF_FLOAT&&(te=i.R16F),U===i.UNSIGNED_BYTE&&(te=i.R8),U===i.UNSIGNED_SHORT&&_e&&(te=_e.R16_EXT),U===i.SHORT&&_e&&(te=_e.R16_SNORM_EXT)),v===i.RED_INTEGER&&(U===i.UNSIGNED_BYTE&&(te=i.R8UI),U===i.UNSIGNED_SHORT&&(te=i.R16UI),U===i.UNSIGNED_INT&&(te=i.R32UI),U===i.BYTE&&(te=i.R8I),U===i.SHORT&&(te=i.R16I),U===i.INT&&(te=i.R32I)),v===i.RG&&(U===i.FLOAT&&(te=i.RG32F),U===i.HALF_FLOAT&&(te=i.RG16F),U===i.UNSIGNED_BYTE&&(te=i.RG8),U===i.UNSIGNED_SHORT&&_e&&(te=_e.RG16_EXT),U===i.SHORT&&_e&&(te=_e.RG16_SNORM_EXT)),v===i.RG_INTEGER&&(U===i.UNSIGNED_BYTE&&(te=i.RG8UI),U===i.UNSIGNED_SHORT&&(te=i.RG16UI),U===i.UNSIGNED_INT&&(te=i.RG32UI),U===i.BYTE&&(te=i.RG8I),U===i.SHORT&&(te=i.RG16I),U===i.INT&&(te=i.RG32I)),v===i.RGB_INTEGER&&(U===i.UNSIGNED_BYTE&&(te=i.RGB8UI),U===i.UNSIGNED_SHORT&&(te=i.RGB16UI),U===i.UNSIGNED_INT&&(te=i.RGB32UI),U===i.BYTE&&(te=i.RGB8I),U===i.SHORT&&(te=i.RGB16I),U===i.INT&&(te=i.RGB32I)),v===i.RGBA_INTEGER&&(U===i.UNSIGNED_BYTE&&(te=i.RGBA8UI),U===i.UNSIGNED_SHORT&&(te=i.RGBA16UI),U===i.UNSIGNED_INT&&(te=i.RGBA32UI),U===i.BYTE&&(te=i.RGBA8I),U===i.SHORT&&(te=i.RGBA16I),U===i.INT&&(te=i.RGBA32I)),v===i.RGB&&(U===i.UNSIGNED_SHORT&&_e&&(te=_e.RGB16_EXT),U===i.SHORT&&_e&&(te=_e.RGB16_SNORM_EXT),U===i.UNSIGNED_INT_5_9_9_9_REV&&(te=i.RGB9_E5),U===i.UNSIGNED_INT_10F_11F_11F_REV&&(te=i.R11F_G11F_B10F)),v===i.RGBA){let re=pe?la:ht.getTransfer(k);U===i.FLOAT&&(te=i.RGBA32F),U===i.HALF_FLOAT&&(te=i.RGBA16F),U===i.UNSIGNED_BYTE&&(te=re===pt?i.SRGB8_ALPHA8:i.RGBA8),U===i.UNSIGNED_SHORT&&_e&&(te=_e.RGBA16_EXT),U===i.SHORT&&_e&&(te=_e.RGBA16_SNORM_EXT),U===i.UNSIGNED_SHORT_4_4_4_4&&(te=i.RGBA4),U===i.UNSIGNED_SHORT_5_5_5_1&&(te=i.RGB5_A1)}return(te===i.R16F||te===i.R32F||te===i.RG16F||te===i.RG32F||te===i.RGBA16F||te===i.RGBA32F)&&e.get("EXT_color_buffer_float"),te}function T(A,v){let U;return A?v===null||v===ni||v===fs?U=i.DEPTH24_STENCIL8:v===Vn?U=i.DEPTH32F_STENCIL8:v===Dr&&(U=i.DEPTH24_STENCIL8,Ze("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):v===null||v===ni||v===fs?U=i.DEPTH_COMPONENT24:v===Vn?U=i.DEPTH_COMPONENT32F:v===Dr&&(U=i.DEPTH_COMPONENT16),U}function b(A,v){return m(A)===!0||A.isFramebufferTexture&&A.minFilter!==Ot&&A.minFilter!==en?Math.log2(Math.max(v.width,v.height))+1:A.mipmaps!==void 0&&A.mipmaps.length>0?A.mipmaps.length:A.isCompressedTexture&&Array.isArray(A.image)?v.mipmaps.length:1}function P(A){let v=A.target;v.removeEventListener("dispose",P),E(v),v.isVideoTexture&&h.delete(v),v.isHTMLTexture&&d.delete(v)}function x(A){let v=A.target;v.removeEventListener("dispose",x),I(v)}function E(A){let v=n.get(A);if(v.__webglInit===void 0)return;let U=A.source,B=f.get(U);if(B){let k=B[v.__cacheKey];k.usedTimes--,k.usedTimes===0&&C(A),Object.keys(B).length===0&&f.delete(U)}n.remove(A)}function C(A){let v=n.get(A);i.deleteTexture(v.__webglTexture);let U=A.source,B=f.get(U);delete B[v.__cacheKey],a.memory.textures--}function I(A){let v=n.get(A);if(A.depthTexture&&(A.depthTexture.dispose(),n.remove(A.depthTexture)),A.isWebGLCubeRenderTarget)for(let B=0;B<6;B++){if(Array.isArray(v.__webglFramebuffer[B]))for(let k=0;k<v.__webglFramebuffer[B].length;k++)i.deleteFramebuffer(v.__webglFramebuffer[B][k]);else i.deleteFramebuffer(v.__webglFramebuffer[B]);v.__webglDepthbuffer&&i.deleteRenderbuffer(v.__webglDepthbuffer[B])}else{if(Array.isArray(v.__webglFramebuffer))for(let B=0;B<v.__webglFramebuffer.length;B++)i.deleteFramebuffer(v.__webglFramebuffer[B]);else i.deleteFramebuffer(v.__webglFramebuffer);if(v.__webglDepthbuffer&&i.deleteRenderbuffer(v.__webglDepthbuffer),v.__webglMultisampledFramebuffer&&i.deleteFramebuffer(v.__webglMultisampledFramebuffer),v.__webglColorRenderbuffer)for(let B=0;B<v.__webglColorRenderbuffer.length;B++)v.__webglColorRenderbuffer[B]&&i.deleteRenderbuffer(v.__webglColorRenderbuffer[B]);v.__webglDepthRenderbuffer&&i.deleteRenderbuffer(v.__webglDepthRenderbuffer)}let U=A.textures;for(let B=0,k=U.length;B<k;B++){let pe=n.get(U[B]);pe.__webglTexture&&(i.deleteTexture(pe.__webglTexture),a.memory.textures--),n.remove(U[B])}n.remove(A)}let L=0;function X(){L=0}function q(){return L}function F(A){L=A}function Y(){let A=L;return A>=s.maxTextures&&Ze("WebGLTextures: Trying to use "+A+" texture units while this GPU supports only "+s.maxTextures),L+=1,A}function W(A){let v=[];return v.push(A.wrapS),v.push(A.wrapT),v.push(A.wrapR||0),v.push(A.magFilter),v.push(A.minFilter),v.push(A.anisotropy),v.push(A.internalFormat),v.push(A.format),v.push(A.type),v.push(A.generateMipmaps),v.push(A.premultiplyAlpha),v.push(A.flipY),v.push(A.unpackAlignment),v.push(A.colorSpace),v.join()}function ie(A,v){let U=n.get(A);if(A.isVideoTexture&&D(A),A.isRenderTargetTexture===!1&&A.isExternalTexture!==!0&&A.version>0&&U.__version!==A.version){let B=A.image;if(B===null)Ze("WebGLRenderer: Texture marked for update but no image data found.");else if(B.complete===!1)Ze("WebGLRenderer: Texture marked for update but image is incomplete");else{Te(U,A,v);return}}else A.isExternalTexture&&(U.__webglTexture=A.sourceTexture?A.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,U.__webglTexture,i.TEXTURE0+v)}function ne(A,v){let U=n.get(A);if(A.isRenderTargetTexture===!1&&A.version>0&&U.__version!==A.version){Te(U,A,v);return}else A.isExternalTexture&&(U.__webglTexture=A.sourceTexture?A.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,U.__webglTexture,i.TEXTURE0+v)}function ge(A,v){let U=n.get(A);if(A.isRenderTargetTexture===!1&&A.version>0&&U.__version!==A.version){Te(U,A,v);return}t.bindTexture(i.TEXTURE_3D,U.__webglTexture,i.TEXTURE0+v)}function ue(A,v){let U=n.get(A);if(A.isCubeDepthTexture!==!0&&A.version>0&&U.__version!==A.version){Fe(U,A,v);return}t.bindTexture(i.TEXTURE_CUBE_MAP,U.__webglTexture,i.TEXTURE0+v)}let xe={[kn]:i.REPEAT,[ui]:i.CLAMP_TO_EDGE,[gl]:i.MIRRORED_REPEAT},Ne={[Ot]:i.NEAREST,[Wf]:i.NEAREST_MIPMAP_NEAREST,[eo]:i.NEAREST_MIPMAP_LINEAR,[en]:i.LINEAR,[Ql]:i.LINEAR_MIPMAP_NEAREST,[ds]:i.LINEAR_MIPMAP_LINEAR},it={[Yf]:i.NEVER,[Kf]:i.ALWAYS,[Zf]:i.LESS,[Oc]:i.LEQUAL,[$f]:i.EQUAL,[Bc]:i.GEQUAL,[Jf]:i.GREATER,[jf]:i.NOTEQUAL};function Xe(A,v){if(v.type===Vn&&e.has("OES_texture_float_linear")===!1&&(v.magFilter===en||v.magFilter===Ql||v.magFilter===eo||v.magFilter===ds||v.minFilter===en||v.minFilter===Ql||v.minFilter===eo||v.minFilter===ds)&&Ze("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(A,i.TEXTURE_WRAP_S,xe[v.wrapS]),i.texParameteri(A,i.TEXTURE_WRAP_T,xe[v.wrapT]),(A===i.TEXTURE_3D||A===i.TEXTURE_2D_ARRAY)&&i.texParameteri(A,i.TEXTURE_WRAP_R,xe[v.wrapR]),i.texParameteri(A,i.TEXTURE_MAG_FILTER,Ne[v.magFilter]),i.texParameteri(A,i.TEXTURE_MIN_FILTER,Ne[v.minFilter]),v.compareFunction&&(i.texParameteri(A,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(A,i.TEXTURE_COMPARE_FUNC,it[v.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(v.magFilter===Ot||v.minFilter!==eo&&v.minFilter!==ds||v.type===Vn&&e.has("OES_texture_float_linear")===!1)return;if(v.anisotropy>1||n.get(v).__currentAnisotropy){let U=e.get("EXT_texture_filter_anisotropic");i.texParameterf(A,U.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(v.anisotropy,s.getMaxAnisotropy())),n.get(v).__currentAnisotropy=v.anisotropy}}}function j(A,v){let U=!1;A.__webglInit===void 0&&(A.__webglInit=!0,v.addEventListener("dispose",P));let B=v.source,k=f.get(B);k===void 0&&(k={},f.set(B,k));let pe=W(v);if(pe!==A.__cacheKey){k[pe]===void 0&&(k[pe]={texture:i.createTexture(),usedTimes:0},a.memory.textures++,U=!0),k[pe].usedTimes++;let _e=k[A.__cacheKey];_e!==void 0&&(k[A.__cacheKey].usedTimes--,_e.usedTimes===0&&C(v)),A.__cacheKey=pe,A.__webglTexture=k[pe].texture}return U}function he(A,v,U){return Math.floor(Math.floor(A/U)/v)}function le(A,v,U,B){let pe=A.updateRanges;if(pe.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,v.width,v.height,U,B,v.data);else{pe.sort((Ie,ve)=>Ie.start-ve.start);let _e=0;for(let Ie=1;Ie<pe.length;Ie++){let ve=pe[_e],ye=pe[Ie],Be=ve.start+ve.count,qe=he(ye.start,v.width,4),Je=he(ve.start,v.width,4);ye.start<=Be+1&&qe===Je&&he(ye.start+ye.count-1,v.width,4)===qe?ve.count=Math.max(ve.count,ye.start+ye.count-ve.start):(++_e,pe[_e]=ye)}pe.length=_e+1;let te=t.getParameter(i.UNPACK_ROW_LENGTH),re=t.getParameter(i.UNPACK_SKIP_PIXELS),Se=t.getParameter(i.UNPACK_SKIP_ROWS);t.pixelStorei(i.UNPACK_ROW_LENGTH,v.width);for(let Ie=0,ve=pe.length;Ie<ve;Ie++){let ye=pe[Ie],Be=Math.floor(ye.start/4),qe=Math.ceil(ye.count/4),Je=Be%v.width,N=Math.floor(Be/v.width),Ee=qe,ae=1;t.pixelStorei(i.UNPACK_SKIP_PIXELS,Je),t.pixelStorei(i.UNPACK_SKIP_ROWS,N),t.texSubImage2D(i.TEXTURE_2D,0,Je,N,Ee,ae,U,B,v.data)}A.clearUpdateRanges(),t.pixelStorei(i.UNPACK_ROW_LENGTH,te),t.pixelStorei(i.UNPACK_SKIP_PIXELS,re),t.pixelStorei(i.UNPACK_SKIP_ROWS,Se)}}function Te(A,v,U){let B=i.TEXTURE_2D;(v.isDataArrayTexture||v.isCompressedArrayTexture)&&(B=i.TEXTURE_2D_ARRAY),v.isData3DTexture&&(B=i.TEXTURE_3D);let k=j(A,v),pe=v.source;t.bindTexture(B,A.__webglTexture,i.TEXTURE0+U);let _e=n.get(pe);if(pe.version!==_e.__version||k===!0){if(t.activeTexture(i.TEXTURE0+U),(typeof ImageBitmap<"u"&&v.image instanceof ImageBitmap)===!1){let ae=ht.getPrimaries(ht.workingColorSpace),we=v.colorSpace===Oi?null:ht.getPrimaries(v.colorSpace),Ce=v.colorSpace===Oi||ae===we?i.NONE:i.BROWSER_DEFAULT_WEBGL;t.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,v.flipY),t.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),t.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ce)}t.pixelStorei(i.UNPACK_ALIGNMENT,v.unpackAlignment);let re=p(v.image,!1,s.maxTextureSize);re=Me(v,re);let Se=r.convert(v.format,v.colorSpace),Ie=r.convert(v.type),ve=y(v.internalFormat,Se,Ie,v.normalized,v.colorSpace,v.isVideoTexture);Xe(B,v);let ye,Be=v.mipmaps,qe=v.isVideoTexture!==!0,Je=_e.__version===void 0||k===!0,N=pe.dataReady,Ee=b(v,re);if(v.isDepthTexture)ve=T(v.format===_i,v.type),Je&&(qe?t.texStorage2D(i.TEXTURE_2D,1,ve,re.width,re.height):t.texImage2D(i.TEXTURE_2D,0,ve,re.width,re.height,0,Se,Ie,null));else if(v.isDataTexture)if(Be.length>0){qe&&Je&&t.texStorage2D(i.TEXTURE_2D,Ee,ve,Be[0].width,Be[0].height);for(let ae=0,we=Be.length;ae<we;ae++)ye=Be[ae],qe?N&&t.texSubImage2D(i.TEXTURE_2D,ae,0,0,ye.width,ye.height,Se,Ie,ye.data):t.texImage2D(i.TEXTURE_2D,ae,ve,ye.width,ye.height,0,Se,Ie,ye.data);v.generateMipmaps=!1}else qe?(Je&&t.texStorage2D(i.TEXTURE_2D,Ee,ve,re.width,re.height),N&&le(v,re,Se,Ie)):t.texImage2D(i.TEXTURE_2D,0,ve,re.width,re.height,0,Se,Ie,re.data);else if(v.isCompressedTexture)if(v.isCompressedArrayTexture){qe&&Je&&t.texStorage3D(i.TEXTURE_2D_ARRAY,Ee,ve,Be[0].width,Be[0].height,re.depth);for(let ae=0,we=Be.length;ae<we;ae++)if(ye=Be[ae],v.format!==bn)if(Se!==null)if(qe){if(N)if(v.layerUpdates.size>0){let Ce=Iu(ye.width,ye.height,v.format,v.type);for(let de of v.layerUpdates){let He=ye.data.subarray(de*Ce/ye.data.BYTES_PER_ELEMENT,(de+1)*Ce/ye.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,ae,0,0,de,ye.width,ye.height,1,Se,He)}v.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,ae,0,0,0,ye.width,ye.height,re.depth,Se,ye.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,ae,ve,ye.width,ye.height,re.depth,0,ye.data,0,0);else Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else qe?N&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,ae,0,0,0,ye.width,ye.height,re.depth,Se,Ie,ye.data):t.texImage3D(i.TEXTURE_2D_ARRAY,ae,ve,ye.width,ye.height,re.depth,0,Se,Ie,ye.data)}else{qe&&Je&&t.texStorage2D(i.TEXTURE_2D,Ee,ve,Be[0].width,Be[0].height);for(let ae=0,we=Be.length;ae<we;ae++)ye=Be[ae],v.format!==bn?Se!==null?qe?N&&t.compressedTexSubImage2D(i.TEXTURE_2D,ae,0,0,ye.width,ye.height,Se,ye.data):t.compressedTexImage2D(i.TEXTURE_2D,ae,ve,ye.width,ye.height,0,ye.data):Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):qe?N&&t.texSubImage2D(i.TEXTURE_2D,ae,0,0,ye.width,ye.height,Se,Ie,ye.data):t.texImage2D(i.TEXTURE_2D,ae,ve,ye.width,ye.height,0,Se,Ie,ye.data)}else if(v.isDataArrayTexture)if(qe){if(Je&&t.texStorage3D(i.TEXTURE_2D_ARRAY,Ee,ve,re.width,re.height,re.depth),N)if(v.layerUpdates.size>0){let ae=Iu(re.width,re.height,v.format,v.type);for(let we of v.layerUpdates){let Ce=re.data.subarray(we*ae/re.data.BYTES_PER_ELEMENT,(we+1)*ae/re.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,we,re.width,re.height,1,Se,Ie,Ce)}v.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,re.width,re.height,re.depth,Se,Ie,re.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,ve,re.width,re.height,re.depth,0,Se,Ie,re.data);else if(v.isData3DTexture)qe?(Je&&t.texStorage3D(i.TEXTURE_3D,Ee,ve,re.width,re.height,re.depth),N&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,re.width,re.height,re.depth,Se,Ie,re.data)):t.texImage3D(i.TEXTURE_3D,0,ve,re.width,re.height,re.depth,0,Se,Ie,re.data);else if(v.isFramebufferTexture){if(Je)if(qe)t.texStorage2D(i.TEXTURE_2D,Ee,ve,re.width,re.height);else{let ae=re.width,we=re.height;for(let Ce=0;Ce<Ee;Ce++)t.texImage2D(i.TEXTURE_2D,Ce,ve,ae,we,0,Se,Ie,null),ae>>=1,we>>=1}}else if(v.isHTMLTexture){if("texElementImage2D"in i){let ae=i.canvas;if(ae.hasAttribute("layoutsubtree")||ae.setAttribute("layoutsubtree","true"),re.parentNode!==ae){ae.appendChild(re),d.add(v),ae.onpaint=we=>{let Ce=we.changedElements;for(let de of d)Ce.includes(de.image)&&(de.needsUpdate=!0)},ae.requestPaint();return}if(i.texElementImage2D.length===3)i.texElementImage2D(i.TEXTURE_2D,i.RGBA8,re);else{let Ce=i.RGBA,de=i.RGBA,He=i.UNSIGNED_BYTE;i.texElementImage2D(i.TEXTURE_2D,0,Ce,de,He,re)}i.texParameteri(i.TEXTURE_2D,i.TEXTURE_MIN_FILTER,i.LINEAR),i.texParameteri(i.TEXTURE_2D,i.TEXTURE_WRAP_S,i.CLAMP_TO_EDGE),i.texParameteri(i.TEXTURE_2D,i.TEXTURE_WRAP_T,i.CLAMP_TO_EDGE)}}else if(Be.length>0){if(qe&&Je){let ae=Ve(Be[0]);t.texStorage2D(i.TEXTURE_2D,Ee,ve,ae.width,ae.height)}for(let ae=0,we=Be.length;ae<we;ae++)ye=Be[ae],qe?N&&t.texSubImage2D(i.TEXTURE_2D,ae,0,0,Se,Ie,ye):t.texImage2D(i.TEXTURE_2D,ae,ve,Se,Ie,ye);v.generateMipmaps=!1}else if(qe){if(Je){let ae=Ve(re);t.texStorage2D(i.TEXTURE_2D,Ee,ve,ae.width,ae.height)}N&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,Se,Ie,re)}else t.texImage2D(i.TEXTURE_2D,0,ve,Se,Ie,re);m(v)&&M(B),_e.__version=pe.version,v.onUpdate&&v.onUpdate(v)}A.__version=v.version}function Fe(A,v,U){if(v.image.length!==6)return;let B=j(A,v),k=v.source;t.bindTexture(i.TEXTURE_CUBE_MAP,A.__webglTexture,i.TEXTURE0+U);let pe=n.get(k);if(k.version!==pe.__version||B===!0){t.activeTexture(i.TEXTURE0+U);let _e=ht.getPrimaries(ht.workingColorSpace),te=v.colorSpace===Oi?null:ht.getPrimaries(v.colorSpace),re=v.colorSpace===Oi||_e===te?i.NONE:i.BROWSER_DEFAULT_WEBGL;t.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,v.flipY),t.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),t.pixelStorei(i.UNPACK_ALIGNMENT,v.unpackAlignment),t.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,re);let Se=v.isCompressedTexture||v.image[0].isCompressedTexture,Ie=v.image[0]&&v.image[0].isDataTexture,ve=[];for(let de=0;de<6;de++)!Se&&!Ie?ve[de]=p(v.image[de],!0,s.maxCubemapSize):ve[de]=Ie?v.image[de].image:v.image[de],ve[de]=Me(v,ve[de]);let ye=ve[0],Be=r.convert(v.format,v.colorSpace),qe=r.convert(v.type),Je=y(v.internalFormat,Be,qe,v.normalized,v.colorSpace),N=v.isVideoTexture!==!0,Ee=pe.__version===void 0||B===!0,ae=k.dataReady,we=b(v,ye);Xe(i.TEXTURE_CUBE_MAP,v);let Ce;if(Se){N&&Ee&&t.texStorage2D(i.TEXTURE_CUBE_MAP,we,Je,ye.width,ye.height);for(let de=0;de<6;de++){Ce=ve[de].mipmaps;for(let He=0;He<Ce.length;He++){let Oe=Ce[He];v.format!==bn?Be!==null?N?ae&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,0,0,Oe.width,Oe.height,Be,Oe.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,Je,Oe.width,Oe.height,0,Oe.data):Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):N?ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,0,0,Oe.width,Oe.height,Be,qe,Oe.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He,Je,Oe.width,Oe.height,0,Be,qe,Oe.data)}}}else{if(Ce=v.mipmaps,N&&Ee){Ce.length>0&&we++;let de=Ve(ve[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,we,Je,de.width,de.height)}for(let de=0;de<6;de++)if(Ie){N?ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,ve[de].width,ve[de].height,Be,qe,ve[de].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,Je,ve[de].width,ve[de].height,0,Be,qe,ve[de].data);for(let He=0;He<Ce.length;He++){let It=Ce[He].image[de].image;N?ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,0,0,It.width,It.height,Be,qe,It.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,Je,It.width,It.height,0,Be,qe,It.data)}}else{N?ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,Be,qe,ve[de]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,Je,Be,qe,ve[de]);for(let He=0;He<Ce.length;He++){let Oe=Ce[He];N?ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,0,0,Be,qe,Oe.image[de]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,He+1,Je,Be,qe,Oe.image[de])}}}m(v)&&M(i.TEXTURE_CUBE_MAP),pe.__version=k.version,v.onUpdate&&v.onUpdate(v)}A.__version=v.version}function ke(A,v,U,B,k,pe){let _e=r.convert(U.format,U.colorSpace),te=r.convert(U.type),re=y(U.internalFormat,_e,te,U.normalized,U.colorSpace),Se=n.get(v),Ie=n.get(U);if(Ie.__renderTarget=v,!Se.__hasExternalTextures){let ve=Math.max(1,v.width>>pe),ye=Math.max(1,v.height>>pe);k===i.TEXTURE_3D||k===i.TEXTURE_2D_ARRAY?t.texImage3D(k,pe,re,ve,ye,v.depth,0,_e,te,null):t.texImage2D(k,pe,re,ve,ye,0,_e,te,null)}t.bindFramebuffer(i.FRAMEBUFFER,A),me(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,B,k,Ie.__webglTexture,0,fe(v)):(k===i.TEXTURE_2D||k>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&k<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,B,k,Ie.__webglTexture,pe),t.bindFramebuffer(i.FRAMEBUFFER,null)}function oe(A,v,U){if(i.bindRenderbuffer(i.RENDERBUFFER,A),v.depthBuffer){let B=v.depthTexture,k=B&&B.isDepthTexture?B.type:null,pe=T(v.stencilBuffer,k),_e=v.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;me(v)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,fe(v),pe,v.width,v.height):U?i.renderbufferStorageMultisample(i.RENDERBUFFER,fe(v),pe,v.width,v.height):i.renderbufferStorage(i.RENDERBUFFER,pe,v.width,v.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,_e,i.RENDERBUFFER,A)}else{let B=v.textures;for(let k=0;k<B.length;k++){let pe=B[k],_e=r.convert(pe.format,pe.colorSpace),te=r.convert(pe.type),re=y(pe.internalFormat,_e,te,pe.normalized,pe.colorSpace);me(v)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,fe(v),re,v.width,v.height):U?i.renderbufferStorageMultisample(i.RENDERBUFFER,fe(v),re,v.width,v.height):i.renderbufferStorage(i.RENDERBUFFER,re,v.width,v.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function ee(A,v,U){let B=v.isWebGLCubeRenderTarget===!0;if(t.bindFramebuffer(i.FRAMEBUFFER,A),!(v.depthTexture&&v.depthTexture.isDepthTexture))throw new Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");let k=n.get(v.depthTexture);if(k.__renderTarget=v,(!k.__webglTexture||v.depthTexture.image.width!==v.width||v.depthTexture.image.height!==v.height)&&(v.depthTexture.image.width=v.width,v.depthTexture.image.height=v.height,v.depthTexture.needsUpdate=!0),B){if(k.__webglInit===void 0&&(k.__webglInit=!0,v.depthTexture.addEventListener("dispose",P)),k.__webglTexture===void 0){k.__webglTexture=i.createTexture(),t.bindTexture(i.TEXTURE_CUBE_MAP,k.__webglTexture),Xe(i.TEXTURE_CUBE_MAP,v.depthTexture);let Se=r.convert(v.depthTexture.format),Ie=r.convert(v.depthTexture.type),ve;v.depthTexture.format===fi?ve=i.DEPTH_COMPONENT24:v.depthTexture.format===_i&&(ve=i.DEPTH24_STENCIL8);for(let ye=0;ye<6;ye++)i.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ye,0,ve,v.width,v.height,0,Se,Ie,null)}}else ie(v.depthTexture,0);let pe=k.__webglTexture,_e=fe(v),te=B?i.TEXTURE_CUBE_MAP_POSITIVE_X+U:i.TEXTURE_2D,re=v.depthTexture.format===_i?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;if(v.depthTexture.format===fi)me(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,re,te,pe,0,_e):i.framebufferTexture2D(i.FRAMEBUFFER,re,te,pe,0);else if(v.depthTexture.format===_i)me(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,re,te,pe,0,_e):i.framebufferTexture2D(i.FRAMEBUFFER,re,te,pe,0);else throw new Error("THREE.WebGLTextures: Unknown depthTexture format.")}function O(A){let v=n.get(A),U=A.isWebGLCubeRenderTarget===!0;if(v.__boundDepthTexture!==A.depthTexture){let B=A.depthTexture;if(v.__depthDisposeCallback&&v.__depthDisposeCallback(),B){let k=()=>{delete v.__boundDepthTexture,delete v.__depthDisposeCallback,B.removeEventListener("dispose",k)};B.addEventListener("dispose",k),v.__depthDisposeCallback=k}v.__boundDepthTexture=B}if(A.depthTexture&&!v.__autoAllocateDepthBuffer)if(U)for(let B=0;B<6;B++)ee(v.__webglFramebuffer[B],A,B);else{let B=A.texture.mipmaps;B&&B.length>0?ee(v.__webglFramebuffer[0],A,0):ee(v.__webglFramebuffer,A,0)}else if(U){v.__webglDepthbuffer=[];for(let B=0;B<6;B++)if(t.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer[B]),v.__webglDepthbuffer[B]===void 0)v.__webglDepthbuffer[B]=i.createRenderbuffer(),oe(v.__webglDepthbuffer[B],A,!1);else{let k=A.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,pe=v.__webglDepthbuffer[B];i.bindRenderbuffer(i.RENDERBUFFER,pe),i.framebufferRenderbuffer(i.FRAMEBUFFER,k,i.RENDERBUFFER,pe)}}else{let B=A.texture.mipmaps;if(B&&B.length>0?t.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer),v.__webglDepthbuffer===void 0)v.__webglDepthbuffer=i.createRenderbuffer(),oe(v.__webglDepthbuffer,A,!1);else{let k=A.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,pe=v.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,pe),i.framebufferRenderbuffer(i.FRAMEBUFFER,k,i.RENDERBUFFER,pe)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function H(A,v,U){let B=n.get(A);v!==void 0&&ke(B.__webglFramebuffer,A,A.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),U!==void 0&&O(A)}function Q(A){let v=A.texture,U=n.get(A),B=n.get(v);A.addEventListener("dispose",x);let k=A.textures,pe=A.isWebGLCubeRenderTarget===!0,_e=k.length>1;if(_e||(B.__webglTexture===void 0&&(B.__webglTexture=i.createTexture()),B.__version=v.version,a.memory.textures++),pe){U.__webglFramebuffer=[];for(let te=0;te<6;te++)if(v.mipmaps&&v.mipmaps.length>0){U.__webglFramebuffer[te]=[];for(let re=0;re<v.mipmaps.length;re++)U.__webglFramebuffer[te][re]=i.createFramebuffer()}else U.__webglFramebuffer[te]=i.createFramebuffer()}else{if(v.mipmaps&&v.mipmaps.length>0){U.__webglFramebuffer=[];for(let te=0;te<v.mipmaps.length;te++)U.__webglFramebuffer[te]=i.createFramebuffer()}else U.__webglFramebuffer=i.createFramebuffer();if(_e)for(let te=0,re=k.length;te<re;te++){let Se=n.get(k[te]);Se.__webglTexture===void 0&&(Se.__webglTexture=i.createTexture(),a.memory.textures++)}if(A.samples>0&&me(A)===!1){U.__webglMultisampledFramebuffer=i.createFramebuffer(),U.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,U.__webglMultisampledFramebuffer);for(let te=0;te<k.length;te++){let re=k[te];U.__webglColorRenderbuffer[te]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,U.__webglColorRenderbuffer[te]);let Se=r.convert(re.format,re.colorSpace),Ie=r.convert(re.type),ve=y(re.internalFormat,Se,Ie,re.normalized,re.colorSpace,A.isXRRenderTarget===!0),ye=fe(A);i.renderbufferStorageMultisample(i.RENDERBUFFER,ye,ve,A.width,A.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+te,i.RENDERBUFFER,U.__webglColorRenderbuffer[te])}i.bindRenderbuffer(i.RENDERBUFFER,null),A.depthBuffer&&(U.__webglDepthRenderbuffer=i.createRenderbuffer(),oe(U.__webglDepthRenderbuffer,A,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(pe){t.bindTexture(i.TEXTURE_CUBE_MAP,B.__webglTexture),Xe(i.TEXTURE_CUBE_MAP,v);for(let te=0;te<6;te++)if(v.mipmaps&&v.mipmaps.length>0)for(let re=0;re<v.mipmaps.length;re++)ke(U.__webglFramebuffer[te][re],A,v,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+te,re);else ke(U.__webglFramebuffer[te],A,v,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+te,0);m(v)&&M(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(_e){for(let te=0,re=k.length;te<re;te++){let Se=k[te],Ie=n.get(Se),ve=i.TEXTURE_2D;(A.isWebGL3DRenderTarget||A.isWebGLArrayRenderTarget)&&(ve=A.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(ve,Ie.__webglTexture),Xe(ve,Se),ke(U.__webglFramebuffer,A,Se,i.COLOR_ATTACHMENT0+te,ve,0),m(Se)&&M(ve)}t.unbindTexture()}else{let te=i.TEXTURE_2D;if((A.isWebGL3DRenderTarget||A.isWebGLArrayRenderTarget)&&(te=A.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(te,B.__webglTexture),Xe(te,v),v.mipmaps&&v.mipmaps.length>0)for(let re=0;re<v.mipmaps.length;re++)ke(U.__webglFramebuffer[re],A,v,i.COLOR_ATTACHMENT0,te,re);else ke(U.__webglFramebuffer,A,v,i.COLOR_ATTACHMENT0,te,0);m(v)&&M(te),t.unbindTexture()}A.depthBuffer&&O(A)}function G(A){let v=A.textures;for(let U=0,B=v.length;U<B;U++){let k=v[U];if(m(k)){let pe=S(A),_e=n.get(k).__webglTexture;t.bindTexture(pe,_e),M(pe),t.unbindTexture()}}}let V=[],se=[];function ce(A){if(A.samples>0){if(me(A)===!1){let v=A.textures,U=A.width,B=A.height,k=i.COLOR_BUFFER_BIT,pe=A.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,_e=n.get(A),te=v.length>1;if(te)for(let Se=0;Se<v.length;Se++)t.bindFramebuffer(i.FRAMEBUFFER,_e.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Se,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,_e.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Se,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,_e.__webglMultisampledFramebuffer);let re=A.texture.mipmaps;re&&re.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,_e.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,_e.__webglFramebuffer);for(let Se=0;Se<v.length;Se++){if(A.resolveDepthBuffer&&(A.depthBuffer&&(k|=i.DEPTH_BUFFER_BIT),A.stencilBuffer&&A.resolveStencilBuffer&&(k|=i.STENCIL_BUFFER_BIT)),te){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,_e.__webglColorRenderbuffer[Se]);let Ie=n.get(v[Se]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,Ie,0)}i.blitFramebuffer(0,0,U,B,0,0,U,B,k,i.NEAREST),c===!0&&(V.length=0,se.length=0,V.push(i.COLOR_ATTACHMENT0+Se),A.depthBuffer&&A.resolveDepthBuffer===!1&&(V.push(pe),se.push(pe),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,se)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,V))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),te)for(let Se=0;Se<v.length;Se++){t.bindFramebuffer(i.FRAMEBUFFER,_e.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Se,i.RENDERBUFFER,_e.__webglColorRenderbuffer[Se]);let Ie=n.get(v[Se]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,_e.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Se,i.TEXTURE_2D,Ie,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,_e.__webglMultisampledFramebuffer)}else if(A.depthBuffer&&A.resolveDepthBuffer===!1&&c){let v=A.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[v])}}}function fe(A){return Math.min(s.maxSamples,A.samples)}function me(A){let v=n.get(A);return A.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&v.__useRenderToTexture!==!1}function D(A){let v=a.render.frame;h.get(A)!==v&&(h.set(A,v),A.update())}function Me(A,v){let U=A.colorSpace,B=A.format,k=A.type;return A.isCompressedTexture===!0||A.isVideoTexture===!0||U!==oa&&U!==Oi&&(ht.getTransfer(U)===pt?(B!==bn||k!==hn)&&Ze("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):$e("WebGLTextures: Unsupported texture color space:",U)),v}function Ve(A){return typeof HTMLImageElement<"u"&&A instanceof HTMLImageElement?(l.width=A.naturalWidth||A.width,l.height=A.naturalHeight||A.height):typeof VideoFrame<"u"&&A instanceof VideoFrame?(l.width=A.displayWidth,l.height=A.displayHeight):(l.width=A.width,l.height=A.height),l}this.allocateTextureUnit=Y,this.resetTextureUnits=X,this.getTextureUnits=q,this.setTextureUnits=F,this.setTexture2D=ie,this.setTexture2DArray=ne,this.setTexture3D=ge,this.setTextureCube=ue,this.rebindTextures=H,this.setupRenderTarget=Q,this.updateRenderTargetMipmap=G,this.updateMultisampleRenderTarget=ce,this.setupDepthRenderbuffer=O,this.setupFrameBufferTexture=ke,this.useMultisampledRTT=me,this.isReversedDepthBuffer=function(){return t.buffers.depth.getReversed()}}function eM(i,e){function t(n,s=Oi){let r,a=ht.getTransfer(s);if(n===hn)return i.UNSIGNED_BYTE;if(n===tc)return i.UNSIGNED_SHORT_4_4_4_4;if(n===nc)return i.UNSIGNED_SHORT_5_5_5_1;if(n===Mu)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===Su)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===vu)return i.BYTE;if(n===yu)return i.SHORT;if(n===Dr)return i.UNSIGNED_SHORT;if(n===ec)return i.INT;if(n===ni)return i.UNSIGNED_INT;if(n===Vn)return i.FLOAT;if(n===nn)return i.HALF_FLOAT;if(n===bu)return i.ALPHA;if(n===Eu)return i.RGB;if(n===bn)return i.RGBA;if(n===fi)return i.DEPTH_COMPONENT;if(n===_i)return i.DEPTH_STENCIL;if(n===ic)return i.RED;if(n===sc)return i.RED_INTEGER;if(n===ps)return i.RG;if(n===rc)return i.RG_INTEGER;if(n===ac)return i.RGBA_INTEGER;if(n===to||n===no||n===io||n===so)if(a===pt)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===to)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===no)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===io)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===so)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===to)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===no)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===io)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===so)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===oc||n===lc||n===cc||n===hc)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===oc)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===lc)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===cc)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===hc)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===uc||n===dc||n===fc||n===pc||n===mc||n===ro||n===gc)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(n===uc||n===dc)return a===pt?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===fc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC;if(n===pc)return r.COMPRESSED_R11_EAC;if(n===mc)return r.COMPRESSED_SIGNED_R11_EAC;if(n===ro)return r.COMPRESSED_RG11_EAC;if(n===gc)return r.COMPRESSED_SIGNED_RG11_EAC}else return null;if(n===_c||n===xc||n===vc||n===yc||n===Mc||n===Sc||n===bc||n===Ec||n===wc||n===Tc||n===Ac||n===Rc||n===Cc||n===Pc)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(n===_c)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===xc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===vc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===yc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Mc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===Sc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===bc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Ec)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===wc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Tc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===Ac)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===Rc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===Cc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===Pc)return a===pt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===Ic||n===Dc||n===Lc)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(n===Ic)return a===pt?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===Dc)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===Lc)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Uc||n===Nc||n===ao||n===Fc)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(n===Uc)return r.COMPRESSED_RED_RGTC1_EXT;if(n===Nc)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===ao)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===Fc)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===fs?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}var tM=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,nM=`
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

}`,Ju=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){let n=new ya(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){let t=e.cameras[0].viewport,n=new bt({vertexShader:tM,fragmentShader:nM,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new et(new Hn(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},ju=class extends jn{constructor(e,t){super();let n=this,s=null,r=1,a=null,o="local-floor",c=1,l=null,h=null,d=null,u=null,f=null,g=null,_=typeof XRWebGLBinding<"u",p=new Ju,m={},M=t.getContextAttributes(),S=null,y=null,T=[],b=[],P=new Z,x=null,E=new Qt;E.viewport=new mt;let C=new Qt;C.viewport=new mt;let I=[E,C],L=new Yl,X=null,q=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(j){let he=T[j];return he===void 0&&(he=new Mr,T[j]=he),he.getTargetRaySpace()},this.getControllerGrip=function(j){let he=T[j];return he===void 0&&(he=new Mr,T[j]=he),he.getGripSpace()},this.getHand=function(j){let he=T[j];return he===void 0&&(he=new Mr,T[j]=he),he.getHandSpace()};function F(j){let he=b.indexOf(j.inputSource);if(he===-1)return;let le=T[he];le!==void 0&&(le.update(j.inputSource,j.frame,l||a),le.dispatchEvent({type:j.type,data:j.inputSource}))}function Y(){s.removeEventListener("select",F),s.removeEventListener("selectstart",F),s.removeEventListener("selectend",F),s.removeEventListener("squeeze",F),s.removeEventListener("squeezestart",F),s.removeEventListener("squeezeend",F),s.removeEventListener("end",Y),s.removeEventListener("inputsourceschange",W);for(let j=0;j<T.length;j++){let he=b[j];he!==null&&(b[j]=null,T[j].disconnect(he))}X=null,q=null,p.reset();for(let j in m)delete m[j];e.setRenderTarget(S),f=null,u=null,d=null,s=null,y=null,Xe.stop(),n.isPresenting=!1,e.setPixelRatio(x),e.setSize(P.width,P.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(j){r=j,n.isPresenting===!0&&Ze("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(j){o=j,n.isPresenting===!0&&Ze("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||a},this.setReferenceSpace=function(j){l=j},this.getBaseLayer=function(){return u!==null?u:f},this.getBinding=function(){return d===null&&_&&(d=new XRWebGLBinding(s,t)),d},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(j){if(s=j,s!==null){if(S=e.getRenderTarget(),s.addEventListener("select",F),s.addEventListener("selectstart",F),s.addEventListener("selectend",F),s.addEventListener("squeeze",F),s.addEventListener("squeezestart",F),s.addEventListener("squeezeend",F),s.addEventListener("end",Y),s.addEventListener("inputsourceschange",W),M.xrCompatible!==!0&&await t.makeXRCompatible(),x=e.getPixelRatio(),e.getSize(P),_&&"createProjectionLayer"in XRWebGLBinding.prototype){let le=null,Te=null,Fe=null;M.depth&&(Fe=M.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,le=M.stencil?_i:fi,Te=M.stencil?fs:ni);let ke={colorFormat:t.RGBA8,depthFormat:Fe,scaleFactor:r};d=this.getBinding(),u=d.createProjectionLayer(ke),s.updateRenderState({layers:[u]}),e.setPixelRatio(1),e.setSize(u.textureWidth,u.textureHeight,!1),y=new Ht(u.textureWidth,u.textureHeight,{format:bn,type:hn,depthTexture:new Kn(u.textureWidth,u.textureHeight,Te,void 0,void 0,void 0,void 0,void 0,void 0,le),stencilBuffer:M.stencil,colorSpace:e.outputColorSpace,samples:M.antialias?4:0,resolveDepthBuffer:u.ignoreDepthValues===!1,resolveStencilBuffer:u.ignoreDepthValues===!1})}else{let le={antialias:M.antialias,alpha:!0,depth:M.depth,stencil:M.stencil,framebufferScaleFactor:r};f=new XRWebGLLayer(s,t,le),s.updateRenderState({baseLayer:f}),e.setPixelRatio(1),e.setSize(f.framebufferWidth,f.framebufferHeight,!1),y=new Ht(f.framebufferWidth,f.framebufferHeight,{format:bn,type:hn,colorSpace:e.outputColorSpace,stencilBuffer:M.stencil,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}y.isXRRenderTarget=!0,this.setFoveation(c),l=null,a=await s.requestReferenceSpace(o),Xe.setContext(s),Xe.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return p.getDepthTexture()};function W(j){for(let he=0;he<j.removed.length;he++){let le=j.removed[he],Te=b.indexOf(le);Te>=0&&(b[Te]=null,T[Te].disconnect(le))}for(let he=0;he<j.added.length;he++){let le=j.added[he],Te=b.indexOf(le);if(Te===-1){for(let ke=0;ke<T.length;ke++)if(ke>=b.length){b.push(le),Te=ke;break}else if(b[ke]===null){b[ke]=le,Te=ke;break}if(Te===-1)break}let Fe=T[Te];Fe&&Fe.connect(le)}}let ie=new R,ne=new R;function ge(j,he,le){ie.setFromMatrixPosition(he.matrixWorld),ne.setFromMatrixPosition(le.matrixWorld);let Te=ie.distanceTo(ne),Fe=he.projectionMatrix.elements,ke=le.projectionMatrix.elements,oe=Fe[14]/(Fe[10]-1),ee=Fe[14]/(Fe[10]+1),O=(Fe[9]+1)/Fe[5],H=(Fe[9]-1)/Fe[5],Q=(Fe[8]-1)/Fe[0],G=(ke[8]+1)/ke[0],V=oe*Q,se=oe*G,ce=Te/(-Q+G),fe=ce*-Q;if(he.matrixWorld.decompose(j.position,j.quaternion,j.scale),j.translateX(fe),j.translateZ(ce),j.matrixWorld.compose(j.position,j.quaternion,j.scale),j.matrixWorldInverse.copy(j.matrixWorld).invert(),Fe[10]===-1)j.projectionMatrix.copy(he.projectionMatrix),j.projectionMatrixInverse.copy(he.projectionMatrixInverse);else{let me=oe+ce,D=ee+ce,Me=V-fe,Ve=se+(Te-fe),A=O*ee/D*me,v=H*ee/D*me;j.projectionMatrix.makePerspective(Me,Ve,A,v,me,D),j.projectionMatrixInverse.copy(j.projectionMatrix).invert()}}function ue(j,he){he===null?j.matrixWorld.copy(j.matrix):j.matrixWorld.multiplyMatrices(he.matrixWorld,j.matrix),j.matrixWorldInverse.copy(j.matrixWorld).invert()}this.updateCamera=function(j){if(s===null)return;let he=j.near,le=j.far;p.texture!==null&&(p.depthNear>0&&(he=p.depthNear),p.depthFar>0&&(le=p.depthFar)),L.near=C.near=E.near=he,L.far=C.far=E.far=le,(X!==L.near||q!==L.far)&&(s.updateRenderState({depthNear:L.near,depthFar:L.far}),X=L.near,q=L.far),L.layers.mask=j.layers.mask|6,E.layers.mask=L.layers.mask&-5,C.layers.mask=L.layers.mask&-3;let Te=j.parent,Fe=L.cameras;ue(L,Te);for(let ke=0;ke<Fe.length;ke++)ue(Fe[ke],Te);Fe.length===2?ge(L,E,C):L.projectionMatrix.copy(E.projectionMatrix),xe(j,L,Te)};function xe(j,he,le){le===null?j.matrix.copy(he.matrixWorld):(j.matrix.copy(le.matrixWorld),j.matrix.invert(),j.matrix.multiply(he.matrixWorld)),j.matrix.decompose(j.position,j.quaternion,j.scale),j.updateMatrixWorld(!0),j.projectionMatrix.copy(he.projectionMatrix),j.projectionMatrixInverse.copy(he.projectionMatrixInverse),j.isPerspectiveCamera&&(j.fov=xr*2*Math.atan(1/j.projectionMatrix.elements[5]),j.zoom=1)}this.getCamera=function(){return L},this.getFoveation=function(){if(!(u===null&&f===null))return c},this.setFoveation=function(j){c=j,u!==null&&(u.fixedFoveation=j),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=j)},this.hasDepthSensing=function(){return p.texture!==null},this.getDepthSensingMesh=function(){return p.getMesh(L)},this.getCameraTexture=function(j){return m[j]};let Ne=null;function it(j,he){if(h=he.getViewerPose(l||a),g=he,h!==null){let le=h.views;f!==null&&(e.setRenderTargetFramebuffer(y,f.framebuffer),e.setRenderTarget(y));let Te=!1;le.length!==L.cameras.length&&(L.cameras.length=0,Te=!0);for(let ee=0;ee<le.length;ee++){let O=le[ee],H=null;if(f!==null)H=f.getViewport(O);else{let G=d.getViewSubImage(u,O);H=G.viewport,ee===0&&(e.setRenderTargetTextures(y,G.colorTexture,G.depthStencilTexture),e.setRenderTarget(y))}let Q=I[ee];Q===void 0&&(Q=new Qt,Q.layers.enable(ee),Q.viewport=new mt,I[ee]=Q),Q.matrix.fromArray(O.transform.matrix),Q.matrix.decompose(Q.position,Q.quaternion,Q.scale),Q.projectionMatrix.fromArray(O.projectionMatrix),Q.projectionMatrixInverse.copy(Q.projectionMatrix).invert(),Q.viewport.set(H.x,H.y,H.width,H.height),ee===0&&(L.matrix.copy(Q.matrix),L.matrix.decompose(L.position,L.quaternion,L.scale)),Te===!0&&L.cameras.push(Q)}let Fe=s.enabledFeatures;if(Fe&&Fe.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&_){d=n.getBinding();let ee=d.getDepthInformation(le[0]);ee&&ee.isValid&&ee.texture&&p.init(ee,s.renderState)}if(Fe&&Fe.includes("camera-access")&&_){e.state.unbindTexture(),d=n.getBinding();for(let ee=0;ee<le.length;ee++){let O=le[ee].camera;if(O){let H=m[O];H||(H=new ya,m[O]=H);let Q=d.getCameraImage(O);H.sourceTexture=Q}}}}for(let le=0;le<T.length;le++){let Te=b[le],Fe=T[le];Te!==null&&Fe!==void 0&&Fe.update(Te,he,l||a)}Ne&&Ne(j,he),he.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:he}),g=null}let Xe=new Dp;Xe.setAnimationLoop(it),this.setAnimationLoop=function(j){Ne=j},this.dispose=function(){}}},iM=new st,Bp=new Qe;Bp.set(-1,0,0,0,1,0,0,0,1);function sM(i,e){function t(p,m){p.matrixAutoUpdate===!0&&p.updateMatrix(),m.value.copy(p.matrix)}function n(p,m){m.color.getRGB(p.fogColor.value,Ru(i)),m.isFog?(p.fogNear.value=m.near,p.fogFar.value=m.far):m.isFogExp2&&(p.fogDensity.value=m.density)}function s(p,m,M,S,y){m.isNodeMaterial?m.uniformsNeedUpdate=!1:m.isMeshBasicMaterial?r(p,m):m.isMeshLambertMaterial?(r(p,m),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)):m.isMeshToonMaterial?(r(p,m),d(p,m)):m.isMeshPhongMaterial?(r(p,m),h(p,m),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)):m.isMeshStandardMaterial?(r(p,m),u(p,m),m.isMeshPhysicalMaterial&&f(p,m,y)):m.isMeshMatcapMaterial?(r(p,m),g(p,m)):m.isMeshDepthMaterial?r(p,m):m.isMeshDistanceMaterial?(r(p,m),_(p,m)):m.isMeshNormalMaterial?r(p,m):m.isLineBasicMaterial?(a(p,m),m.isLineDashedMaterial&&o(p,m)):m.isPointsMaterial?c(p,m,M,S):m.isSpriteMaterial?l(p,m):m.isShadowMaterial?(p.color.value.copy(m.color),p.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(p,m){p.opacity.value=m.opacity,m.color&&p.diffuse.value.copy(m.color),m.emissive&&p.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(p.map.value=m.map,t(m.map,p.mapTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.bumpMap&&(p.bumpMap.value=m.bumpMap,t(m.bumpMap,p.bumpMapTransform),p.bumpScale.value=m.bumpScale,m.side===tn&&(p.bumpScale.value*=-1)),m.normalMap&&(p.normalMap.value=m.normalMap,t(m.normalMap,p.normalMapTransform),p.normalScale.value.copy(m.normalScale),m.side===tn&&p.normalScale.value.negate()),m.displacementMap&&(p.displacementMap.value=m.displacementMap,t(m.displacementMap,p.displacementMapTransform),p.displacementScale.value=m.displacementScale,p.displacementBias.value=m.displacementBias),m.emissiveMap&&(p.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,p.emissiveMapTransform)),m.specularMap&&(p.specularMap.value=m.specularMap,t(m.specularMap,p.specularMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest);let M=e.get(m),S=M.envMap,y=M.envMapRotation;S&&(p.envMap.value=S,p.envMapRotation.value.setFromMatrix4(iM.makeRotationFromEuler(y)).transpose(),S.isCubeTexture&&S.isRenderTargetTexture===!1&&p.envMapRotation.value.premultiply(Bp),p.reflectivity.value=m.reflectivity,p.ior.value=m.ior,p.refractionRatio.value=m.refractionRatio),m.lightMap&&(p.lightMap.value=m.lightMap,p.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,p.lightMapTransform)),m.aoMap&&(p.aoMap.value=m.aoMap,p.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,p.aoMapTransform))}function a(p,m){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,m.map&&(p.map.value=m.map,t(m.map,p.mapTransform))}function o(p,m){p.dashSize.value=m.dashSize,p.totalSize.value=m.dashSize+m.gapSize,p.scale.value=m.scale}function c(p,m,M,S){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,p.size.value=m.size*M,p.scale.value=S*.5,m.map&&(p.map.value=m.map,t(m.map,p.uvTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest)}function l(p,m){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,p.rotation.value=m.rotation,m.map&&(p.map.value=m.map,t(m.map,p.mapTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest)}function h(p,m){p.specular.value.copy(m.specular),p.shininess.value=Math.max(m.shininess,1e-4)}function d(p,m){m.gradientMap&&(p.gradientMap.value=m.gradientMap)}function u(p,m){p.metalness.value=m.metalness,m.metalnessMap&&(p.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,p.metalnessMapTransform)),p.roughness.value=m.roughness,m.roughnessMap&&(p.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,p.roughnessMapTransform)),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)}function f(p,m,M){p.ior.value=m.ior,m.sheen>0&&(p.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),p.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(p.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,p.sheenColorMapTransform)),m.sheenRoughnessMap&&(p.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,p.sheenRoughnessMapTransform))),m.clearcoat>0&&(p.clearcoat.value=m.clearcoat,p.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(p.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,p.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(p.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,p.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(p.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,p.clearcoatNormalMapTransform),p.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===tn&&p.clearcoatNormalScale.value.negate())),m.dispersion>0&&(p.dispersion.value=m.dispersion),m.iridescence>0&&(p.iridescence.value=m.iridescence,p.iridescenceIOR.value=m.iridescenceIOR,p.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],p.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(p.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,p.iridescenceMapTransform)),m.iridescenceThicknessMap&&(p.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,p.iridescenceThicknessMapTransform))),m.transmission>0&&(p.transmission.value=m.transmission,p.transmissionSamplerMap.value=M.texture,p.transmissionSamplerSize.value.set(M.width,M.height),m.transmissionMap&&(p.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,p.transmissionMapTransform)),p.thickness.value=m.thickness,m.thicknessMap&&(p.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,p.thicknessMapTransform)),p.attenuationDistance.value=m.attenuationDistance,p.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(p.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(p.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,p.anisotropyMapTransform))),p.specularIntensity.value=m.specularIntensity,p.specularColor.value.copy(m.specularColor),m.specularColorMap&&(p.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,p.specularColorMapTransform)),m.specularIntensityMap&&(p.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,p.specularIntensityMapTransform))}function g(p,m){m.matcap&&(p.matcap.value=m.matcap)}function _(p,m){let M=e.get(m).light;p.referencePosition.value.setFromMatrixPosition(M.matrixWorld),p.nearDistance.value=M.shadow.camera.near,p.farDistance.value=M.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function rM(i,e,t,n){let s={},r={},a=[],o=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function c(y,T){let b=T.program;n.uniformBlockBinding(y,b)}function l(y,T){let b=s[y.id];b===void 0&&(p(y),b=h(y),s[y.id]=b,y.addEventListener("dispose",M));let P=T.program;n.updateUBOMapping(y,P);let x=e.render.frame;r[y.id]!==x&&(u(y),r[y.id]=x)}function h(y){let T=d();y.__bindingPointIndex=T;let b=i.createBuffer(),P=y.__size,x=y.usage;return i.bindBuffer(i.UNIFORM_BUFFER,b),i.bufferData(i.UNIFORM_BUFFER,P,x),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,T,b),b}function d(){for(let y=0;y<o;y++)if(a.indexOf(y)===-1)return a.push(y),y;return $e("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function u(y){let T=s[y.id],b=y.uniforms,P=y.__cache;i.bindBuffer(i.UNIFORM_BUFFER,T);for(let x=0,E=b.length;x<E;x++){let C=b[x];if(Array.isArray(C))for(let I=0,L=C.length;I<L;I++)f(C[I],x,I,P);else f(C,x,0,P)}i.bindBuffer(i.UNIFORM_BUFFER,null)}function f(y,T,b,P){if(_(y,T,b,P)===!0){let x=y.__offset,E=y.value;if(Array.isArray(E)){let C=0;for(let I=0;I<E.length;I++){let L=E[I],X=m(L);g(L,y.__data,C),typeof L!="number"&&typeof L!="boolean"&&!L.isMatrix3&&!ArrayBuffer.isView(L)&&(C+=X.storage/Float32Array.BYTES_PER_ELEMENT)}}else g(E,y.__data,0);i.bufferSubData(i.UNIFORM_BUFFER,x,y.__data)}}function g(y,T,b){typeof y=="number"||typeof y=="boolean"?T[0]=y:y.isMatrix3?(T[0]=y.elements[0],T[1]=y.elements[1],T[2]=y.elements[2],T[3]=0,T[4]=y.elements[3],T[5]=y.elements[4],T[6]=y.elements[5],T[7]=0,T[8]=y.elements[6],T[9]=y.elements[7],T[10]=y.elements[8],T[11]=0):ArrayBuffer.isView(y)?T.set(new y.constructor(y.buffer,y.byteOffset,T.length)):y.toArray(T,b)}function _(y,T,b,P){let x=y.value,E=T+"_"+b;if(P[E]===void 0)return typeof x=="number"||typeof x=="boolean"?P[E]=x:ArrayBuffer.isView(x)?P[E]=x.slice():P[E]=x.clone(),!0;{let C=P[E];if(typeof x=="number"||typeof x=="boolean"){if(C!==x)return P[E]=x,!0}else{if(ArrayBuffer.isView(x))return!0;if(C.equals(x)===!1)return C.copy(x),!0}}return!1}function p(y){let T=y.uniforms,b=0,P=16;for(let E=0,C=T.length;E<C;E++){let I=Array.isArray(T[E])?T[E]:[T[E]];for(let L=0,X=I.length;L<X;L++){let q=I[L],F=Array.isArray(q.value)?q.value:[q.value];for(let Y=0,W=F.length;Y<W;Y++){let ie=F[Y],ne=m(ie),ge=b%P,ue=ge%ne.boundary,xe=ge+ue;b+=ue,xe!==0&&P-xe<ne.storage&&(b+=P-xe),q.__data=new Float32Array(ne.storage/Float32Array.BYTES_PER_ELEMENT),q.__offset=b,b+=ne.storage}}}let x=b%P;return x>0&&(b+=P-x),y.__size=b,y.__cache={},this}function m(y){let T={boundary:0,storage:0};return typeof y=="number"||typeof y=="boolean"?(T.boundary=4,T.storage=4):y.isVector2?(T.boundary=8,T.storage=8):y.isVector3||y.isColor?(T.boundary=16,T.storage=12):y.isVector4?(T.boundary=16,T.storage=16):y.isMatrix3?(T.boundary=48,T.storage=48):y.isMatrix4?(T.boundary=64,T.storage=64):y.isTexture?Ze("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(y)?(T.boundary=16,T.storage=y.byteLength):Ze("WebGLRenderer: Unsupported uniform value type.",y),T}function M(y){let T=y.target;T.removeEventListener("dispose",M);let b=a.indexOf(T.__bindingPointIndex);a.splice(b,1),i.deleteBuffer(s[T.id]),delete s[T.id],delete r[T.id]}function S(){for(let y in s)i.deleteBuffer(s[y]);a=[],s={},r={}}return{bind:c,update:l,dispose:S}}var aM=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),vi=null;function oM(){return vi===null&&(vi=new Ui(aM,16,16,ps,nn),vi.name="DFG_LUT",vi.minFilter=en,vi.magFilter=en,vi.wrapS=ui,vi.wrapT=ui,vi.generateMipmaps=!1,vi.needsUpdate=!0),vi}var Vc=class{constructor(e={}){let{canvas:t=Qf(),context:n=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:d=!1,reversedDepthBuffer:u=!1,outputBufferType:f=hn}=e;this.isWebGLRenderer=!0;let g;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");g=n.getContextAttributes().alpha}else g=a;let _=f,p=new Set([ac,rc,sc]),m=new Set([hn,ni,Dr,fs,tc,nc]),M=new Uint32Array(4),S=new Int32Array(4),y=new R,T=null,b=null,P=[],x=[],E=null;this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=ti,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let C=this,I=!1,L=null,X=null,q=null,F=null;this._outputColorSpace=Lt;let Y=0,W=0,ie=null,ne=-1,ge=null,ue=new mt,xe=new mt,Ne=null,it=new Pe(0),Xe=0,j=t.width,he=t.height,le=1,Te=null,Fe=null,ke=new mt(0,0,j,he),oe=new mt(0,0,j,he),ee=!1,O=new br,H=!1,Q=!1,G=new st,V=new R,se=new mt,ce={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},fe=!1;function me(){return ie===null?le:1}let D=n;function Me(w,z){return t.getContext(w,z)}try{let w={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:h,failIfMajorPerformanceCaveat:d};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${"185"}`),t.addEventListener("webglcontextlost",It,!1),t.addEventListener("webglcontextrestored",Mt,!1),t.addEventListener("webglcontextcreationerror",ai,!1),D===null){let z="webgl2";if(D=Me(z,w),D===null)throw Me(z)?new Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes."):new Error("THREE.WebGLRenderer: Error creating WebGL context.")}}catch(w){throw $e("WebGLRenderer: "+w.message),w}let Ve,A,v,U,B,k,pe,_e,te,re,Se,Ie,ve,ye,Be,qe,Je,N,Ee,ae,we,Ce,de;function He(){Ve=new pv(D),Ve.init(),we=new eM(D,Ve),A=new av(D,Ve,e,we),v=new Ky(D,Ve),A.reversedDepthBuffer&&u&&v.buffers.depth.setReversed(!0),X=D.createFramebuffer(),q=D.createFramebuffer(),F=D.createFramebuffer(),U=new _v(D),B=new By,k=new Qy(D,Ve,v,B,A,we,U),pe=new fv(C),_e=new M0(D),Ce=new sv(D,_e),te=new mv(D,_e,U,Ce),re=new vv(D,te,_e,Ce,U),N=new xv(D,A,k),Be=new ov(B),Se=new Oy(C,pe,Ve,A,Ce,Be),Ie=new sM(C,B),ve=new ky,ye=new qy(Ve),Je=new iv(C,pe,v,re,g,c),qe=new jy(C,re,A),de=new rM(D,U,A,v),Ee=new rv(D,Ve,U),ae=new gv(D,Ve,U),U.programs=Se.programs,C.capabilities=A,C.extensions=Ve,C.properties=B,C.renderLists=ve,C.shadowMap=qe,C.state=v,C.info=U}He(),_!==hn&&(E=new Mv(_,t.width,t.height,o,s,r));let Oe=new ju(C,D);this.xr=Oe,this.getContext=function(){return D},this.getContextAttributes=function(){return D.getContextAttributes()},this.forceContextLoss=function(){let w=Ve.get("WEBGL_lose_context");w&&w.loseContext()},this.forceContextRestore=function(){let w=Ve.get("WEBGL_lose_context");w&&w.restoreContext()},this.getPixelRatio=function(){return le},this.setPixelRatio=function(w){w!==void 0&&(le=w,this.setSize(j,he,!1))},this.getSize=function(w){return w.set(j,he)},this.setSize=function(w,z,K=!0){if(Oe.isPresenting){Ze("WebGLRenderer: Can't change size while VR device is presenting.");return}j=w,he=z,t.width=Math.floor(w*le),t.height=Math.floor(z*le),K===!0&&(t.style.width=w+"px",t.style.height=z+"px"),E!==null&&E.setSize(t.width,t.height),this.setViewport(0,0,w,z)},this.getDrawingBufferSize=function(w){return w.set(j*le,he*le).floor()},this.setDrawingBufferSize=function(w,z,K){j=w,he=z,le=K,t.width=Math.floor(w*K),t.height=Math.floor(z*K),this.setViewport(0,0,w,z)},this.setEffects=function(w){if(_===hn){$e("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(w){for(let z=0;z<w.length;z++)if(w[z].isOutputPass===!0){Ze("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}E.setEffects(w||[])},this.getCurrentViewport=function(w){return w.copy(ue)},this.getViewport=function(w){return w.copy(ke)},this.setViewport=function(w,z,K,$){w.isVector4?ke.set(w.x,w.y,w.z,w.w):ke.set(w,z,K,$),v.viewport(ue.copy(ke).multiplyScalar(le).round())},this.getScissor=function(w){return w.copy(oe)},this.setScissor=function(w,z,K,$){w.isVector4?oe.set(w.x,w.y,w.z,w.w):oe.set(w,z,K,$),v.scissor(xe.copy(oe).multiplyScalar(le).round())},this.getScissorTest=function(){return ee},this.setScissorTest=function(w){v.setScissorTest(ee=w)},this.setOpaqueSort=function(w){Te=w},this.setTransparentSort=function(w){Fe=w},this.getClearColor=function(w){return w.copy(Je.getClearColor())},this.setClearColor=function(){Je.setClearColor(...arguments)},this.getClearAlpha=function(){return Je.getClearAlpha()},this.setClearAlpha=function(){Je.setClearAlpha(...arguments)},this.clear=function(w=!0,z=!0,K=!0){let $=0;if(w){let J=!1;if(ie!==null){let Re=ie.texture.format;J=p.has(Re)}if(J){let Re=ie.texture.type,Ue=m.has(Re),Ae=Je.getClearColor(),ze=Je.getClearAlpha(),Ge=Ae.r,nt=Ae.g,lt=Ae.b;Ue?(M[0]=Ge,M[1]=nt,M[2]=lt,M[3]=ze,D.clearBufferuiv(D.COLOR,0,M)):(S[0]=Ge,S[1]=nt,S[2]=lt,S[3]=ze,D.clearBufferiv(D.COLOR,0,S))}else $|=D.COLOR_BUFFER_BIT}z&&($|=D.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),K&&($|=D.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),$!==0&&D.clear($)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(w){w.setRenderer(this),L=w},this.dispose=function(){t.removeEventListener("webglcontextlost",It,!1),t.removeEventListener("webglcontextrestored",Mt,!1),t.removeEventListener("webglcontextcreationerror",ai,!1),Je.dispose(),ve.dispose(),ye.dispose(),B.dispose(),pe.dispose(),re.dispose(),Ce.dispose(),de.dispose(),Se.dispose(),Oe.dispose(),Oe.removeEventListener("sessionstart",Cd),Oe.removeEventListener("sessionend",Pd),Ss.stop()};function It(w){w.preventDefault(),ha("WebGLRenderer: Context Lost."),I=!0}function Mt(){ha("WebGLRenderer: Context Restored."),I=!1;let w=U.autoReset,z=qe.enabled,K=qe.autoUpdate,$=qe.needsUpdate,J=qe.type;He(),U.autoReset=w,qe.enabled=z,qe.autoUpdate=K,qe.needsUpdate=$,qe.type=J}function ai(w){$e("WebGLRenderer: A WebGL context could not be created. Reason: ",w.statusMessage)}function oi(w){let z=w.target;z.removeEventListener("dispose",oi),Gm(z)}function Gm(w){Wm(w),B.remove(w)}function Wm(w){let z=B.get(w).programs;z!==void 0&&(z.forEach(function(K){Se.releaseProgram(K)}),w.isShaderMaterial&&Se.releaseShaderCache(w))}this.renderBufferDirect=function(w,z,K,$,J,Re){z===null&&(z=ce);let Ue=J.isMesh&&J.matrixWorld.determinantAffine()<0,Ae=Ym(w,z,K,$,J);v.setMaterial($,Ue);let ze=K.index,Ge=1;if($.wireframe===!0){if(ze=te.getWireframeAttribute(K),ze===void 0)return;Ge=2}let nt=K.drawRange,lt=K.attributes.position,Ye=nt.start*Ge,_t=(nt.start+nt.count)*Ge;Re!==null&&(Ye=Math.max(Ye,Re.start*Ge),_t=Math.min(_t,(Re.start+Re.count)*Ge)),ze!==null?(Ye=Math.max(Ye,0),_t=Math.min(_t,ze.count)):lt!=null&&(Ye=Math.max(Ye,0),_t=Math.min(_t,lt.count));let Nt=_t-Ye;if(Nt<0||Nt===1/0)return;Ce.setup(J,$,Ae,K,ze);let Dt,vt=Ee;if(ze!==null&&(Dt=_e.get(ze),vt=ae,vt.setIndex(Dt)),J.isMesh)$.wireframe===!0?(v.setLineWidth($.wireframeLinewidth*me()),vt.setMode(D.LINES)):vt.setMode(D.TRIANGLES);else if(J.isLine){let on=$.linewidth;on===void 0&&(on=1),v.setLineWidth(on*me()),J.isLineSegments?vt.setMode(D.LINES):J.isLineLoop?vt.setMode(D.LINE_LOOP):vt.setMode(D.LINE_STRIP)}else J.isPoints?vt.setMode(D.POINTS):J.isSprite&&vt.setMode(D.TRIANGLES);if(J.isBatchedMesh)if(Ve.get("WEBGL_multi_draw"))vt.renderMultiDraw(J._multiDrawStarts,J._multiDrawCounts,J._multiDrawCount);else{let on=J._multiDrawStarts,Le=J._multiDrawCounts,An=J._multiDrawCount,dt=ze?_e.get(ze).bytesPerElement:1,On=B.get($).currentProgram.getUniforms();for(let li=0;li<An;li++)On.setValue(D,"_gl_DrawID",li),vt.render(on[li]/dt,Le[li])}else if(J.isInstancedMesh)vt.renderInstances(Ye,Nt,J.count);else if(K.isInstancedBufferGeometry){let on=K._maxInstanceCount!==void 0?K._maxInstanceCount:1/0,Le=Math.min(K.instanceCount,on);vt.renderInstances(Ye,Nt,Le)}else vt.render(Ye,Nt)};function Rd(w,z,K){w.transparent===!0&&w.side===Sn&&w.forceSinglePass===!1?(w.side=tn,w.needsUpdate=!0,wo(w,z,K),w.side=Jn,w.needsUpdate=!0,wo(w,z,K),w.side=Sn):wo(w,z,K)}this.compile=function(w,z,K=null){K===null&&(K=w),b=ye.get(K),b.init(z),x.push(b),K.traverseVisible(function(J){J.isLight&&J.layers.test(z.layers)&&(b.pushLight(J),J.castShadow&&b.pushShadow(J))}),w!==K&&w.traverseVisible(function(J){J.isLight&&J.layers.test(z.layers)&&(b.pushLight(J),J.castShadow&&b.pushShadow(J))}),b.setupLights();let $=new Set;return w.traverse(function(J){if(!(J.isMesh||J.isPoints||J.isLine||J.isSprite))return;let Re=J.material;if(Re)if(Array.isArray(Re))for(let Ue=0;Ue<Re.length;Ue++){let Ae=Re[Ue];Rd(Ae,K,J),$.add(Ae)}else Rd(Re,K,J),$.add(Re)}),b=x.pop(),$},this.compileAsync=function(w,z,K=null){let $=this.compile(w,z,K);return new Promise(J=>{function Re(){if($.forEach(function(Ue){B.get(Ue).currentProgram.isReady()&&$.delete(Ue)}),$.size===0){J(w);return}setTimeout(Re,10)}Ve.get("KHR_parallel_shader_compile")!==null?Re():setTimeout(Re,10)})};let Sh=null;function Xm(w){Sh&&Sh(w)}function Cd(){Ss.stop()}function Pd(){Ss.start()}let Ss=new Dp;Ss.setAnimationLoop(Xm),typeof self<"u"&&Ss.setContext(self),this.setAnimationLoop=function(w){Sh=w,Oe.setAnimationLoop(w),w===null?Ss.stop():Ss.start()},Oe.addEventListener("sessionstart",Cd),Oe.addEventListener("sessionend",Pd),this.render=function(w,z){if(z!==void 0&&z.isCamera!==!0){$e("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(I===!0)return;L!==null&&L.renderStart(w,z);let K=Oe.enabled===!0&&Oe.isPresenting===!0,$=E!==null&&(ie===null||K)&&E.begin(C,ie);if(w.matrixWorldAutoUpdate===!0&&w.updateMatrixWorld(),z.parent===null&&z.matrixWorldAutoUpdate===!0&&z.updateMatrixWorld(),Oe.enabled===!0&&Oe.isPresenting===!0&&(E===null||E.isCompositing()===!1)&&(Oe.cameraAutoUpdate===!0&&Oe.updateCamera(z),z=Oe.getCamera()),w.isScene===!0&&w.onBeforeRender(C,w,z,ie),b=ye.get(w,x.length),b.init(z),b.state.textureUnits=k.getTextureUnits(),x.push(b),G.multiplyMatrices(z.projectionMatrix,z.matrixWorldInverse),O.setFromProjectionMatrix(G,$n,z.reversedDepth),Q=this.localClippingEnabled,H=Be.init(this.clippingPlanes,Q),T=ve.get(w,P.length),T.init(),P.push(T),Oe.enabled===!0&&Oe.isPresenting===!0){let Ue=C.xr.getDepthSensingMesh();Ue!==null&&bh(Ue,z,-1/0,C.sortObjects)}bh(w,z,0,C.sortObjects),T.finish(),C.sortObjects===!0&&T.sort(Te,Fe,z.reversedDepth),fe=Oe.enabled===!1||Oe.isPresenting===!1||Oe.hasDepthSensing()===!1,fe&&Je.addToRenderList(T,w),this.info.render.frame++,this.info.autoReset===!0&&this.info.reset(),H===!0&&Be.beginShadows();let J=b.state.shadowsArray;if(qe.render(J,w,z),H===!0&&Be.endShadows(),($&&E.hasRenderPass())===!1){let Ue=T.opaque,Ae=T.transmissive;if(b.setupLights(),z.isArrayCamera){let ze=z.cameras;if(Ae.length>0)for(let Ge=0,nt=ze.length;Ge<nt;Ge++){let lt=ze[Ge];Dd(Ue,Ae,w,lt)}fe&&Je.render(w);for(let Ge=0,nt=ze.length;Ge<nt;Ge++){let lt=ze[Ge];Id(T,w,lt,lt.viewport)}}else Ae.length>0&&Dd(Ue,Ae,w,z),fe&&Je.render(w),Id(T,w,z)}ie!==null&&W===0&&(k.updateMultisampleRenderTarget(ie),k.updateRenderTargetMipmap(ie)),$&&E.end(C),w.isScene===!0&&w.onAfterRender(C,w,z),Ce.resetDefaultState(),ne=-1,ge=null,x.pop(),x.length>0?(b=x[x.length-1],k.setTextureUnits(b.state.textureUnits),H===!0&&Be.setGlobalState(C.clippingPlanes,b.state.camera)):b=null,P.pop(),P.length>0?T=P[P.length-1]:T=null,L!==null&&L.renderEnd()};function bh(w,z,K,$){if(w.visible===!1)return;if(w.layers.test(z.layers)){if(w.isGroup)K=w.renderOrder;else if(w.isLOD)w.autoUpdate===!0&&w.update(z);else if(w.isLightProbeGrid)b.pushLightProbeGrid(w);else if(w.isLight)b.pushLight(w),w.castShadow&&b.pushShadow(w);else if(w.isSprite){if(!w.frustumCulled||O.intersectsSprite(w)){$&&se.setFromMatrixPosition(w.matrixWorld).applyMatrix4(G);let Ue=re.update(w),Ae=w.material;Ae.visible&&T.push(w,Ue,Ae,K,se.z,null)}}else if((w.isMesh||w.isLine||w.isPoints)&&(!w.frustumCulled||O.intersectsObject(w))){let Ue=re.update(w),Ae=w.material;if($&&(w.boundingSphere!==void 0?(w.boundingSphere===null&&w.computeBoundingSphere(),se.copy(w.boundingSphere.center)):(Ue.boundingSphere===null&&Ue.computeBoundingSphere(),se.copy(Ue.boundingSphere.center)),se.applyMatrix4(w.matrixWorld).applyMatrix4(G)),Array.isArray(Ae)){let ze=Ue.groups;for(let Ge=0,nt=ze.length;Ge<nt;Ge++){let lt=ze[Ge],Ye=Ae[lt.materialIndex];Ye&&Ye.visible&&T.push(w,Ue,Ye,K,se.z,lt)}}else Ae.visible&&T.push(w,Ue,Ae,K,se.z,null)}}let Re=w.children;for(let Ue=0,Ae=Re.length;Ue<Ae;Ue++)bh(Re[Ue],z,K,$)}function Id(w,z,K,$){let{opaque:J,transmissive:Re,transparent:Ue}=w;b.setupLightsView(K),H===!0&&Be.setGlobalState(C.clippingPlanes,K),$&&v.viewport(ue.copy($)),J.length>0&&Eo(J,z,K),Re.length>0&&Eo(Re,z,K),Ue.length>0&&Eo(Ue,z,K),v.buffers.depth.setTest(!0),v.buffers.depth.setMask(!0),v.buffers.color.setMask(!0),v.setPolygonOffset(!1)}function Dd(w,z,K,$){if((K.isScene===!0?K.overrideMaterial:null)!==null)return;if(b.state.transmissionRenderTarget[$.id]===void 0){let Ye=Ve.has("EXT_color_buffer_half_float")||Ve.has("EXT_color_buffer_float");b.state.transmissionRenderTarget[$.id]=new Ht(1,1,{generateMipmaps:!0,type:Ye?nn:hn,minFilter:ds,samples:Math.max(4,A.samples),stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:ht.workingColorSpace})}let Re=b.state.transmissionRenderTarget[$.id],Ue=$.viewport||ue;Re.setSize(Ue.z*C.transmissionResolutionScale,Ue.w*C.transmissionResolutionScale);let Ae=C.getRenderTarget(),ze=C.getActiveCubeFace(),Ge=C.getActiveMipmapLevel();C.setRenderTarget(Re),C.getClearColor(it),Xe=C.getClearAlpha(),Xe<1&&C.setClearColor(16777215,.5),C.clear(),fe&&Je.render(K);let nt=C.toneMapping;C.toneMapping=ti;let lt=$.viewport;if($.viewport!==void 0&&($.viewport=void 0),b.setupLightsView($),H===!0&&Be.setGlobalState(C.clippingPlanes,$),Eo(w,K,$),k.updateMultisampleRenderTarget(Re),k.updateRenderTargetMipmap(Re),Ve.has("WEBGL_multisampled_render_to_texture")===!1){let Ye=!1;for(let _t=0,Nt=z.length;_t<Nt;_t++){let Dt=z[_t],{object:vt,geometry:on,material:Le,group:An}=Dt;if(Le.side===Sn&&vt.layers.test($.layers)){let dt=Le.side;Le.side=tn,Le.needsUpdate=!0,Ld(vt,K,$,on,Le,An),Le.side=dt,Le.needsUpdate=!0,Ye=!0}}Ye===!0&&(k.updateMultisampleRenderTarget(Re),k.updateRenderTargetMipmap(Re))}C.setRenderTarget(Ae,ze,Ge),C.setClearColor(it,Xe),lt!==void 0&&($.viewport=lt),C.toneMapping=nt}function Eo(w,z,K){let $=z.isScene===!0?z.overrideMaterial:null;for(let J=0,Re=w.length;J<Re;J++){let Ue=w[J],{object:Ae,geometry:ze,group:Ge}=Ue,nt=Ue.material;nt.allowOverride===!0&&$!==null&&(nt=$),Ae.layers.test(K.layers)&&Ld(Ae,z,K,ze,nt,Ge)}}function Ld(w,z,K,$,J,Re){w.onBeforeRender(C,z,K,$,J,Re),w.modelViewMatrix.multiplyMatrices(K.matrixWorldInverse,w.matrixWorld),w.normalMatrix.getNormalMatrix(w.modelViewMatrix),J.onBeforeRender(C,z,K,$,w,Re),J.transparent===!0&&J.side===Sn&&J.forceSinglePass===!1?(J.side=tn,J.needsUpdate=!0,C.renderBufferDirect(K,z,$,J,w,Re),J.side=Jn,J.needsUpdate=!0,C.renderBufferDirect(K,z,$,J,w,Re),J.side=Sn):C.renderBufferDirect(K,z,$,J,w,Re),w.onAfterRender(C,z,K,$,J,Re)}function wo(w,z,K){z.isScene!==!0&&(z=ce);let $=B.get(w),J=b.state.lights,Re=b.state.shadowsArray,Ue=J.state.version,Ae=Se.getParameters(w,J.state,Re,z,K,b.state.lightProbeGridArray),ze=Se.getProgramCacheKey(Ae),Ge=$.programs;$.environment=w.isMeshStandardMaterial||w.isMeshLambertMaterial||w.isMeshPhongMaterial?z.environment:null,$.fog=z.fog;let nt=w.isMeshStandardMaterial||w.isMeshLambertMaterial&&!w.envMap||w.isMeshPhongMaterial&&!w.envMap;$.envMap=pe.get(w.envMap||$.environment,nt),$.envMapRotation=$.environment!==null&&w.envMap===null?z.environmentRotation:w.envMapRotation,Ge===void 0&&(w.addEventListener("dispose",oi),Ge=new Map,$.programs=Ge);let lt=Ge.get(ze);if(lt!==void 0){if($.currentProgram===lt&&$.lightsStateVersion===Ue)return Nd(w,Ae),lt}else Ae.uniforms=Se.getUniforms(w),L!==null&&w.isNodeMaterial&&L.build(w,K,Ae),w.onBeforeCompile(Ae,C),lt=Se.acquireProgram(Ae,ze),Ge.set(ze,lt),$.uniforms=Ae.uniforms;let Ye=$.uniforms;return(!w.isShaderMaterial&&!w.isRawShaderMaterial||w.clipping===!0)&&(Ye.clippingPlanes=Be.uniform),Nd(w,Ae),$.needsLights=$m(w),$.lightsStateVersion=Ue,$.needsLights&&(Ye.ambientLightColor.value=J.state.ambient,Ye.lightProbe.value=J.state.probe,Ye.directionalLights.value=J.state.directional,Ye.directionalLightShadows.value=J.state.directionalShadow,Ye.spotLights.value=J.state.spot,Ye.spotLightShadows.value=J.state.spotShadow,Ye.rectAreaLights.value=J.state.rectArea,Ye.ltc_1.value=J.state.rectAreaLTC1,Ye.ltc_2.value=J.state.rectAreaLTC2,Ye.pointLights.value=J.state.point,Ye.pointLightShadows.value=J.state.pointShadow,Ye.hemisphereLights.value=J.state.hemi,Ye.directionalShadowMatrix.value=J.state.directionalShadowMatrix,Ye.spotLightMatrix.value=J.state.spotLightMatrix,Ye.spotLightMap.value=J.state.spotLightMap,Ye.pointShadowMatrix.value=J.state.pointShadowMatrix),$.lightProbeGrid=b.state.lightProbeGridArray.length>0,$.currentProgram=lt,$.uniformsList=null,lt}function Ud(w){if(w.uniformsList===null){let z=w.currentProgram.getUniforms();w.uniformsList=Nr.seqWithValue(z.seq,w.uniforms)}return w.uniformsList}function Nd(w,z){let K=B.get(w);K.outputColorSpace=z.outputColorSpace,K.batching=z.batching,K.batchingColor=z.batchingColor,K.instancing=z.instancing,K.instancingColor=z.instancingColor,K.instancingMorph=z.instancingMorph,K.skinning=z.skinning,K.morphTargets=z.morphTargets,K.morphNormals=z.morphNormals,K.morphColors=z.morphColors,K.morphTargetsCount=z.morphTargetsCount,K.numClippingPlanes=z.numClippingPlanes,K.numIntersection=z.numClipIntersection,K.vertexAlphas=z.vertexAlphas,K.vertexTangents=z.vertexTangents,K.toneMapping=z.toneMapping}function qm(w,z){if(w.length===0)return null;if(w.length===1)return w[0].texture!==null?w[0]:null;y.setFromMatrixPosition(z.matrixWorld);for(let K=0,$=w.length;K<$;K++){let J=w[K];if(J.texture!==null&&J.boundingBox.containsPoint(y))return J}return null}function Ym(w,z,K,$,J){z.isScene!==!0&&(z=ce),k.resetTextureUnits();let Re=z.fog,Ue=$.isMeshStandardMaterial||$.isMeshLambertMaterial||$.isMeshPhongMaterial?z.environment:null,Ae=ie===null?C.outputColorSpace:ie.isXRRenderTarget===!0?ie.texture.colorSpace:ht.workingColorSpace,ze=$.isMeshStandardMaterial||$.isMeshLambertMaterial&&!$.envMap||$.isMeshPhongMaterial&&!$.envMap,Ge=pe.get($.envMap||Ue,ze),nt=$.vertexColors===!0&&!!K.attributes.color&&K.attributes.color.itemSize===4,lt=!!K.attributes.tangent&&(!!$.normalMap||$.anisotropy>0),Ye=!!K.morphAttributes.position,_t=!!K.morphAttributes.normal,Nt=!!K.morphAttributes.color,Dt=ti;$.toneMapped&&(ie===null||ie.isXRRenderTarget===!0)&&(Dt=C.toneMapping);let vt=K.morphAttributes.position||K.morphAttributes.normal||K.morphAttributes.color,on=vt!==void 0?vt.length:0,Le=B.get($),An=b.state.lights;if(H===!0&&(Q===!0||w!==ge)){let St=w===ge&&$.id===ne;Be.setState($,w,St)}let dt=!1;$.version===Le.__version?(Le.needsLights&&Le.lightsStateVersion!==An.state.version||Le.outputColorSpace!==Ae||J.isBatchedMesh&&Le.batching===!1||!J.isBatchedMesh&&Le.batching===!0||J.isBatchedMesh&&Le.batchingColor===!0&&J.colorTexture===null||J.isBatchedMesh&&Le.batchingColor===!1&&J.colorTexture!==null||J.isInstancedMesh&&Le.instancing===!1||!J.isInstancedMesh&&Le.instancing===!0||J.isSkinnedMesh&&Le.skinning===!1||!J.isSkinnedMesh&&Le.skinning===!0||J.isInstancedMesh&&Le.instancingColor===!0&&J.instanceColor===null||J.isInstancedMesh&&Le.instancingColor===!1&&J.instanceColor!==null||J.isInstancedMesh&&Le.instancingMorph===!0&&J.morphTexture===null||J.isInstancedMesh&&Le.instancingMorph===!1&&J.morphTexture!==null||Le.envMap!==Ge||$.fog===!0&&Le.fog!==Re||Le.numClippingPlanes!==void 0&&(Le.numClippingPlanes!==Be.numPlanes||Le.numIntersection!==Be.numIntersection)||Le.vertexAlphas!==nt||Le.vertexTangents!==lt||Le.morphTargets!==Ye||Le.morphNormals!==_t||Le.morphColors!==Nt||Le.toneMapping!==Dt||Le.morphTargetsCount!==on||!!Le.lightProbeGrid!=b.state.lightProbeGridArray.length>0)&&(dt=!0):(dt=!0,Le.__version=$.version);let On=Le.currentProgram;dt===!0&&(On=wo($,z,J),L&&$.isNodeMaterial&&L.onUpdateProgram($,On,Le));let li=!1,Yi=!1,Ys=!1,yt=On.getUniforms(),Ft=Le.uniforms;if(v.useProgram(On.program)&&(li=!0,Yi=!0,Ys=!0),$.id!==ne&&(ne=$.id,Yi=!0),Le.needsLights){let St=qm(b.state.lightProbeGridArray,J);Le.lightProbeGrid!==St&&(Le.lightProbeGrid=St,Yi=!0)}if(li||ge!==w){v.buffers.depth.getReversed()&&w.reversedDepth!==!0&&(w._reversedDepth=!0,w.updateProjectionMatrix()),yt.setValue(D,"projectionMatrix",w.projectionMatrix),yt.setValue(D,"viewMatrix",w.matrixWorldInverse);let $i=yt.map.cameraPosition;$i!==void 0&&$i.setValue(D,V.setFromMatrixPosition(w.matrixWorld)),A.logarithmicDepthBuffer&&yt.setValue(D,"logDepthBufFC",2/(Math.log(w.far+1)/Math.LN2)),($.isMeshPhongMaterial||$.isMeshToonMaterial||$.isMeshLambertMaterial||$.isMeshBasicMaterial||$.isMeshStandardMaterial||$.isShaderMaterial)&&yt.setValue(D,"isOrthographic",w.isOrthographicCamera===!0),ge!==w&&(ge=w,Yi=!0,Ys=!0)}if(Le.needsLights&&(An.state.directionalShadowMap.length>0&&yt.setValue(D,"directionalShadowMap",An.state.directionalShadowMap,k),An.state.spotShadowMap.length>0&&yt.setValue(D,"spotShadowMap",An.state.spotShadowMap,k),An.state.pointShadowMap.length>0&&yt.setValue(D,"pointShadowMap",An.state.pointShadowMap,k)),J.isSkinnedMesh){yt.setOptional(D,J,"bindMatrix"),yt.setOptional(D,J,"bindMatrixInverse");let St=J.skeleton;St&&(St.boneTexture===null&&St.computeBoneTexture(),yt.setValue(D,"boneTexture",St.boneTexture,k))}J.isBatchedMesh&&(yt.setOptional(D,J,"batchingTexture"),yt.setValue(D,"batchingTexture",J._matricesTexture,k),yt.setOptional(D,J,"batchingIdTexture"),yt.setValue(D,"batchingIdTexture",J._indirectTexture,k),yt.setOptional(D,J,"batchingColorTexture"),J._colorsTexture!==null&&yt.setValue(D,"batchingColorTexture",J._colorsTexture,k));let Zi=K.morphAttributes;if((Zi.position!==void 0||Zi.normal!==void 0||Zi.color!==void 0)&&N.update(J,K,On),(Yi||Le.receiveShadow!==J.receiveShadow)&&(Le.receiveShadow=J.receiveShadow,yt.setValue(D,"receiveShadow",J.receiveShadow)),($.isMeshStandardMaterial||$.isMeshLambertMaterial||$.isMeshPhongMaterial)&&$.envMap===null&&z.environment!==null&&(Ft.envMapIntensity.value=z.environmentIntensity),Ft.dfgLUT!==void 0&&(Ft.dfgLUT.value=oM()),Yi){if(yt.setValue(D,"toneMappingExposure",C.toneMappingExposure),Le.needsLights&&Zm(Ft,Ys),Re&&$.fog===!0&&Ie.refreshFogUniforms(Ft,Re),Ie.refreshMaterialUniforms(Ft,$,le,he,b.state.transmissionRenderTarget[w.id]),Le.needsLights&&Le.lightProbeGrid){let St=Le.lightProbeGrid;Ft.probesSH.value=St.texture,Ft.probesMin.value.copy(St.boundingBox.min),Ft.probesMax.value.copy(St.boundingBox.max),Ft.probesResolution.value.copy(St.resolution)}Nr.upload(D,Ud(Le),Ft,k)}if($.isShaderMaterial&&$.uniformsNeedUpdate===!0&&(Nr.upload(D,Ud(Le),Ft,k),$.uniformsNeedUpdate=!1),$.isSpriteMaterial&&yt.setValue(D,"center",J.center),yt.setValue(D,"modelViewMatrix",J.modelViewMatrix),yt.setValue(D,"normalMatrix",J.normalMatrix),yt.setValue(D,"modelMatrix",J.matrixWorld),$.uniformsGroups!==void 0){let St=$.uniformsGroups;for(let $i=0,Zs=St.length;$i<Zs;$i++){let Fd=St[$i];de.update(Fd,On),de.bind(Fd,On)}}return On}function Zm(w,z){w.ambientLightColor.needsUpdate=z,w.lightProbe.needsUpdate=z,w.directionalLights.needsUpdate=z,w.directionalLightShadows.needsUpdate=z,w.pointLights.needsUpdate=z,w.pointLightShadows.needsUpdate=z,w.spotLights.needsUpdate=z,w.spotLightShadows.needsUpdate=z,w.rectAreaLights.needsUpdate=z,w.hemisphereLights.needsUpdate=z}function $m(w){return w.isMeshLambertMaterial||w.isMeshToonMaterial||w.isMeshPhongMaterial||w.isMeshStandardMaterial||w.isShadowMaterial||w.isShaderMaterial&&w.lights===!0}this.getActiveCubeFace=function(){return Y},this.getActiveMipmapLevel=function(){return W},this.getRenderTarget=function(){return ie},this.setRenderTargetTextures=function(w,z,K){let $=B.get(w);$.__autoAllocateDepthBuffer=w.resolveDepthBuffer===!1,$.__autoAllocateDepthBuffer===!1&&($.__useRenderToTexture=!1),B.get(w.texture).__webglTexture=z,B.get(w.depthTexture).__webglTexture=$.__autoAllocateDepthBuffer?void 0:K,$.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(w,z){let K=B.get(w);K.__webglFramebuffer=z,K.__useDefaultFramebuffer=z===void 0},this.setRenderTarget=function(w,z=0,K=0){ie=w,Y=z,W=K;let $=null,J=!1,Re=!1;if(w){let Ae=B.get(w);if(Ae.__useDefaultFramebuffer!==void 0){v.bindFramebuffer(D.FRAMEBUFFER,Ae.__webglFramebuffer),ue.copy(w.viewport),xe.copy(w.scissor),Ne=w.scissorTest,v.viewport(ue),v.scissor(xe),v.setScissorTest(Ne),ne=-1;return}else if(Ae.__webglFramebuffer===void 0)k.setupRenderTarget(w);else if(Ae.__hasExternalTextures)k.rebindTextures(w,B.get(w.texture).__webglTexture,B.get(w.depthTexture).__webglTexture);else if(w.depthBuffer){let nt=w.depthTexture;if(Ae.__boundDepthTexture!==nt){if(nt!==null&&B.has(nt)&&(w.width!==nt.image.width||w.height!==nt.image.height))throw new Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");k.setupDepthRenderbuffer(w)}}let ze=w.texture;(ze.isData3DTexture||ze.isDataArrayTexture||ze.isCompressedArrayTexture)&&(Re=!0);let Ge=B.get(w).__webglFramebuffer;w.isWebGLCubeRenderTarget?(Array.isArray(Ge[z])?$=Ge[z][K]:$=Ge[z],J=!0):w.samples>0&&k.useMultisampledRTT(w)===!1?$=B.get(w).__webglMultisampledFramebuffer:Array.isArray(Ge)?$=Ge[K]:$=Ge,ue.copy(w.viewport),xe.copy(w.scissor),Ne=w.scissorTest}else ue.copy(ke).multiplyScalar(le).floor(),xe.copy(oe).multiplyScalar(le).floor(),Ne=ee;if(K!==0&&($=X),v.bindFramebuffer(D.FRAMEBUFFER,$)&&v.drawBuffers(w,$),v.viewport(ue),v.scissor(xe),v.setScissorTest(Ne),J){let Ae=B.get(w.texture);D.framebufferTexture2D(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_CUBE_MAP_POSITIVE_X+z,Ae.__webglTexture,K)}else if(Re){let Ae=z;for(let ze=0;ze<w.textures.length;ze++){let Ge=B.get(w.textures[ze]);D.framebufferTextureLayer(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0+ze,Ge.__webglTexture,K,Ae)}}else if(w!==null&&K!==0){let Ae=B.get(w.texture);D.framebufferTexture2D(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Ae.__webglTexture,K)}ne=-1},this.readRenderTargetPixels=function(w,z,K,$,J,Re,Ue,Ae=0){if(!(w&&w.isWebGLRenderTarget)){$e("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ze=B.get(w).__webglFramebuffer;if(w.isWebGLCubeRenderTarget&&Ue!==void 0&&(ze=ze[Ue]),ze){v.bindFramebuffer(D.FRAMEBUFFER,ze);try{let Ge=w.textures[Ae],nt=Ge.format,lt=Ge.type;if(w.textures.length>1&&D.readBuffer(D.COLOR_ATTACHMENT0+Ae),!A.textureFormatReadable(nt)){$e("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!A.textureTypeReadable(lt)){$e("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}z>=0&&z<=w.width-$&&K>=0&&K<=w.height-J&&D.readPixels(z,K,$,J,we.convert(nt),we.convert(lt),Re)}finally{let Ge=ie!==null?B.get(ie).__webglFramebuffer:null;v.bindFramebuffer(D.FRAMEBUFFER,Ge)}}},this.readRenderTargetPixelsAsync=async function(w,z,K,$,J,Re,Ue,Ae=0){if(!(w&&w.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ze=B.get(w).__webglFramebuffer;if(w.isWebGLCubeRenderTarget&&Ue!==void 0&&(ze=ze[Ue]),ze)if(z>=0&&z<=w.width-$&&K>=0&&K<=w.height-J){v.bindFramebuffer(D.FRAMEBUFFER,ze);let Ge=w.textures[Ae],nt=Ge.format,lt=Ge.type;if(w.textures.length>1&&D.readBuffer(D.COLOR_ATTACHMENT0+Ae),!A.textureFormatReadable(nt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!A.textureTypeReadable(lt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");let Ye=D.createBuffer();D.bindBuffer(D.PIXEL_PACK_BUFFER,Ye),D.bufferData(D.PIXEL_PACK_BUFFER,Re.byteLength,D.STREAM_READ),D.readPixels(z,K,$,J,we.convert(nt),we.convert(lt),0);let _t=ie!==null?B.get(ie).__webglFramebuffer:null;v.bindFramebuffer(D.FRAMEBUFFER,_t);let Nt=D.fenceSync(D.SYNC_GPU_COMMANDS_COMPLETE,0);return D.flush(),await tp(D,Nt,4),D.bindBuffer(D.PIXEL_PACK_BUFFER,Ye),D.getBufferSubData(D.PIXEL_PACK_BUFFER,0,Re),D.deleteBuffer(Ye),D.deleteSync(Nt),Re}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(w,z=null,K=0){let $=Math.pow(2,-K),J=Math.floor(w.image.width*$),Re=Math.floor(w.image.height*$),Ue=z!==null?z.x:0,Ae=z!==null?z.y:0;k.setTexture2D(w,0),D.copyTexSubImage2D(D.TEXTURE_2D,K,0,0,Ue,Ae,J,Re),v.unbindTexture()},this.copyTextureToTexture=function(w,z,K=null,$=null,J=0,Re=0){let Ue,Ae,ze,Ge,nt,lt,Ye,_t,Nt,Dt=w.isCompressedTexture?w.mipmaps[Re]:w.image;if(K!==null)Ue=K.max.x-K.min.x,Ae=K.max.y-K.min.y,ze=K.isBox3?K.max.z-K.min.z:1,Ge=K.min.x,nt=K.min.y,lt=K.isBox3?K.min.z:0;else{let Ft=Math.pow(2,-J);Ue=Math.floor(Dt.width*Ft),Ae=Math.floor(Dt.height*Ft),w.isDataArrayTexture?ze=Dt.depth:w.isData3DTexture?ze=Math.floor(Dt.depth*Ft):ze=1,Ge=0,nt=0,lt=0}$!==null?(Ye=$.x,_t=$.y,Nt=$.z):(Ye=0,_t=0,Nt=0);let vt=we.convert(z.format),on=we.convert(z.type),Le;z.isData3DTexture?(k.setTexture3D(z,0),Le=D.TEXTURE_3D):z.isDataArrayTexture||z.isCompressedArrayTexture?(k.setTexture2DArray(z,0),Le=D.TEXTURE_2D_ARRAY):(k.setTexture2D(z,0),Le=D.TEXTURE_2D),v.activeTexture(D.TEXTURE0),v.pixelStorei(D.UNPACK_FLIP_Y_WEBGL,z.flipY),v.pixelStorei(D.UNPACK_PREMULTIPLY_ALPHA_WEBGL,z.premultiplyAlpha),v.pixelStorei(D.UNPACK_ALIGNMENT,z.unpackAlignment);let An=v.getParameter(D.UNPACK_ROW_LENGTH),dt=v.getParameter(D.UNPACK_IMAGE_HEIGHT),On=v.getParameter(D.UNPACK_SKIP_PIXELS),li=v.getParameter(D.UNPACK_SKIP_ROWS),Yi=v.getParameter(D.UNPACK_SKIP_IMAGES);v.pixelStorei(D.UNPACK_ROW_LENGTH,Dt.width),v.pixelStorei(D.UNPACK_IMAGE_HEIGHT,Dt.height),v.pixelStorei(D.UNPACK_SKIP_PIXELS,Ge),v.pixelStorei(D.UNPACK_SKIP_ROWS,nt),v.pixelStorei(D.UNPACK_SKIP_IMAGES,lt);let Ys=w.isDataArrayTexture||w.isData3DTexture,yt=z.isDataArrayTexture||z.isData3DTexture;if(w.isDepthTexture){let Ft=B.get(w),Zi=B.get(z),St=B.get(Ft.__renderTarget),$i=B.get(Zi.__renderTarget);v.bindFramebuffer(D.READ_FRAMEBUFFER,St.__webglFramebuffer),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,$i.__webglFramebuffer);for(let Zs=0;Zs<ze;Zs++)Ys&&(D.framebufferTextureLayer(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,B.get(w).__webglTexture,J,lt+Zs),D.framebufferTextureLayer(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,B.get(z).__webglTexture,Re,Nt+Zs)),D.blitFramebuffer(Ge,nt,Ue,Ae,Ye,_t,Ue,Ae,D.DEPTH_BUFFER_BIT,D.NEAREST);v.bindFramebuffer(D.READ_FRAMEBUFFER,null),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,null)}else if(J!==0||w.isRenderTargetTexture||B.has(w)){let Ft=B.get(w),Zi=B.get(z);v.bindFramebuffer(D.READ_FRAMEBUFFER,q),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,F);for(let St=0;St<ze;St++)Ys?D.framebufferTextureLayer(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,Ft.__webglTexture,J,lt+St):D.framebufferTexture2D(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Ft.__webglTexture,J),yt?D.framebufferTextureLayer(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,Zi.__webglTexture,Re,Nt+St):D.framebufferTexture2D(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Zi.__webglTexture,Re),J!==0?D.blitFramebuffer(Ge,nt,Ue,Ae,Ye,_t,Ue,Ae,D.COLOR_BUFFER_BIT,D.NEAREST):yt?D.copyTexSubImage3D(Le,Re,Ye,_t,Nt+St,Ge,nt,Ue,Ae):D.copyTexSubImage2D(Le,Re,Ye,_t,Ge,nt,Ue,Ae);v.bindFramebuffer(D.READ_FRAMEBUFFER,null),v.bindFramebuffer(D.DRAW_FRAMEBUFFER,null)}else yt?w.isDataTexture||w.isData3DTexture?D.texSubImage3D(Le,Re,Ye,_t,Nt,Ue,Ae,ze,vt,on,Dt.data):z.isCompressedArrayTexture?D.compressedTexSubImage3D(Le,Re,Ye,_t,Nt,Ue,Ae,ze,vt,Dt.data):D.texSubImage3D(Le,Re,Ye,_t,Nt,Ue,Ae,ze,vt,on,Dt):w.isDataTexture?D.texSubImage2D(D.TEXTURE_2D,Re,Ye,_t,Ue,Ae,vt,on,Dt.data):w.isCompressedTexture?D.compressedTexSubImage2D(D.TEXTURE_2D,Re,Ye,_t,Dt.width,Dt.height,vt,Dt.data):D.texSubImage2D(D.TEXTURE_2D,Re,Ye,_t,Ue,Ae,vt,on,Dt);v.pixelStorei(D.UNPACK_ROW_LENGTH,An),v.pixelStorei(D.UNPACK_IMAGE_HEIGHT,dt),v.pixelStorei(D.UNPACK_SKIP_PIXELS,On),v.pixelStorei(D.UNPACK_SKIP_ROWS,li),v.pixelStorei(D.UNPACK_SKIP_IMAGES,Yi),Re===0&&z.generateMipmaps&&D.generateMipmap(Le),v.unbindTexture()},this.initRenderTarget=function(w){B.get(w).__webglFramebuffer===void 0&&k.setupRenderTarget(w)},this.initTexture=function(w){w.isCubeTexture?k.setTextureCube(w,0):w.isData3DTexture?k.setTexture3D(w,0):w.isDataArrayTexture||w.isCompressedArrayTexture?k.setTexture2DArray(w,0):k.setTexture2D(w,0),v.unbindTexture()},this.resetState=function(){Y=0,W=0,ie=null,v.reset(),Ce.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return $n}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;let t=this.getContext();t.drawingBufferColorSpace=ht._getDrawingBufferColorSpace(e),t.unpackColorSpace=ht._getUnpackColorSpace()}};var zp={type:"change"},Qu={type:"start"},Hp={type:"end"},Xc=new Di,kp=new zn,lM=Math.cos(70*Vt.DEG2RAD),qt=new R,En=2*Math.PI,xt={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},Ku=1e-6,qc=class extends Xa{constructor(e,t=null){super(e,t),this.state=xt.NONE,this.target=new R,this.cursor=new R,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:ls.ROTATE,MIDDLE:ls.DOLLY,RIGHT:ls.PAN},this.touches={ONE:cs.ROTATE,TWO:cs.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._cursorStyle="auto",this._domElementKeyEvents=null,this._lastPosition=new R,this._lastQuaternion=new In,this._lastTargetPosition=new R,this._quat=new In().setFromUnitVectors(e.up,new R(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new Pr,this._sphericalDelta=new Pr,this._scale=1,this._panOffset=new R,this._rotateStart=new Z,this._rotateEnd=new Z,this._rotateDelta=new Z,this._panStart=new Z,this._panEnd=new Z,this._panDelta=new Z,this._dollyStart=new Z,this._dollyEnd=new Z,this._dollyDelta=new Z,this._dollyDirection=new R,this._mouse=new Z,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=hM.bind(this),this._onPointerDown=cM.bind(this),this._onPointerUp=uM.bind(this),this._onContextMenu=xM.bind(this),this._onMouseWheel=pM.bind(this),this._onKeyDown=mM.bind(this),this._onTouchStart=gM.bind(this),this._onTouchMove=_M.bind(this),this._onMouseDown=dM.bind(this),this._onMouseMove=fM.bind(this),this._interceptControlDown=vM.bind(this),this._interceptControlUp=yM.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}set cursorStyle(e){this._cursorStyle=e,e==="grab"?this.domElement.style.cursor="grab":this.domElement.style.cursor="auto"}get cursorStyle(){return this._cursorStyle}connect(e){super.connect(e),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction=""}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(e){e.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=e}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(zp),this.update(),this.state=xt.NONE}pan(e,t){this._pan(e,t),this.update()}dollyIn(e){this._dollyIn(e),this.update()}dollyOut(e){this._dollyOut(e),this.update()}rotateLeft(e){this._rotateLeft(e),this.update()}rotateUp(e){this._rotateUp(e),this.update()}update(e=null){let t=this.object.position;qt.copy(t).sub(this.target),qt.applyQuaternion(this._quat),this._spherical.setFromVector3(qt),this.autoRotate&&this.state===xt.NONE&&this._rotateLeft(this._getAutoRotationAngle(e)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let n=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(n)&&isFinite(s)&&(n<-Math.PI?n+=En:n>Math.PI&&(n-=En),s<-Math.PI?s+=En:s>Math.PI&&(s-=En),n<=s?this._spherical.theta=Math.max(n,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(n+s)/2?Math.max(n,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{let a=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=a!=this._spherical.radius}if(qt.setFromSpherical(this._spherical),qt.applyQuaternion(this._quatInverse),t.copy(this.target).add(qt),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let a=null;if(this.object.isPerspectiveCamera){let o=qt.length();a=this._clampDistance(o*this._scale);let c=o-a;this.object.position.addScaledVector(this._dollyDirection,c),this.object.updateMatrixWorld(),r=!!c}else if(this.object.isOrthographicCamera){let o=new R(this._mouse.x,this._mouse.y,0);o.unproject(this.object);let c=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=c!==this.object.zoom;let l=new R(this._mouse.x,this._mouse.y,0);l.unproject(this.object),this.object.position.sub(l).add(o),this.object.updateMatrixWorld(),a=qt.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;a!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(a).add(this.object.position):(Xc.origin.copy(this.object.position),Xc.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Xc.direction))<lM?this.object.lookAt(this.target):(kp.setFromNormalAndCoplanarPoint(this.object.up,this.target),Xc.intersectPlane(kp,this.target))))}else if(this.object.isOrthographicCamera){let a=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),a!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>Ku||8*(1-this._lastQuaternion.dot(this.object.quaternion))>Ku||this._lastTargetPosition.distanceToSquared(this.target)>Ku?(this.dispatchEvent(zp),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(e){return e!==null?En/60*this.autoRotateSpeed*e:En/60/60*this.autoRotateSpeed}_getZoomScale(e){let t=Math.abs(e*.01);return Math.pow(.95,this.zoomSpeed*t)}_rotateLeft(e){this._sphericalDelta.theta-=e}_rotateUp(e){this._sphericalDelta.phi-=e}_panLeft(e,t){qt.setFromMatrixColumn(t,0),qt.multiplyScalar(-e),this._panOffset.add(qt)}_panUp(e,t){this.screenSpacePanning===!0?qt.setFromMatrixColumn(t,1):(qt.setFromMatrixColumn(t,0),qt.crossVectors(this.object.up,qt)),qt.multiplyScalar(e),this._panOffset.add(qt)}_pan(e,t){let n=this.domElement;if(this.object.isPerspectiveCamera){let s=this.object.position;qt.copy(s).sub(this.target);let r=qt.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*e*r/n.clientHeight,this.object.matrix),this._panUp(2*t*r/n.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(e*(this.object.right-this.object.left)/this.object.zoom/n.clientWidth,this.object.matrix),this._panUp(t*(this.object.top-this.object.bottom)/this.object.zoom/n.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(e,t){if(!this.zoomToCursor)return;this._performCursorZoom=!0;let n=this.domElement.getBoundingClientRect(),s=e-n.left,r=t-n.top,a=n.width,o=n.height;this._mouse.x=s/a*2-1,this._mouse.y=-(r/o)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(e){return Math.max(this.minDistance,Math.min(this.maxDistance,e))}_handleMouseDownRotate(e){this._rotateStart.set(e.clientX,e.clientY)}_handleMouseDownDolly(e){this._updateZoomParameters(e.clientX,e.clientX),this._dollyStart.set(e.clientX,e.clientY)}_handleMouseDownPan(e){this._panStart.set(e.clientX,e.clientY)}_handleMouseMoveRotate(e){this._rotateEnd.set(e.clientX,e.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let t=this.domElement;this._rotateLeft(En*this._rotateDelta.x/t.clientHeight),this._rotateUp(En*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(e){this._dollyEnd.set(e.clientX,e.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(e){this._panEnd.set(e.clientX,e.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(e){this._updateZoomParameters(e.clientX,e.clientY),e.deltaY<0?this._dollyIn(this._getZoomScale(e.deltaY)):e.deltaY>0&&this._dollyOut(this._getZoomScale(e.deltaY)),this.update()}_handleKeyDown(e){let t=!1;switch(e.code){case this.keys.UP:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(En*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),t=!0;break;case this.keys.BOTTOM:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(-En*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),t=!0;break;case this.keys.LEFT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(En*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),t=!0;break;case this.keys.RIGHT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(-En*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),this.update())}_handleTouchStartRotate(e){if(this._pointers.length===1)this._rotateStart.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._rotateStart.set(n,s)}}_handleTouchStartPan(e){if(this._pointers.length===1)this._panStart.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panStart.set(n,s)}}_handleTouchStartDolly(e){let t=this._getSecondPointerPosition(e),n=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(n*n+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enablePan&&this._handleTouchStartPan(e)}_handleTouchStartDollyRotate(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enableRotate&&this._handleTouchStartRotate(e)}_handleTouchMoveRotate(e){if(this._pointers.length==1)this._rotateEnd.set(e.pageX,e.pageY);else{let n=this._getSecondPointerPosition(e),s=.5*(e.pageX+n.x),r=.5*(e.pageY+n.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let t=this.domElement;this._rotateLeft(En*this._rotateDelta.x/t.clientHeight),this._rotateUp(En*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(e){if(this._pointers.length===1)this._panEnd.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panEnd.set(n,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(e){let t=this._getSecondPointerPosition(e),n=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(n*n+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);let a=(e.pageX+t.x)*.5,o=(e.pageY+t.y)*.5;this._updateZoomParameters(a,o)}_handleTouchMoveDollyPan(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enablePan&&this._handleTouchMovePan(e)}_handleTouchMoveDollyRotate(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enableRotate&&this._handleTouchMoveRotate(e)}_addPointer(e){this._pointers.push(e.pointerId)}_removePointer(e){delete this._pointerPositions[e.pointerId];for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId){this._pointers.splice(t,1);return}}_isTrackingPointer(e){for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId)return!0;return!1}_trackPointer(e){let t=this._pointerPositions[e.pointerId];t===void 0&&(t=new Z,this._pointerPositions[e.pointerId]=t),t.set(e.pageX,e.pageY)}_getSecondPointerPosition(e){let t=e.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[t]}_customWheelEvent(e){let t=e.deltaMode,n={clientX:e.clientX,clientY:e.clientY,deltaY:e.deltaY};switch(t){case 1:n.deltaY*=16;break;case 2:n.deltaY*=100;break}return e.ctrlKey&&!this._controlActive&&(n.deltaY*=10),n}};function cM(i){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(i.pointerId),this.domElement.ownerDocument.addEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(i)&&(this._addPointer(i),i.pointerType==="touch"?this._onTouchStart(i):this._onMouseDown(i),this._cursorStyle==="grab"&&(this.domElement.style.cursor="grabbing")))}function hM(i){this.enabled!==!1&&(i.pointerType==="touch"?this._onTouchMove(i):this._onMouseMove(i))}function uM(i){switch(this._removePointer(i),this._pointers.length){case 0:this.domElement.releasePointerCapture(i.pointerId),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Hp),this.state=xt.NONE,this._cursorStyle==="grab"&&(this.domElement.style.cursor="grab");break;case 1:let e=this._pointers[0],t=this._pointerPositions[e];this._onTouchStart({pointerId:e,pageX:t.x,pageY:t.y});break}}function dM(i){let e;switch(i.button){case 0:e=this.mouseButtons.LEFT;break;case 1:e=this.mouseButtons.MIDDLE;break;case 2:e=this.mouseButtons.RIGHT;break;default:e=-1}switch(e){case ls.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(i),this.state=xt.DOLLY;break;case ls.ROTATE:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=xt.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=xt.ROTATE}break;case ls.PAN:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=xt.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=xt.PAN}break;default:this.state=xt.NONE}this.state!==xt.NONE&&this.dispatchEvent(Qu)}function fM(i){switch(this.state){case xt.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(i);break;case xt.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(i);break;case xt.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(i);break}}function pM(i){this.enabled===!1||this.enableZoom===!1||this.state!==xt.NONE||(i.preventDefault(),this.dispatchEvent(Qu),this._handleMouseWheel(this._customWheelEvent(i)),this.dispatchEvent(Hp))}function mM(i){this.enabled!==!1&&this._handleKeyDown(i)}function gM(i){switch(this._trackPointer(i),this._pointers.length){case 1:switch(this.touches.ONE){case cs.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(i),this.state=xt.TOUCH_ROTATE;break;case cs.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(i),this.state=xt.TOUCH_PAN;break;default:this.state=xt.NONE}break;case 2:switch(this.touches.TWO){case cs.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(i),this.state=xt.TOUCH_DOLLY_PAN;break;case cs.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(i),this.state=xt.TOUCH_DOLLY_ROTATE;break;default:this.state=xt.NONE}break;default:this.state=xt.NONE}this.state!==xt.NONE&&this.dispatchEvent(Qu)}function _M(i){switch(this._trackPointer(i),this.state){case xt.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(i),this.update();break;case xt.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(i),this.update();break;case xt.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(i),this.update();break;case xt.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(i),this.update();break;default:this.state=xt.NONE}}function xM(i){this.enabled!==!1&&i.preventDefault()}function vM(i){i.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function yM(i){i.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}var Yc=class extends Ds{constructor(){super(),this.name="RoomEnvironment",this.position.y=-3.5;let e=new Bt;e.deleteAttribute("uv");let t=new Ke({side:tn}),n=new Ke,s=new ka(16777215,900,28,2);s.position.set(.418,16.199,.3),this.add(s);let r=new et(e,t);r.position.set(-.757,13.219,.717),r.scale.set(31.713,28.305,28.591),this.add(r);let a=new jt(e,n,6),o=new ft;o.position.set(-10.906,2.009,1.846),o.rotation.set(0,-.195,0),o.scale.set(2.328,7.905,4.651),o.updateMatrix(),a.setMatrixAt(0,o.matrix),o.position.set(-5.607,-.754,-.758),o.rotation.set(0,.994,0),o.scale.set(1.97,1.534,3.955),o.updateMatrix(),a.setMatrixAt(1,o.matrix),o.position.set(6.167,.857,7.803),o.rotation.set(0,.561,0),o.scale.set(3.927,6.285,3.687),o.updateMatrix(),a.setMatrixAt(2,o.matrix),o.position.set(-2.017,.018,6.124),o.rotation.set(0,.333,0),o.scale.set(2.002,4.566,2.064),o.updateMatrix(),a.setMatrixAt(3,o.matrix),o.position.set(2.291,-.756,-2.621),o.rotation.set(0,-.286,0),o.scale.set(1.546,1.552,1.496),o.updateMatrix(),a.setMatrixAt(4,o.matrix),o.position.set(-2.193,-.369,-5.547),o.rotation.set(0,.516,0),o.scale.set(3.875,3.487,2.986),o.updateMatrix(),a.setMatrixAt(5,o.matrix),this.add(a);let c=new et(e,Br(50));c.position.set(-16.116,14.37,8.208),c.scale.set(.1,2.428,2.739),this.add(c);let l=new et(e,Br(50));l.position.set(-16.109,18.021,-8.207),l.scale.set(.1,2.425,2.751),this.add(l);let h=new et(e,Br(17));h.position.set(14.904,12.198,-1.832),h.scale.set(.15,4.265,6.331),this.add(h);let d=new et(e,Br(43));d.position.set(-.462,8.89,14.52),d.scale.set(4.38,5.441,.088),this.add(d);let u=new et(e,Br(20));u.position.set(3.235,11.486,-12.541),u.scale.set(2.5,2,.1),this.add(u);let f=new et(e,Br(100));f.position.set(0,20,0),f.scale.set(1,.1,1),this.add(f)}dispose(){let e=new Set;this.traverse(t=>{t.isMesh&&(e.add(t.geometry),e.add(t.material))});for(let t of e)t.dispose()}};function Br(i){return new Fa({color:0,emissive:16777215,emissiveIntensity:i})}var Vp=new mn,Zc=new R,ks=class extends Ha{constructor(){super(),this.isLineSegmentsGeometry=!0,this.type="LineSegmentsGeometry";let e=[-1,2,0,1,2,0,-1,1,0,1,1,0,-1,0,0,1,0,0,-1,-1,0,1,-1,0],t=[-1,2,1,2,-1,1,1,1,-1,-1,1,-1,-1,-2,1,-2],n=[0,2,1,2,3,1,2,4,3,4,5,3,4,6,5,6,7,5];this.setIndex(n),this.setAttribute("position",new rt(e,3)),this.setAttribute("uv",new rt(t,2))}applyMatrix4(e){let t=this.attributes.instanceStart,n=this.attributes.instanceEnd;return t!==void 0&&(t.applyMatrix4(e),n.applyMatrix4(e),t.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}setPositions(e){let t;e instanceof Float32Array?t=e:Array.isArray(e)&&(t=new Float32Array(e));let n=new os(t,6,1);return this.setAttribute("instanceStart",new Ln(n,3,0)),this.setAttribute("instanceEnd",new Ln(n,3,3)),this.instanceCount=this.attributes.instanceStart.count,this.computeBoundingBox(),this.computeBoundingSphere(),this}setColors(e){let t;e instanceof Float32Array?t=e:Array.isArray(e)&&(t=new Float32Array(e));let n=new os(t,6,1);return this.setAttribute("instanceColorStart",new Ln(n,3,0)),this.setAttribute("instanceColorEnd",new Ln(n,3,3)),this}fromWireframeGeometry(e){return this.setPositions(e.attributes.position.array),this}fromEdgesGeometry(e){return this.setPositions(e.attributes.position.array),this}fromMesh(e){return this.fromWireframeGeometry(new Da(e.geometry)),this}fromLineSegments(e){let t=e.geometry;return this.setPositions(t.attributes.position.array),this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new mn);let e=this.attributes.instanceStart,t=this.attributes.instanceEnd;e!==void 0&&t!==void 0&&(this.boundingBox.setFromBufferAttribute(e),Vp.setFromBufferAttribute(t),this.boundingBox.union(Vp))}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new yn),this.boundingBox===null&&this.computeBoundingBox();let e=this.attributes.instanceStart,t=this.attributes.instanceEnd;if(e!==void 0&&t!==void 0){let n=this.boundingSphere.center;this.boundingBox.getCenter(n);let s=0;for(let r=0,a=e.count;r<a;r++)Zc.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(Zc)),Zc.fromBufferAttribute(t,r),s=Math.max(s,n.distanceToSquared(Zc));this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&console.error("THREE.LineSegmentsGeometry.computeBoundingSphere(): Computed radius is NaN. The instanced position data is likely to have NaN values.",this)}}toJSON(){}};be.line={worldUnits:{value:1},linewidth:{value:1},resolution:{value:new Z},dashOffset:{value:0},dashScale:{value:1},dashSize:{value:1},gapSize:{value:1}};_n.line={uniforms:gn.merge([be.common,be.fog,be.line]),vertexShader:`
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
		`};var zr=class extends bt{constructor(e){super({type:"LineMaterial",uniforms:gn.clone(_n.line.uniforms),vertexShader:_n.line.vertexShader,fragmentShader:_n.line.fragmentShader,clipping:!0}),this.isLineMaterial=!0,this.setValues(e)}get color(){return this.uniforms.diffuse.value}set color(e){this.uniforms.diffuse.value=e}get worldUnits(){return"WORLD_UNITS"in this.defines}set worldUnits(e){e===!0!==this.worldUnits&&(this.needsUpdate=!0),e===!0?this.defines.WORLD_UNITS="":delete this.defines.WORLD_UNITS}get linewidth(){return this.uniforms.linewidth.value}set linewidth(e){this.uniforms.linewidth&&(this.uniforms.linewidth.value=e)}get dashed(){return"USE_DASH"in this.defines}set dashed(e){e===!0!==this.dashed&&(this.needsUpdate=!0),e===!0?this.defines.USE_DASH="":delete this.defines.USE_DASH}get dashScale(){return this.uniforms.dashScale.value}set dashScale(e){this.uniforms.dashScale.value=e}get dashSize(){return this.uniforms.dashSize.value}set dashSize(e){this.uniforms.dashSize.value=e}get dashOffset(){return this.uniforms.dashOffset.value}set dashOffset(e){this.uniforms.dashOffset.value=e}get gapSize(){return this.uniforms.gapSize.value}set gapSize(e){this.uniforms.gapSize.value=e}get opacity(){return this.uniforms.opacity.value}set opacity(e){this.uniforms&&(this.uniforms.opacity.value=e)}get resolution(){return this.uniforms.resolution.value}set resolution(e){this.uniforms.resolution.value.copy(e)}get alphaToCoverage(){return"USE_ALPHA_TO_COVERAGE"in this.defines}set alphaToCoverage(e){this.defines&&(e===!0!==this.alphaToCoverage&&(this.needsUpdate=!0),e===!0?this.defines.USE_ALPHA_TO_COVERAGE="":delete this.defines.USE_ALPHA_TO_COVERAGE)}};var ed=new mt,Gp=new R,Wp=new R,sn=new mt,rn=new mt,yi=new mt,td=new R,nd=new st,an=new Wa,Xp=new R,$c=new mn,Jc=new yn,Mi=new mt,Si,Hs;function qp(i,e,t){return Mi.set(0,0,-e,1).applyMatrix4(i.projectionMatrix),Mi.multiplyScalar(1/Mi.w),Mi.x=Hs/t.width,Mi.y=Hs/t.height,Mi.applyMatrix4(i.projectionMatrixInverse),Mi.multiplyScalar(1/Mi.w),Math.abs(Math.max(Mi.x,Mi.y))}function MM(i,e){let t=i.matrixWorld,n=i.geometry,s=n.attributes.instanceStart,r=n.attributes.instanceEnd,a=Math.min(n.instanceCount,s.count);for(let o=0,c=a;o<c;o++){an.start.fromBufferAttribute(s,o),an.end.fromBufferAttribute(r,o),an.applyMatrix4(t);let l=new R,h=new R;Si.distanceSqToSegment(an.start,an.end,h,l),h.distanceTo(l)<Hs*.5&&e.push({point:h,pointOnLine:l,distance:Si.origin.distanceTo(h),object:i,face:null,faceIndex:o,uv:null,uv1:null})}}function SM(i,e,t){let n=e.projectionMatrix,r=i.material.resolution,a=i.matrixWorld,o=i.geometry,c=o.attributes.instanceStart,l=o.attributes.instanceEnd,h=Math.min(o.instanceCount,c.count),d=-e.near;Si.at(1,yi),yi.w=1,yi.applyMatrix4(e.matrixWorldInverse),yi.applyMatrix4(n),yi.multiplyScalar(1/yi.w),yi.x*=r.x/2,yi.y*=r.y/2,yi.z=0,td.copy(yi),nd.multiplyMatrices(e.matrixWorldInverse,a);for(let u=0,f=h;u<f;u++){if(sn.fromBufferAttribute(c,u),rn.fromBufferAttribute(l,u),sn.w=1,rn.w=1,sn.applyMatrix4(nd),rn.applyMatrix4(nd),sn.z>d&&rn.z>d)continue;if(sn.z>d){let S=sn.z-rn.z,y=(sn.z-d)/S;sn.lerp(rn,y)}else if(rn.z>d){let S=rn.z-sn.z,y=(rn.z-d)/S;rn.lerp(sn,y)}sn.applyMatrix4(n),rn.applyMatrix4(n),sn.multiplyScalar(1/sn.w),rn.multiplyScalar(1/rn.w),sn.x*=r.x/2,sn.y*=r.y/2,rn.x*=r.x/2,rn.y*=r.y/2,an.start.copy(sn),an.start.z=0,an.end.copy(rn),an.end.z=0;let _=an.closestPointToPointParameter(td,!0);an.at(_,Xp);let p=Vt.lerp(sn.z,rn.z,_),m=p>=-1&&p<=1,M=td.distanceTo(Xp)<Hs*.5;if(m&&M){an.start.fromBufferAttribute(c,u),an.end.fromBufferAttribute(l,u),an.start.applyMatrix4(a),an.end.applyMatrix4(a);let S=new R,y=new R;Si.distanceSqToSegment(an.start,an.end,y,S),t.push({point:y,pointOnLine:S,distance:Si.origin.distanceTo(y),object:i,face:null,faceIndex:u,uv:null,uv1:null})}}}var jc=class extends et{constructor(e=new ks,t=new zr({color:Math.random()*16777215})){super(e,t),this.isLineSegments2=!0,this.type="LineSegments2"}computeLineDistances(){let e=this.geometry,t=e.attributes.instanceStart,n=e.attributes.instanceEnd,s=new Float32Array(2*t.count);for(let a=0,o=0,c=t.count;a<c;a++,o+=2)Gp.fromBufferAttribute(t,a),Wp.fromBufferAttribute(n,a),s[o]=o===0?0:s[o-1],s[o+1]=s[o]+Gp.distanceTo(Wp);let r=new os(s,2,1);return e.setAttribute("instanceDistanceStart",new Ln(r,1,0)),e.setAttribute("instanceDistanceEnd",new Ln(r,1,1)),this}raycast(e,t){let n=this.material.worldUnits,s=e.camera;if(s===null&&!n&&console.error('LineSegments2: "Raycaster.camera" needs to be set in order to raycast against LineSegments2 while worldUnits is set to false.'),n===!1&&(this.material.resolution.x===0||this.material.resolution.y===0))return;let r=e.params.Line2!==void 0&&e.params.Line2.threshold||0;Si=e.ray;let a=this.matrixWorld,o=this.geometry,c=this.material;Hs=c.linewidth+r,o.boundingSphere===null&&o.computeBoundingSphere(),Jc.copy(o.boundingSphere).applyMatrix4(a);let l;if(n)l=Hs*.5;else{let d=Math.max(s.near,Jc.distanceToPoint(Si.origin));l=qp(s,d,c.resolution)}if(Jc.radius+=l,Si.intersectsSphere(Jc)===!1)return;o.boundingBox===null&&o.computeBoundingBox(),$c.copy(o.boundingBox).applyMatrix4(a);let h;if(n)h=Hs*.5;else{let d=Math.max(s.near,$c.distanceToPoint(Si.origin));h=qp(s,d,c.resolution)}$c.expandByScalar(h),Si.intersectsBox($c)!==!1&&(n?MM(this,t):SM(this,s,t))}onBeforeRender(e){let t=this.material.uniforms;t&&t.resolution&&(e.getViewport(ed),this.material.uniforms.resolution.value.set(ed.z,ed.w))}};var ho=new R;function Gn(i,e,t,n,s,r){let a=2*Math.PI*s/4,o=Math.max(r-2*s,0),c=Math.PI/4;ho.copy(e),ho[n]=0,ho.normalize();let l=.5*a/(a+o),h=1-ho.angleTo(i)/c;return Math.sign(ho[t])===1?h*l:o/(a+o)+l+l*(1-h)}var Kc=class i extends Bt{constructor(e=1,t=1,n=1,s=2,r=.1){let a=s*2+1;if(r=Math.min(e/2,t/2,n/2,r),super(1,1,1,a,a,a),this.type="RoundedBoxGeometry",this.parameters={width:e,height:t,depth:n,segments:s,radius:r},a===1)return;let o=this.toNonIndexed();this.index=null,this.attributes.position=o.attributes.position,this.attributes.normal=o.attributes.normal,this.attributes.uv=o.attributes.uv;let c=new R,l=new R,h=new R(e,t,n).divideScalar(2).subScalar(r),d=this.attributes.position.array,u=this.attributes.normal.array,f=this.attributes.uv.array,g=d.length/6,_=new R,p=.5/a;for(let m=0,M=0;m<d.length;m+=3,M+=2)switch(c.fromArray(d,m),l.copy(c),l.x-=Math.sign(l.x)*p,l.y-=Math.sign(l.y)*p,l.z-=Math.sign(l.z)*p,l.normalize(),d[m+0]=h.x*Math.sign(c.x)+l.x*r,d[m+1]=h.y*Math.sign(c.y)+l.y*r,d[m+2]=h.z*Math.sign(c.z)+l.z*r,u[m+0]=l.x,u[m+1]=l.y,u[m+2]=l.z,Math.floor(m/g)){case 0:_.set(1,0,0),f[M+0]=Gn(_,l,"z","y",r,n),f[M+1]=1-Gn(_,l,"y","z",r,t);break;case 1:_.set(-1,0,0),f[M+0]=1-Gn(_,l,"z","y",r,n),f[M+1]=1-Gn(_,l,"y","z",r,t);break;case 2:_.set(0,1,0),f[M+0]=1-Gn(_,l,"x","z",r,e),f[M+1]=Gn(_,l,"z","x",r,n);break;case 3:_.set(0,-1,0),f[M+0]=1-Gn(_,l,"x","z",r,e),f[M+1]=1-Gn(_,l,"z","x",r,n);break;case 4:_.set(0,0,1),f[M+0]=1-Gn(_,l,"x","y",r,e),f[M+1]=1-Gn(_,l,"y","x",r,t);break;case 5:_.set(0,0,-1),f[M+0]=Gn(_,l,"x","y",r,e),f[M+1]=1-Gn(_,l,"y","x",r,t);break}}static fromJSON(e){return new i(e.width,e.height,e.depth,e.segments,e.radius)}};var Kp=[16756767,3262128,16740193,10194175],id=new Map;function Gt(i,e={}){let t=`${i}:${JSON.stringify(e)}`;return id.has(t)||id.set(t,new Ke({color:i,roughness:.52,metalness:0,...e})),id.get(t)}var We={dark:Gt(3883079,{roughness:.7}),darker:Gt(2830133,{roughness:.8}),rubber:Gt(2763824,{roughness:.95}),metal:Gt(10989748,{roughness:.35,metalness:.55}),chrome:Gt(14936298,{roughness:.18,metalness:.85}),glass:Gt(8242390,{roughness:.08,metalness:.1,emissive:1454650,emissiveIntensity:.35}),seat:Gt(3093304,{roughness:.9}),soil:Gt(11039551,{roughness:1,flatShading:!0}),lamp:Gt(16773570,{emissive:16770720,emissiveIntensity:.9}),tail:Gt(16730682,{emissive:12590608,emissiveIntensity:.6}),white:Gt(16052712,{roughness:.6})},bM=[{body:16036379,accent:16765788,trim:3883079},{body:14964026,accent:16165179,trim:3883079,bed:15903035},{body:16022304,accent:16752717,trim:3093304}],Yp=[14721067,4158630,5934942,9071536],EM=13194813,wM=14645804,sd=new Map;function TM(i){return sd.has(i)||sd.set(i,new Ua({color:i,roughness:.4,metalness:.05,clearcoat:.55,clearcoatRoughness:.28})),sd.get(i)}var Tn={glass:Gt(3820888,{roughness:.07,metalness:.25}),steel:Gt(4935766,{roughness:.52,metalness:.3}),worn:Gt(10330790,{roughness:.32,metalness:.8}),lamp:Gt(16183256,{emissive:16771512,emissiveIntensity:.25}),soil:Gt(7295544,{roughness:1,flatShading:!0})};function AM(i,e){if(e!=="studio")return{...bM[i.type],paint:Gt};let t=i.type===0?Yp[i.id%Yp.length]:i.type===1?EM:wM;return{body:t,accent:t,trim:3027511,bed:t,studio:!0,paint:TM}}var ud=new Bt(1,1,1),Vs=new Qn(1,1,1,24),dd=new Qn(1,1,1,10);function $t(i,e,t,n=0,s=0,r=0){let a=new et(e,t);return a.position.set(n,s,r),a.castShadow=!0,a.receiveShadow=!0,i.add(a),a}function at(i,e,t,n,s,r,a,o){let c=$t(i,ud,e,t,n,s);return c.scale.set(r,a,o),c}function Wn(i,e,t,n,s,r,a,o,c){return $t(i,new Kc(r,a,o,3,Math.min(c,r/2,a/2,o/2)),e,t,n,s)}function vn(i,e,t,n,s,r,a,o=0,c=Vs){let l=$t(i,c,e,t,n,s);return l.scale.set(r,a,r),l.rotation.x=o,l}function uo(i,e,t,n,s){let r=new R(...t),a=new R(...n),o=a.clone().sub(r),c=$t(i,Vs,e);return c.position.copy(r.add(a).multiplyScalar(.5)),c.scale.set(s,o.length(),s),c.quaternion.setFromUnitVectors(new R(0,1,0),o.normalize()),c}function rd(i,e,t,n,s,r=!1){let a=new gi;a.moveTo(0,-t*.4),r?(a.quadraticCurveTo(-t*.14,t*.15,e*.18,t*.5),a.quadraticCurveTo(e*.24,t*.6,e*.33,t*.48),a.lineTo(e*.89,t*.2),a.quadraticCurveTo(e+t*.2,t*.2,e+t*.15,-t*.08),a.quadraticCurveTo(e+t*.1,-t*.4,e*.9,-t*.32),a.lineTo(e*.26,-t*.25)):(a.lineTo(e*.22,t*.55),a.lineTo(e*.7,t*.3),a.lineTo(e,t*.12),a.lineTo(e,-t*.32),a.lineTo(e*.24,-t*.25)),a.closePath();let o=new Fi(a,{depth:n,bevelEnabled:!0,bevelSegments:r?3:1,curveSegments:10,steps:1,bevelSize:t*.085,bevelThickness:t*.085});return $t(i,o,s,0,0,-n/2)}function dn(i,e,t,n,s){let r=new ft;return r.name=e,r.position.set(t,n,s),i.add(r),r}function xn(i,e,t,n,s,r,a){return vn(i,e,t,n,s,r,a,Math.PI/2)}function th(i,e,t,n){let s=t.clone().sub(e);i.position.copy(e).add(t).multiplyScalar(.5),i.scale.set(n,s.length(),n),i.quaternion.setFromUnitVectors(new R(0,1,0),s.normalize())}function ad(i,e,t,n,s){let r=$t(i,Vs,We.dark),a=$t(i,Vs,We.chrome);return r.name=`${s}-barrel`,a.name=`${s}-piston`,{start:e,end:t,barrel:r,piston:a,update(){let o=i.worldToLocal(e.getWorldPosition(new R)),c=i.worldToLocal(t.getWorldPosition(new R));th(r,o,o.clone().lerp(c,.58),n),th(a,o.clone().lerp(c,.43),c,n*.56)}}}function RM(i,e,t,n){let s=e.x-i.x,r=e.y-i.y,a=Math.hypot(s,r);if(a<=Math.abs(t-n)||a>=t+n)throw new Error("Bucket linkage pose is outside its mechanical range.");let o=(t**2-n**2+a**2)/(2*a),c=Math.sqrt(Math.max(0,t**2-o**2));return new R(i.x+(o*s-c*r)/a,i.y+(o*r+c*s)/a,0)}function fd(i,e,t){if(typeof document>"u")return null;let n=document.createElement("canvas");n.width=i,n.height=e,t(n.getContext("2d"));let s=new pi(n);return s.colorSpace=Lt,s}function CM(){return fd(128,32,i=>{i.fillStyle="#f4b21b",i.fillRect(0,0,128,32),i.fillStyle="#2b2f35";for(let e=-32;e<160;e+=24)i.beginPath(),i.moveTo(e,32),i.lineTo(e+12,32),i.lineTo(e+44,0),i.lineTo(e+32,0),i.fill()})}var od;function PM(i,e,t,n,s,r){let a=new tt;a.position.set(e,n,t),i.add(a);let o=new tt;a.add(o),vn(o,We.rubber,0,0,0,n*.9,s,Math.PI/2);let c=Math.sign(t)||1;vn(o,r,0,0,c*s*.47,n*.56,s*.12,Math.PI/2),vn(o,We.metal,0,0,c*s*.54,n*.2,s*.1,Math.PI/2,dd);for(let l=0;l<6;l++){let h=l*Math.PI/3;at(o,We.darker,Math.cos(h)*n*.36,Math.sin(h)*n*.36,c*s*.53,n*.09,n*.09,s*.05)}for(let l=0;l<14;l++){let h=l*Math.PI*2/14,d=at(o,We.rubber,Math.sin(h)*n*.93,Math.cos(h)*n*.93,(l%2?.18:-.18)*s,n*.26,n*.16,s*.6);d.rotation.z=-h}return{steer:a,spin:o,radius:n}}function IM(i,e,t,n,s,r){let a=new tt;a.position.z=s*t*.35,i.add(a);let o=n*.13,c=e*.34,l=n*.02,h=o+l,d=t*.2,u=4*c+2*Math.PI*o,f=new gi;f.moveTo(-c,l+o*.35),f.lineTo(c,l+o*.35),f.absarc(c,h,o*.65,-Math.PI/2,Math.PI/2,!1),f.lineTo(-c,h+o*.65),f.absarc(-c,h,o*.65,Math.PI/2,Math.PI*1.5,!1);let g=new Fi(f,{depth:d*.7,bevelEnabled:!0,bevelSegments:2,bevelSize:n*.012,bevelThickness:n*.012,steps:1});$t(a,g,r,0,0,-d*.35);let _=[];for(let x=0;x<5;x++)_.push(vn(a,We.metal,e*(-.26+x*.13),l+o*.42,s*d*.38,n*.045,d*.12,Math.PI/2));for(let x of[-c,c]){let E=new tt;E.position.set(x,h,0),a.add(E),_.push(E),vn(E,We.dark,0,0,0,o*.82,d*.86,Math.PI/2),vn(E,We.metal,0,0,s*d*.44,o*.38,d*.06,Math.PI/2,dd);for(let C=0;C<8;C++){let I=C*Math.PI/4;at(E,We.darker,Math.cos(I)*o*.6,Math.sin(I)*o*.6,s*d*.44,o*.14,o*.14,d*.04)}}let p=Math.max(24,Math.round(u/(n*.055))),m=u/p,M=new jt(ud,We.rubber,p);M.castShadow=!0,M.receiveShadow=!0,a.add(M);let S=new ft,y=n*.034,T=(x,E)=>{if(x=(x%u+u)%u,x<2*c){E.set(-c+x,l,-Math.PI/2);return}if(x-=2*c,x<Math.PI*o){let I=-Math.PI/2+x/o;E.set(c+Math.cos(I)*o,h+Math.sin(I)*o,I);return}if(x-=Math.PI*o,x<2*c){E.set(c-x,h+o,Math.PI/2);return}x-=2*c;let C=Math.PI/2+x/o;E.set(-c+Math.cos(C)*o,h+Math.sin(C)*o,C)},b=new R,P=x=>{for(let E=0;E<p;E++){T(E*m+x,b);let C=b.z;S.position.set(b.x+Math.cos(C)*y*.4,b.y+Math.sin(C)*y*.4,0),S.rotation.set(0,0,C-Math.PI/2),S.scale.set(m*.82,y,d),S.updateMatrix(),M.setMatrixAt(E,S.matrix)}M.instanceMatrix.needsUpdate=!0;for(let E of _)E.isGroup&&(E.rotation.z=-x/(o*.82))};return P(0),{update:P,shoes:M}}function DM(i,e,t,n,s,r,a){let o=[],c=[],l=[];if(at(i,We.dark,0,n*.22,0,e*.74,n*.17,t*.6),s){let h=n*(a?.23:.19),d=t*(a?.22:.18);for(let u of[-.3,.3])for(let f of[-.4,.4]){let g=PM(i,u*e,f*t,h,d,Gt(r.accent));c.push(g),u>0&&o.push(g.steer)}for(let u of[-.3,.3])uo(i,We.darker,[u*e,h,-.4*t],[u*e,h,.4*t],n*.05)}else for(let h of[-1,1])l.push({side:h,...IM(i,e,t,n,h,Gt(r.trim,{roughness:.7}))});return{steering:o,spinning:c,tracks:l}}function ld(i,e,t,n,s,r,a,o="x"){let c=Gt(16777215,{transparent:!0,opacity:.45,emissive:16777215,emissiveIntensity:.4,depthWrite:!1});for(let[l,h]of[[-.18,.16],[.1,.07]]){let d=at(i,c,e,t,n,s,r,a);d.castShadow=!1,o==="x"?(d.scale.set(s,r*1.2,a*h),d.position.z+=a*l*2.2,d.rotation.x=.5):(d.scale.set(s*h,r*1.2,a),d.position.x+=s*l*2.2,d.rotation.z=-.5),d.userData.skipAO=!0,d.name="glass-highlight"}}function Zp(i,e,t,n,s,r,a){let o=a.paint(a.body),c=new tt;c.position.set(s,0,r),i.add(c);let l=.3*e,h=.39*t,d=.5*n,u=a.studio?Tn.glass:We.glass;Wn(c,o,0,.12*n,0,l,.2*n,h,n*.04),Wn(c,o,0,.36*n,0,l*.96,d*.72,h*.96,n*.05).name="cab-shell";let f=.39*n,g=d*.56;at(c,u,l*.485,f,0,n*.012,g,h*.84),ld(c,l*.492,f,0,n*.01,g*.8,h*.84,"x");for(let _ of[-1,1])at(c,u,-l*.04,f,_*h*.485,l*.76,g,n*.012),ld(c,-l*.04,f,_*h*.492,l*.76,g*.8,n*.01,"z");at(c,u,-l*.485,f+g*.1,0,n*.012,g*.6,h*.7),at(c,We.seat,-l*.12,.3*n,0,l*.3,.16*n,h*.5),Wn(c,a.paint(a.studio?a.body:a.trim),0,.62*n,0,l*(a.studio?.98:1.06),n*.05,h*(a.studio?.98:1.06),n*.02);for(let _ of[-1,1])at(c,a.studio?Tn.lamp:We.lamp,l*.5,.6*n,_*h*.3,n*.02,n*.035,h*.12);return c}function Hr(i,e,t,n=12){let s=new Fi(i,{depth:e,bevelEnabled:t>0,bevelSegments:2,bevelSize:t,bevelThickness:t,curveSegments:n,steps:1});return s.translate(0,0,-e/2),s}function Vr(i){let e=new gi;return e.setFromPoints(i),e}function Qp(i,e,t){let n=i.map((s,r)=>{let a=i[Math.max(0,r-1)],o=i[Math.min(i.length-1,r+1)],c=new Z(a.y-o.y,o.x-a.x).normalize();return c.dot(t.clone().sub(s))<0&&c.negate(),s.clone().addScaledVector(c,e)});return[...i,...n.reverse()]}function LM(i){let e=[...i].sort((r,a)=>r.x-a.x||r.y-a.y),t=(r,a,o)=>(a.x-r.x)*(o.y-r.y)-(a.y-r.y)*(o.x-r.x),n=[],s=[];for(let r of e){for(;n.length>1&&t(n.at(-2),n.at(-1),r)<=0;)n.pop();n.push(r)}for(let r of e.reverse()){for(;s.length>1&&t(s.at(-2),s.at(-1),r)<=0;)s.pop();s.push(r)}return[...n.slice(0,-1),...s.slice(0,-1)]}function UM(i,e,t,n){let s=new tt;i.add(s),s.name="loader-bucket";let r=e,a=n.studio?Tn.steel:We.dark,o=n.studio?Tn.steel:n.paint(n.body),c=n.studio?Tn.worn:We.metal,l=new mi;l.moveTo(.29*r,-.155*r),l.lineTo(-.05*r,-.135*r),l.quadraticCurveTo(-.16*r,-.13*r,-.165*r,0*r),l.quadraticCurveTo(-.17*r,.13*r,-.12*r,.185*r);let h=l.getPoints(10),d=new Z(.05*r,.02*r);$t(s,Hr(Vr(Qp(h,.022*r,d)),t*.97,.004*r),a).name="loader-bucket-shell";let u=Vr([...h,new Z(-.04*r,.2*r),new Z(.07*r,.2*r)]),f=Hr(u,t*.045,.006*r);for(let p of[-1,1])$t(s,f,o,0,0,p*t*.49);at(s,a,0*r,.195*r,0,.17*r,.02*r,t*.99).rotation.z=.08;let g=at(s,c,.31*r,-.157*r,0,.07*r,.022*r,t*1);g.rotation.z=-.06,at(s,a,-.19*r,.03*r,0,.03*r,.26*r,t*.62);for(let p of[-1,1])at(s,a,-.21*r,.03*r,p*t*.2,.05*r,.24*r,.04*r);dn(s,"loader-edge",.34*r,-.15*r,0),dn(s,"loader-lip",.16*r,.05*r,0);let _=$t(s,pd(3),We.soil,.06*r,-.06*r,0);return _.name="bucket-soil",_.scale.set(r*.2,r*.14,t*.42),_.visible=!1,{root:s,soil:_}}var eh=new Map;function pd(i){if(eh.has(i))return eh.get(i);let e=new ei(1,1),t=e.attributes.position,n=i*9301+49297,s=()=>(n=n*16807%2147483647)/2147483647,r=new Map;for(let a=0;a<t.count;a++){let o=`${t.getX(a).toFixed(3)},${t.getY(a).toFixed(3)},${t.getZ(a).toFixed(3)}`;r.has(o)||r.set(o,.82+s()*.3);let c=r.get(o),l=t.getY(a);t.setXYZ(a,t.getX(a)*c,(l<0?l*.25:l)*c,t.getZ(a)*c)}return e.computeVertexNormals(),eh.set(i,e),e}function NM(i,e,t,n,s){let r=new tt;r.name="bucket-curl",i.add(r);let a=new tt;a.name="bucket-orientation",a.rotation.y=Math.PI,r.add(a);let o=e,c=s.studio?Tn.steel:We.dark,l=s.studio?Tn.steel:s.paint(s.body),h=s.studio?Tn.worn:We.metal,d=s.studio?Tn.steel:s.paint(s.accent),u=new Z(-.205*o,-.07*o),f=new Z(.22*o,-.53*o),g=new Z(.43*o,-.41*o),_=new Z(.125*o,-.05*o),p=new mi;p.moveTo(u.x,u.y),p.bezierCurveTo(-.33*o,-.15*o,-.345*o,-.36*o,-.245*o,-.47*o),p.bezierCurveTo(-.14*o,-.585*o,.07*o,-.61*o,f.x,f.y),p.lineTo(g.x,g.y);let m=p.getPoints(14),M=new Z(.07*o,-.3*o);$t(a,Hr(Vr(Qp(m,.03*o,M)),t*.96,.005*o,16),c).name="bucket-shell";let S=Hr(Vr([...m,_,u]),t*.05,.006*o,16);for(let j of[-1,1])$t(a,S,l,0,0,j*t*.475).name=`bucket-side-${j}`;let y=at(a,c,(u.x+_.x)/2,(u.y+_.y)/2-.012*o,0,_.distanceTo(u)+.02*o,.028*o,t*.97);y.rotation.z=Math.atan2(_.y-u.y,_.x-u.x);for(let j of[-.32,.32]){let he=at(a,h,-.06*o,-.585*o,j*t,.26*o,.018*o,t*.07);he.rotation.z=-.12}let T=g.clone().sub(f).normalize(),b=Math.atan2(T.y,T.x),P=at(a,h,g.x-T.x*.02*o,g.y-T.y*.02*o-.006*o,0,.11*o,.032*o,t*1);P.rotation.z=b,P.name="bucket-cutting-edge";let x=Vr([[0,.026],[.07,.021],[.13,.006],[.145,-.001],[.075,-.014],[0,-.02]].map(([j,he])=>new Z(j*o,he*o))),E=Math.min(.055*o,t*.13),C=Hr(x,E,.005*o,4),I=t>.3*o?5:4;for(let j=0;j<I;j++){let he=(j/(I-1)-.5)*t*.84,le=g.clone().addScaledVector(T,.03*o),Te=at(a,c,g.x,g.y+.004*o,he,.075*o,.045*o,E*1.35);Te.rotation.z=b;let Fe=$t(a,C,h,le.x,le.y,he);Fe.rotation.z=b,Fe.name="bucket-tooth"}let L=g.clone().addScaledVector(T,.175*o),X=t*.075,q=o*.008,F=n+o*.02,Y=(F+X)/2+q,W=Math.max(t*.72,F+2*X+4*q+o*.014),ie=new Z(-.08*o,.12*o),ne=(j,he)=>Array.from({length:20},(le,Te)=>j.clone().add(new Z(Math.cos(Te/20*Math.PI*2)*he,Math.sin(Te/20*Math.PI*2)*he))),ge=LM([...ne(new Z(0,0),.075*o),...ne(ie,.06*o),new Z(.09*o,-.06*o),new Z(-.175*o,-.075*o)]),ue=Hr(Vr(ge),X,q,10);for(let j of[-1,1])$t(a,ue,d,0,0,j*Y).name=`bucket-ear-${j}`;xn(a,h,0,0,0,o*.045,W).name="bucket-main-pin";let xe=dn(a,"bucket-link-pin",ie.x,ie.y,0);xn(xe,h,0,0,0,o*.032,W),dn(a,"bucket-teeth",L.x,L.y,0);let Ne=new Z((g.x+_.x)/2-.04*o,(g.y+_.y)/2),it=new Z(_.y-g.y,g.x-_.x).normalize();a.userData.opening=new R(it.x,it.y,0),dn(a,"bucket-lip",Ne.x,Ne.y,0);let Xe=$t(a,pd(1),We.soil,.09*o,-.28*o,0);return Xe.name="bucket-soil",Xe.scale.set(o*.25,o*.2,t*.4),Xe.visible=!1,{curl:r,orientation:a,soil:Xe,linkPin:xe,toothTip:L,lipPoint:Ne}}function FM(i,e){let t=`#${Kp[i.id%4].toString(16).padStart(6,"0")}`,n=String(i.id+1).padStart(2,"0"),s=fd(160,96,a=>{if(a.textAlign="center",a.textBaseline="middle",e==="paper"){a.font='600 44px Inter, "Helvetica Neue", Arial, sans-serif',a.lineJoin="round",a.lineWidth=10,a.strokeStyle="#ffffffee",a.strokeText(n,80,38),a.fillStyle="#1f2426",a.fillText(n,80,38),a.fillStyle="#ffffffee",a.beginPath(),a.roundRect(52,66,56,14,7),a.fill(),a.fillStyle=t,a.beginPath(),a.roundRect(56,69,48,8,4),a.fill();return}a.fillStyle="#00000033",a.beginPath(),a.roundRect(22,12,116,58,29),a.fill(),a.fillStyle=t,a.beginPath(),a.roundRect(20,8,120,58,29),a.fill(),a.beginPath(),a.moveTo(68,62),a.lineTo(92,62),a.lineTo(80,80),a.closePath(),a.fill(),a.lineWidth=5,a.strokeStyle="#ffffffcc",a.beginPath(),a.roundRect(22.5,10.5,115,53,26.5),a.stroke(),a.font='800 38px ui-rounded, "SF Pro Rounded", system-ui, sans-serif',a.fillStyle="#1f2a2c",a.fillText(n,80,39)}),r=new ga(new Sr({map:s,depthTest:!1,transparent:!0,sizeAttenuation:!1}));return r.center.set(.5,0),r.renderOrder=30,r.scale.set(.05,.03,1),r}function OM(i,e=!0){let t=`#${i.toString(16).padStart(6,"0")}`,n=fd(256,256,r=>{if(e){let a=r.createRadialGradient(128,128,60,128,128,126);a.addColorStop(0,`${t}00`),a.addColorStop(.82,`${t}38`),a.addColorStop(1,`${t}00`),r.fillStyle=a,r.fillRect(0,0,256,256)}r.strokeStyle=t,r.lineWidth=9,r.lineCap="round";for(let a=0;a<16;a++)r.beginPath(),r.arc(128,128,112,a*Math.PI/8+.06,(a+.62)*Math.PI/8),r.stroke()}),s=new et(new Hn(1,1),new Li({map:n,transparent:!0,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-4}));return s.rotation.x=-Math.PI/2,s.renderOrder=16,s.userData.skipAO=!0,s}var Zt=i=>i*i*(3-2*i),nh=i=>Math.min(1,Math.max(0,i)),cd=i=>1+(1.9+1)*(i-1)**3+1.9*(i-1)**2,Yt=(i,e,t)=>nh((i-e)/(t-e));function BM(i,e,t){for(let n=1;n<e.length;n++){let[s,r,a=Zt]=e[n],[o,c]=e[n-1];if(t<=s||n===e.length-1){let l=a(Yt(t,o,s)),h=i[c],d=i[r];return Object.fromEntries(Object.keys(h).map(u=>[u,h[u]+(d[u]-h[u])*l]))}}return i[e[0][1]]}var kr={carry:{boom:.86,stick:-1.92,pitch:.04},reach:{boom:.3,stick:-1.05,pitch:-.42},scoop:{boom:.2,stick:-1.3,pitch:.62},raise:{boom:.8,stick:-1.02,pitch:.1},pour:{boom:.74,stick:-.98,pitch:.94}},$p={dig:[[0,"carry"],[.3,"reach"],[.56,"scoop"],[1,"carry",cd]],dump:[[0,"carry"],[.34,"raise"],[.62,"pour"],[1,"carry",cd]]},_s={empty:-.12,loaded:-.5},wn={ground:{arm:-.33,curl:-.06},raised:{arm:.45},tipped:{curl:-.95}},ii=i=>i<.5?4*i*i*i:1-(-2*i+2)**3/2,zM=i=>i*i,Jp=i=>1-(1-i)*(1-i),Qc=i=>.75*i+.25*Zt(i),fo=(i,e,t)=>Math.min(t,Math.max(e,i));function jp(i,e){let t=1;for(;t<i.length-1&&e>i[t].t;)t++;let n=i[t-1],s=i[t],r=(s.ease??Zt)(nh((e-n.t)/Math.max(1e-6,s.t-n.t))),a={};for(let o of Object.keys(s))o==="t"||o==="ease"||(a[o]=s[o]?.isVector2?n[o].clone().lerp(s[o],r):n[o]+(s[o]-n[o])*r);return a}var hd=(i,e)=>new Z(i.x*Math.cos(e)-i.y*Math.sin(e),i.x*Math.sin(e)+i.y*Math.cos(e));function kM(i,e,t,n,s){let r=new Z(i.position.x,i.position.y),a=d=>new Z(-d.x,d.y),o=a(n.toothTip),c=a(n.lipPoint),l=(d,u)=>new Z(e*Math.cos(d)+t*Math.cos(d+u),e*Math.sin(d)+t*Math.sin(d+u));function h(d){let u=d.clone(),f=(e+t)*.995,g=Math.abs(e-t)*1.05+1e-6,_=u.length();_>f?u.multiplyScalar(f/_):_<g&&u.multiplyScalar(g/Math.max(_,1e-6));let p=u.length(),m=-Math.acos(fo((p*p-e*e-t*t)/(2*e*t),-1,1));return{boom:Math.atan2(u.y,u.x)-Math.atan2(t*Math.sin(m),e+t*Math.cos(m)),stick:m}}return{pivot:r,size:s,hinge:l,solveHinge:h,teeth:o,lip:c,tip:(d,u,f)=>l(d,u).add(hd(o,f)),solveTip:(d,u)=>h(d.clone().sub(hd(o,u)))}}function em(i,e,{labels:t=!0,style:n="diorama"}={}){let s=new tt,r=i.height*e,a=i.width*e,o=Math.min(r,a),c=AM(i,n),l=i.action_type===1||i.type===1,h=n==="studio";s.name=`machine-${i.id}`;let d=DM(s,r,a,o,l,c,i.type===1),u=new tt;u.name="suspension",s.add(u);let f=new tt;f.position.y=.32*o,u.add(f);let g=c.paint(c.body),_=c.paint(c.accent),p=c.paint(c.trim),m=h?Tn.soil:We.soil,M=h?Tn.steel:g,S=h?Tn.steel:_,y=new Ke({color:16753183,roughness:.3,emissive:16742912,emissiveIntensity:.2,transparent:!0,opacity:.92}),T,b,P,x,E,C,I,L,X,q=null,F=null,Y=null,W=[];if(od||(od=new Ke({map:CM(),color:typeof document>"u"?16036379:16777215,roughness:.6})),i.type===0){vn(f,We.dark,0,.045*o,0,o*.31,o*.1),Wn(f,g,-.1*r,.16*o,0,r*.65,o*.23,a*.66,o*.065).name="excavator-upper-body",Wn(f,h?p:We.dark,-.33*r,.255*o,0,r*.2,o*.2,a*.65,o*.068).name="excavator-counterweight",at(f,h?p:od,-.434*r,.255*o,0,r*.012,o*.09,a*.56);for(let v of[-1,1])at(f,We.tail,-.434*r,.3*o,v*a*.29,r*.012,o*.03,a*.05);Wn(f,_,-.22*r,.29*o,.16*a,r*.26,o*.05,a*.3,o*.02);for(let v=0;v<5;v++)at(f,We.darker,(-.3+v*.04)*r,.318*o,.16*a,r*.018,o*.012,a*.22);Zp(f,r,a,o,-.02*r,-.18*a,c),X=vn(f,y,-.1*r,.7*o,-.18*a,o*.032,o*.06),vn(f,We.dark,-.1*r,.665*o,-.18*a,o*.04,o*.02),vn(f,We.dark,-.28*r,.45*o,.26*a,o*.026,o*.32),vn(f,We.darker,-.28*r,.62*o,.26*a,o*.034,o*.03),L=dn(f,"exhaust",-.28*r,.66*o,.26*a),uo(f,We.metal,[-.33*r,.4*o,.32*a],[-.12*r,.4*o,.32*a],o*.012);for(let v of[-.33,-.12])uo(f,We.metal,[v*r,.27*o,.32*a],[v*r,.4*o,.32*a],o*.012);let oe=Math.max(r*.63,i.reach[1]*e*.4),ee=Math.max(r*.48,i.reach[1]*e*.32);T=new tt,T.name="boom-pivot",T.position.set(.16*r,.23*o,.09*a),f.add(T),rd(T,oe,o*.21,a*.12,_,!0),xn(T,We.dark,0,0,0,o*.095,a*.19),xn(T,We.metal,0,0,0,o*.05,a*.205);for(let v of[-1,1])at(T,h?Tn.lamp:We.lamp,oe*.3,o*.1,v*a*.065,o*.04,o*.03,o*.012);for(let v of[-1,1]){let U=dn(f,`boom-cylinder-${v}-start`,.2*r,.12*o,(.09+v*.12)*a),B=dn(T,`boom-cylinder-${v}-end`,oe*.48,-.055*o,v*a*.12);xn(U,M,0,0,0,o*.047,a*.055),xn(B,M,0,0,0,o*.047,a*.055),W.push(ad(f,U,B,o*.036,`boom-cylinder-${v}`))}b=new tt,b.name="stick-pivot",b.position.x=oe,T.add(b),rd(b,ee,o*.17,a*.09,g,!0),Wn(b,g,-.07*ee,.045*o,0,.22*ee,o*.105,a*.09,o*.035),xn(b,We.dark,0,0,0,o*.078,a*.16),xn(b,We.metal,0,0,0,o*.04,a*.175);let O=dn(T,"stick-cylinder-start",oe*.4,o*.145,0),H=dn(b,"stick-cylinder-end",-.09*ee,o*.08,0);xn(O,S,0,0,0,o*.048,a*.11),xn(H,M,0,0,0,o*.043,a*.115),W.push(ad(f,O,H,o*.04,"stick-cylinder"));let Q=c.studio?.56:.65,G=o*Q,V=a*.39*Q,se=a*.145,ce=NM(b,G,V,se,c);P=ce.curl,P.position.x=ee,x=ce.soil,x.material=m,F=ce.orientation.getObjectByName("bucket-teeth"),Y=ce.orientation.getObjectByName("bucket-lip"),dn(b,"bucket-hinge",ee,0,0),xn(b,g,ee,0,0,G*.068,se).name="bucket-hinge-housing";let fe=dn(b,"bucket-rocker-pivot",ee-G*.24,G*.1,0);Wn(b,g,fe.position.x,.04*G,0,G*.115,G*.17,a*.105,G*.025),xn(fe,We.metal,0,0,0,G*.036,V*.72);let me=dn(b,"bucket-rocker-joint",0,0,0);xn(me,We.metal,0,0,0,G*.036,V*.72);let D=G*.22,Me=G*.25,Ve=[];for(let v of[-1,1]){let U=$t(b,Vs,g),B=$t(b,Vs,We.dark);U.name=`bucket-rocker-${v}`,B.name=`bucket-link-${v}`,Ve.push({first:U,second:B,z:v*V*.3})}let A=dn(b,"bucket-cylinder-start",ee*.24,o*.12,0);xn(A,M,0,0,0,G*.038,a*.115),W.push(ad(b,A,me,G*.03,"bucket-cylinder")),I={origin:fe,joint:me,destination:ce.linkPin,firstLength:D,secondLength:Me,links:Ve,update(){let v=b.worldToLocal(ce.linkPin.getWorldPosition(new R)),U=RM(fe.position,v,D,Me);me.position.copy(U);for(let B of Ve){let k=fe.position.clone(),pe=v.clone(),_e=U.clone();k.z=B.z,_e.z=B.z,pe.z=B.z,th(B.first,k,_e,G*.029),th(B.second,_e,pe,G*.025)}}},q=kM(T,oe,ee,ce,G)}else if(i.type===1){at(f,We.dark,0,.02*o,0,r*.92,.09*o,a*.62);for(let H of[-1,1])at(f,p,.05*r,.08*o,H*a*.44,r*.7,.05*o,a*.1);Wn(f,g,.36*r,.16*o,0,.22*r,.26*o,a*.82,o*.05),at(f,We.darker,.475*r,.15*o,0,r*.02,.15*o,a*.5);for(let H=0;H<4;H++)at(f,We.metal,.486*r,(.1+H*.035)*o,0,r*.01,o*.012,a*.44);for(let H of[-1,1])at(f,We.lamp,.478*r,.24*o,H*a*.32,r*.02,.05*o,a*.1),at(f,We.chrome,.44*r,.06*o,H*a*.37,r*.08,.04*o,a*.12);Zp(f,r*.92,a*1.62,o*.95,.3*r,-.14*a,c),L=dn(f,"exhaust",.2*r,.78*o,.3*a),vn(f,We.chrome,.2*r,.5*o,.3*a,o*.03,o*.52);let oe=new tt;oe.name="truck-bed",oe.position.set(-.43*r,.12*o,0),f.add(oe),E=oe;let ee=c.paint(c.bed);at(oe,ee,.3*r,0,0,r*.64,o*.08,a*.86);for(let H of[-1,1]){let Q=at(oe,ee,.3*r,.2*o,H*a*.41,.66*r,o*.38,a*.05);Q.rotation.x=H*.08;for(let G=0;G<4;G++)at(oe,_,(.06+G*.16)*r,.22*o,H*a*.44,r*.025,o*.34,a*.02);at(oe,_,.3*r,.4*o,H*a*.43,.66*r,o*.04,a*.07)}at(oe,ee,.62*r,.28*o,0,.04*r,o*.52,a*.86);let O=at(oe,ee,.72*r,.52*o,0,.22*r,o*.04,a*.86);O.rotation.z=-.06,at(oe,We.dark,-.02*r,.22*o,0,.03*r,o*.3,a*.78),x=$t(oe,pd(2),m,r*.3,o*.2,0),x.scale.set(r*.27,o*.2,a*.33);for(let H of[-1,1])at(f,We.tail,-.47*r,.05*o,H*.32*a,.02*r,.05*o,.1*a)}else{Wn(f,g,-.06*r,.12*o,0,.72*r,.26*o,.64*a,o*.05),Wn(f,We.dark,-.34*r,.2*o,0,.14*r,.22*o,.6*a,o*.04);for(let G=0;G<4;G++)at(f,We.darker,-.412*r,(.12+G*.045)*o,0,r*.01,o*.018,a*.46);let oe=new tt;oe.position.set(-.06*r,.25*o,0),f.add(oe);let ee=.34*r,O=.4*a,H=.46*o;for(let G of[-1,1])for(let V of[-1,1])at(oe,We.darker,G*ee*.47,H/2,V*O*.47,o*.035,H,o*.035);Wn(oe,g,0,H,0,ee*1.06,o*.05,O*1.08,o*.02),at(oe,h?Tn.glass:We.glass,ee*.47,H*.52,0,o*.01,H*.78,O*.86),ld(oe,ee*.478,H*.52,0,o*.01,H*.6,O*.86,"x");for(let G of[-1,1])for(let V=0;V<4;V++)at(oe,We.darker,(-.3+V*.2)*ee,H*.55,G*O*.47,o*.012,H*.8,o*.012);at(oe,We.seat,-ee*.1,H*.25,0,ee*.35,H*.3,O*.5),X=vn(oe,y,-ee*.3,H+o*.05,0,o*.03,o*.05),L=dn(f,"exhaust",-.36*r,.42*o,.2*a),vn(f,We.dark,-.36*r,.36*o,.2*a,o*.025,o*.14),C=new tt,C.name="loader-arm",C.position.set(-.18*r,.22*o,0),f.add(C);for(let G of[-1,1]){let V=new tt;V.position.z=G*a*.36,C.add(V),rd(V,r*.81,o*.13,a*.075,_),uo(V,We.chrome,[r*.1,-.08*o,0],[r*.5,-.06*o,0],o*.022),xn(V,We.metal,0,0,0,o*.05,a*.09)}uo(C,p,[r*.66,-.04*o,-a*.36],[r*.66,-.04*o,a*.36],o*.04);let Q=UM(C,o*1.22,a*.92,c);P=Q.root,P.position.set(.8*r,-.1*o,0),x=Q.soil,x.material=m,F=P.getObjectByName("loader-edge"),Y=P.getObjectByName("loader-lip")}X&&(X.name="beacon");let ie=Kp[i.id%4],ne=OM(ie,n==="diorama"),ge=n==="paper";n!=="diorama"&&s.traverse(oe=>{oe.name==="glass-highlight"&&(oe.visible=!1)}),ne.scale.set(r*1.34,a*1.34+(r-a)*.35,1),ne.position.y=e*.03,s.add(ne);let ue=null;t&&(ue=FM(i,n==="diorama"?"diorama":"paper"),ue.position.set(-.05*r,o*1.12,0),s.add(ue));let xe=new R,Ne={last:null,treads:[0,0],spin:0,active:!1,kind:"",phase:1,lift:0,tags:!0},it=d.spinning[0]?.radius??o*.2,Xe=x?x.scale.clone():null;function j(oe,ee){return s.updateWorldMatrix(!0,!0),oe.cells.map(O=>{let H=ee.worldToLocal(new R(O.x,O.before,O.z)),Q=ee.worldToLocal(new R(O.x,O.after,O.z));return{key:O.key,x:H.x,z:H.z,before:H.y,after:Q.y,weight:Math.max(1,Math.abs(O.delta??1))}})}let he=(oe,ee)=>oe.reduce((O,H)=>O+ee(H)*H.weight,0)/oe.reduce((O,H)=>O+H.weight,0),le=(oe,ee)=>O=>he(oe,H=>Zt(Yt(O,...ee.get(H.key))));function Te(oe){let ee=j(oe,f);if(!ee.length)return null;let O=he(ee,k=>k.x),H=he(ee,k=>k.z),Q=Math.hypot(O,H),G=T.position.z,V=Q>Math.abs(G)*1.5?fo(Math.asin(fo(G/Q,-1,1))-Math.atan2(H,O),-.6,.6):0,se=Math.cos(V),ce=Math.sin(V),fe=q.size,me=new Map;for(let k of ee)k.s=se*k.x-ce*k.z-q.pivot.x,k.before-=q.pivot.y,k.after-=q.pivot.y;let D=k=>q.tip(kr.carry.boom,kr.carry.stick,k?_s.loaded:_s.empty);if(oe.kind==="dig"){let k=Math.max(...ee.map(Ie=>Ie.s))+e*.3,pe=Math.min(...ee.map(Ie=>Ie.s))-e*.35,_e=Math.min(...ee.map(Ie=>Ie.after))-e*.05,te=Math.max(...ee.map(Ie=>Ie.before),_e+e*.2),re=[{t:0,point:D(!1),pitch:_s.empty},{t:.22,point:new Z(k+fe*.12,te+fe*.45),pitch:1.25,ease:ii},{t:.34,point:new Z(k,_e+e*.03),pitch:1.05,ease:zM},{t:.68,point:new Z(pe,_e),pitch:.4,ease:Qc},{t:.8,point:new Z(pe-fe*.08,te+fe*.3),pitch:-.55,ease:Jp},{t:1,point:D(!0),pitch:_s.loaded,ease:ii}],Se=Math.max(k-pe,1e-6);for(let Ie of ee){let ve=.34+.34*nh((k-Ie.s)/Se);me.set(Ie.key,[ve-.05,ve+.07])}return{kind:"dig",space:"tip",keys:re,yaw:V,yawWindow:[.24,.84],timing:me,fill:le(ee,me),events:{bite:.34,drag:[.34,.68],breakout:.76}}}let Me=1.8,Ve=he(ee,k=>k.s),A=Math.max(...ee.map(k=>Math.max(k.before,k.after))),v=new Z(Ve-hd(q.lip,Me).x,A+fe*.78),U=q.hinge(kr.carry.boom,kr.carry.stick),B=[{t:0,point:U,pitch:_s.loaded},{t:.3,point:v,pitch:-.4,ease:ii},{t:.6,point:v.clone().add(new Z(0,fe*.04)),pitch:Me,ease:ii},{t:.72,point:v.clone().add(new Z(0,fe*.07)),pitch:Me+.12,ease:Qc},{t:1,point:U,pitch:_s.empty,ease:ii}];for(let k of ee)me.set(k.key,[.46,.8]);return{kind:"dump",space:"hinge",keys:B,yaw:V,yawWindow:[.3,.78],timing:me,fill:k=>1-Zt(Yt(k,.38,.66)),events:{pour:[.38,.7]}}}function Fe(oe){let ee=j(oe,s);if(!ee.length)return null;let O=[C.rotation.z,P.rotation.z],H=(A,v,U)=>(C.rotation.z=A,P.rotation.z=v,s.updateWorldMatrix(!0,!0),s.worldToLocal(U.getWorldPosition(new R)).x),Q=H(wn.ground.arm,wn.ground.curl,F),G=H(wn.raised.arm,wn.tipped.curl,Y);C.rotation.z=O[0],P.rotation.z=O[1];let V=Math.min(...ee.map(A=>A.x)),se=Math.max(...ee.map(A=>A.x)),ce=new Map,fe=A=>({arm:A.shovel_lifted?.35:-.2,curl:A.loaded>0?.22:0,lunge:0}),me=fe(oe.from),D=fe(oe.to);if(oe.kind==="dig"){let A=fo(V-Q+e*.35,0,3.5),v=[{t:0,...me},{t:.2,arm:wn.ground.arm,curl:wn.ground.curl,lunge:A*.2,ease:ii},{t:.5,arm:wn.ground.arm,curl:wn.ground.curl,lunge:A,ease:Qc},{t:.62,arm:wn.ground.arm+.03,curl:.5,lunge:A,ease:Jp},{t:.8,arm:-.1,curl:.38,lunge:A*.5,ease:ii},{t:1,...D,ease:ii}],U=Math.max(se-V,1e-6);for(let B of ee){let k=.3+.2*nh((B.x-V)/U);ce.set(B.key,[k-.05,k+.07])}return{kind:"dig",space:"loader",keys:v,timing:ce,fill:le(ee,ce),events:{bite:.3,drag:[.3,.55]}}}let Me=fo(he(ee,A=>A.x)-G,0,3.5),Ve=[{t:0,...me},{t:.32,arm:wn.raised.arm,curl:.3,lunge:Me,ease:ii},{t:.55,arm:wn.raised.arm,curl:wn.tipped.curl,lunge:Me,ease:ii},{t:.68,arm:wn.raised.arm-.03,curl:wn.tipped.curl,lunge:Me,ease:Qc},{t:1,...D,ease:ii}];for(let A of ee)ce.set(A.key,[.46,.8]);return{kind:"dump",space:"loader",keys:Ve,timing:ce,fill:A=>1-Zt(Yt(A,.36,.6)),events:{pour:[.36,.66]}}}function ke(oe,ee,O=0,H="",Q=null){f.rotation.y=oe.cabin_yaw;for(let V of d.steering)V.rotation.y=Math.max(-.6,Math.min(.6,oe.wheel_angle*Math.PI/9));Ne.active=ee,Ne.kind=H,Ne.phase=O,ne.visible=ee&&Ne.tags&&!h;let G=oe.loaded>0;if(Q&&Xe){let V=Q.fill(O),se=.35+.65*V;x.visible=V>.03,x.scale.set(Xe.x*(.65+.35*V),Xe.y*se,Xe.z*(.75+.25*V))}else if(x.visible=G||H==="dump"&&O<.52||H==="transfer"&&O<.52||H==="dig"&&O>.5,H==="receive"&&(x.visible=O>.62),Xe&&i.type!==0){let V=H==="dump"?Math.max(1,oe.previous_loaded??oe.loaded):oe.loaded,se=.55+.45*(1-Math.exp(-Math.max(V,1)/18));H==="dump"&&(se*=1-Zt(Yt(O,.22,.5)),x.visible=O<.5),x.scale.set(Xe.x,Xe.y*Math.max(se,.02),Xe.z*(H==="dump"?.7+.3*se:1))}if(T){if(Q){let V=jp(Q.keys,O),[se,ce]=Q.yawWindow;f.rotation.y=oe.cabin_yaw+Q.yaw*Zt(Yt(O,0,se))*(1-Zt(Yt(O,ce,1)));let fe=Q.space==="hinge"?q.solveHinge(V.point):q.solveTip(V.point,V.pitch);T.rotation.z=fe.boom,b.rotation.z=fe.stick,P.rotation.z=V.pitch-fe.boom-fe.stick}else{let V=H==="dig"?$p.dig:H==="dump"||H==="transfer"?$p.dump:null,se={...kr,carry:{...kr.carry,pitch:G?_s.loaded:_s.empty}},ce=V?BM(se,V,O):se.carry;T.rotation.z=ce.boom,b.rotation.z=ce.stick,P.rotation.z=ce.pitch-T.rotation.z-b.rotation.z}s.updateWorldMatrix(!0,!0),I.update();for(let V of W)V.update()}if(E){let V=H==="dump"?O<.45?Zt(Yt(O,0,.45)):O<.7?1:1-Zt(Yt(O,.7,1)):0;E.rotation.z=.62*V}if(C){if(Q){let me=jp(Q.keys,O);C.rotation.z=me.arm,P.rotation.z=me.curl,me.lunge&&s.translateX(me.lunge);return}let V=oe.shovel_lifted?.35:-.2,se=G?.22:0,ce=V,fe=se;H==="dig"?(ce=O<.35?Vt.lerp(-.2,-.3,Zt(Yt(O,0,.35))):O<.6?-.3:Vt.lerp(-.3,V,cd(Yt(O,.6,1))),fe=O<.35?-.15*Zt(Yt(O,0,.35)):O<.6?Vt.lerp(-.15,.3,Zt(Yt(O,.35,.6))):Vt.lerp(.3,se,Zt(Yt(O,.6,1)))):H==="dump"&&(ce=O<.6?Vt.lerp(.35,.45,Zt(Yt(O,0,.35))):Vt.lerp(.45,V,Zt(Yt(O,.6,1))),fe=O<.3?.22*(1-Yt(O,0,.3)):O<.65?-.8*Zt(Yt(O,.3,.5)):Vt.lerp(-.8,se,Zt(Yt(O,.65,1)))),C.rotation.z=ce,P.rotation.z=fe}}return{root:s,agent:i,bucketRig:I,hydraulics:W,suspension:u,ringColor:ie,arm:q,setPose:ke,plan(oe){return q?Te(oe):C?Fe(oe):null},setTags(oe){Ne.tags=oe,ue&&(ue.visible=oe),ne.visible=oe&&Ne.active&&!h},drive(oe,ee){let O=Ne.last;if(Ne.last={x:oe.x,z:oe.z,yaw:ee},!O)return;let H=oe.x-O.x,Q=oe.z-O.z,G=Math.atan2(Math.sin(ee-O.yaw),Math.cos(ee-O.yaw));if(Math.hypot(H,Q)>r*1.5||Math.abs(G)>1.2)return;let V=H*Math.cos(ee)-Q*Math.sin(ee);Ne.speed=V;for(let[se,ce]of d.tracks.entries())Ne.treads[se]-=V+ce.side*a*.35*G,ce.update(Ne.treads[se]);Ne.spin-=V/it;for(let se of d.spinning)se.spin.rotation.z=Ne.spin},tick(oe,{move:ee=null,direction:O=1,reducedMotion:H=!1}={}){let Q=Ne.active;if(ge||h){y.emissiveIntensity=h?.08:.15,ne.material.opacity=Q&&ge?.9:0,ne.rotation.z=0,ue&&(ue.material.opacity=1,ue.position.y=o*1.12),u.position.y=0,u.rotation.z=h&&ee!==null&&!H?-Math.sin(ee*Math.PI*2)*.014*O:0;return}if(y.emissiveIntensity=Q?.5+.9*Math.max(0,Math.sin(oe*7))**3:.15,ne.material.opacity=Q?.75+.25*Math.sin(oe*3.2):0,ne.rotation.z=oe*.25,ue&&(ue.material.opacity=Q?1:.72,ue.position.y=o*1.12+(Q&&!H?Math.sin(oe*3)*o*.03:0)),H){u.position.y=0,u.rotation.z=0;return}let G=ee===null?0:-Math.sin(ee*Math.PI*2)*.035*O;u.rotation.z=G,u.position.y=Q?Math.sin(oe*41)*o*.0015:0,ee!==null&&(u.position.y+=Math.abs(Math.sin(ee*Math.PI*3))*o*.008)},tip(){return x.getWorldPosition(xe),xe.clone()},teeth(){return(F??x).getWorldPosition(new R)},lip(){return(Y??x).getWorldPosition(new R)},exhaust(){return L?L.getWorldPosition(new R):s.position.clone()},bedLip(){return E?E.localToWorld(new R(-.02*r,.05*o,0)):this.tip()},dispose(){let oe=new Set;s.traverse(ee=>{ee.geometry&&ee.geometry!==ud&&ee.geometry!==Vs&&ee.geometry!==dd&&![...eh.values()].includes(ee.geometry)&&oe.add(ee.geometry),ee.isSprite&&(ee.material.map.dispose(),ee.material.dispose())});for(let ee of oe)ee.dispose();ne.material.map?.dispose(),ne.material.dispose(),y.dispose();for(let ee of d.tracks)ee.shoes.dispose()}}}var tm="terra.viewer3d.v1",md=["Excavator","Truck","Skid steer"],HM=["Forward","Backward","Turn clockwise","Turn anticlockwise","Cabin clockwise","Cabin anticlockwise","Work","Wait"],VM=["action","target","padding","dumpability"],nm=["dumpability_static","interaction","traversability"],xs=i=>typeof i=="number"&&Number.isFinite(i),Bi=i=>Number.isSafeInteger(i);function At(i,e){if(!i)throw new Error(e)}function im(i){At(i&&typeof i=="object"&&i.schema===tm,`Expected a ${tm} replay.`),At(i.metadata&&typeof i.metadata.title=="string"&&typeof i.metadata.source=="string","Replay metadata must include title and source strings."),At(Array.isArray(i.frames)&&i.frames.length>0,"The replay contains no frames."),At(i.frames.length<=1e5,"This viewer supports at most 100,000 frames.");for(let[e,t]of i.frames.entries())gd(t,`Frame ${e}`);return i}function gd(i,e="Frame"){At(i&&typeof i=="object",`${e}: expected an object.`);let{grid:t,maps:n,agents:s}=i;At(t&&Bi(t.rows)&&Bi(t.cols)&&t.rows>0&&t.cols>0&&t.rows<=128&&t.cols<=128,`${e}: grid must be between 1 and 128 cells on each side.`),At(xs(t.tile_size_m)&&t.tile_size_m>0,`${e}: invalid tile size.`),At(n&&typeof n=="object",`${e}: missing maps.`);for(let a of[...VM,...nm]){let o=n[a];if(o==null&&nm.includes(a))continue;At(Array.isArray(o)&&o.length===t.rows,`${e}: ${a} has the wrong row count.`);let c=a==="action"||a==="target",l=a==="traversability"?[-1,0,1,!1,!0]:[0,1,!1,!0];for(let h of o)At(Array.isArray(h)&&h.length===t.cols&&h.every(d=>c?Bi(d):l.includes(d)),`${e}: ${a} has invalid cells or columns.`)}At(Bi(i.step)&&i.step>=0&&xs(i.reward),`${e}: invalid step or reward.`),At(i.action===null||Bi(i.action)&&i.action>=0&&i.action<=7,`${e}: invalid action.`),At(typeof i.done=="boolean"&&typeof i.task_done=="boolean",`${e}: invalid episode outcome.`),At(!i.task_done||i.done,`${e}: task_done requires done.`),At(Array.isArray(s)&&s.length>0&&s.length<=4,`${e}: expected 1\u20134 active agents.`);let r=new Set;for(let a of s)At(a&&Bi(a.id)&&a.id>=0&&a.id<=3&&!r.has(a.id),`${e}: agent IDs must be unique original slots from 0 to 3.`),r.add(a.id),At(Bi(a.type)&&a.type>=0&&a.type<=2&&(a.action_type===0||a.action_type===1),`${e}: unknown machine type.`),At(Array.isArray(a.position)&&a.position.length===2&&a.position.every(xs)&&a.position[0]>=0&&a.position[0]<t.rows&&a.position[1]>=0&&a.position[1]<t.cols,`${e}: agent position is outside the map.`),At(xs(a.base_yaw)&&xs(a.cabin_yaw)&&Bi(a.wheel_angle),`${e}: invalid machine angle.`),At(xs(a.width)&&xs(a.height)&&a.width>0&&a.height>0,`${e}: invalid machine footprint.`),At(Bi(a.loaded)&&a.loaded>=0&&(a.shovel_lifted===0||a.shovel_lifted===1),`${e}: invalid machine load or shovel state.`),At(Array.isArray(a.reach)&&a.reach.length===2&&a.reach.every(xs)&&a.reach[0]>=0&&a.reach[1]>=a.reach[0],`${e}: invalid machine reach.`);return At(r.has(i.current_agent),`${e}: the active agent does not exist.`),At(i.actor_id===null||r.has(i.actor_id),`${e}: the preceding actor does not exist.`),i}function _d(i,e){if(i.action===null)return"Initial state";let t=(e||i).agents.find(n=>n.id===i.actor_id)||i.agents[0];return t.action_type===1&&(i.action===2||i.action===3)?i.action===2?"Steer left":"Steer right":i.action===6?t.type===2?"Shovel action":t.loaded>0?"Dump / transfer":t.type===1?"Dump":"Dig":HM[i.action]}function po(i,e){if(!i||e.grid.rows!==i.grid.rows||e.grid.cols!==i.grid.cols)return{kind:"snapshot",changed:[],removed:0,placed:0,message:"Initial state"};let t=[],n=0,s=0;for(let d=0;d<e.grid.rows;d++)for(let u=0;u<e.grid.cols;u++){let f=e.maps.action[d][u]-i.maps.action[d][u];f&&(t.push({row:d,col:u,delta:f}),f<0?n-=f:s+=f)}let r=e.agents.find(d=>d.id===e.actor_id),a=i.agents.find(d=>d.id===e.actor_id),o=r&&a?r.loaded-a.loaded:0,c=e.agents.find(d=>d.id!==e.actor_id&&d.loaded>(i.agents.find(u=>u.id===d.id)?.loaded??d.loaded)),l="unchanged",h="No visible state change";return n>0&&o>0?(l="dig",h=`Picked up ${o} soil units \xB7 ${t.length} cells changed`):s>0&&o<0?(l="dump",h=`Placed ${-o} soil units \xB7 ${t.length} cells changed`):o<0&&c?(l="transfer",h=`Transferred soil to machine ${c.id+1}`):t.length?(l="terrain",h=`${t.length} terrain cells changed`):r&&a&&r.position.some((d,u)=>d!==a.position[u])?(l="move",h="Machine moved"):r&&a&&(r.base_yaw!==a.base_yaw||r.cabin_yaw!==a.cabin_yaw||r.wheel_angle!==a.wheel_angle||r.shovel_lifted!==a.shovel_lifted)&&(l="turn",h="Machine configuration changed"),{kind:l,changed:t,removed:n,placed:s,loadDelta:o,recipient:c,message:h}}function sm(i){let e=0,t=0,n=0;for(let s=0;s<i.grid.rows;s++)for(let r=0;r<i.grid.cols;r++){let a=i.maps.action[s][r];e+=Math.max(0,-a),t+=Math.max(0,a),n+=Math.max(0,-i.maps.target[s][r])}return{cut:e,fill:t,target:n,carried:i.agents.reduce((s,r)=>s+r.loaded,0)}}function xd(i,e,t){let n=Math.atan2(Math.sin(e-i),Math.cos(e-i));return i+n*t}function rm(i){if(i===0)return"0.00";let e=Math.abs(i)<.01?i.toPrecision(3):i.toFixed(2);return`${i>0?"+":""}${e}`}var zi={name:"diorama",sand:14069366,dug:[12749400,11366984,9855037,8343860],loose:11037754,strata:[13013084,11433289,13673068,10250821],rock:9340541,grass:[9224530,7646533],grassEdge:6263612,sky:[9225962,13624815,16246732],clods:[11039039,12157001,9723951,12881752]},mo={diorama:zi,paper:{...zi,name:"paper",sand:14208441,dug:[12889485,11441525,9928288,8415310],loose:11242084,strata:[13153428,11705468,13878182,10718574],rock:10196622,clods:[11242084,10255448,12098164,9400400]},studio:{...zi,name:"studio",sand:10849641,dug:[8939851,7953730,6967609,6047281],loose:10124118,strata:[9992538,8677193,10585447,7822659],rock:7301731,clods:[7098165,8084800,6112302,9071694]}},si={uTime:{value:0},uUnit:{value:.27},uTile:{value:.57},uFloor:{value:-1},uMotion:{value:1}},am=`
varying vec3 vTWorld;
varying vec3 vTNormal;
float tHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float tNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
  return mix(mix(tHash(i), tHash(i + vec2(1, 0)), f.x), mix(tHash(i + vec2(0, 1)), tHash(i + vec2(1, 1)), f.x), f.y);
}
float tFbm(vec2 p) { return tNoise(p) * .55 + tNoise(p * 2.13 + 7.1) * .3 + tNoise(p * 4.7 + 3.3) * .15; }
`,om=`
vec4 tWorld = vec4(transformed, 1.);
vec3 tNormal = objectNormal;
#ifdef USE_INSTANCING
tWorld = instanceMatrix * tWorld; tNormal = mat3(instanceMatrix) * tNormal;
#endif
tWorld = modelMatrix * tWorld; vTWorld = tWorld.xyz; vTNormal = normalize(mat3(modelMatrix) * tNormal);
`,ih=i=>new Pe(i),sh=i=>`vec3(${i.r.toFixed(4)}, ${i.g.toFixed(4)}, ${i.b.toFixed(4)})`;function GM(i,e){let[t,n,s,r]=e.strata.map(a=>sh(ih(a)));return`
  {
    float depth = -vTWorld.y / uUnit;
    float along = vTWorld.x * .83 + vTWorld.z * 1.17;
    float wobble = (tNoise(vec2(along * 1.4, depth * .35)) - .5) * .32;
    float band = depth + wobble, index = mod(floor(band), 4.), phase = fract(band);
    vec3 stratum = index < 1. ? ${t} : index < 2. ? ${n} : index < 3. ? ${s} : ${r};
    stratum *= .93 + tNoise(vec2(along * 5.1, vTWorld.y * 9.)) * .12;
    stratum *= mix(.8, 1., smoothstep(0., .1, phase));
    stratum *= 1. - clamp(depth * .025, 0., .28);
    if (vTWorld.y < uFloor) stratum = ${sh(ih(e.rock))} * (.86 + tNoise(vec2(along * 2.3, vTWorld.y * 2.7)) * .2);
    ${i?`if (vTWorld.y > -uTile * .2) stratum = ${sh(ih(e.grassEdge))} * (.92 + tNoise(vec2(along * 3., 1.)) * .14);`:""}
    diffuseColor.rgb = stratum;
  }`}function ki(i,e={},t=zi){let n=new Ke({roughness:1,metalness:0,...e}),[s,r]=t.grass.map(o=>sh(ih(o))),a=i==="island"?`float g = smoothstep(.3, .72, tFbm(vTWorld.xz * .28)); diffuseColor.rgb = mix(${s}, ${r}, g) * (.94 + tNoise(vTWorld.xz * 3.1) * .1);`:i==="pile"?`diffuseColor.rgb *= (.84 + .2 * smoothstep(0., 5., vTWorld.y / uUnit)) * (.92 + tFbm(vTWorld.xz * 2.6) * .16)${t.name==="studio"?" * (.94 + tNoise(vTWorld.xz * 11.) * .12)":""};`:t.name==="studio"?"diffuseColor.rgb *= (.88 + tFbm(vTWorld.xz * 1.15) * .2) * (.95 + tNoise(vTWorld.xz * 9.) * .1);":"diffuseColor.rgb *= .9 + tFbm(vTWorld.xz * 1.15) * .2;";return n.onBeforeCompile=o=>{Object.assign(o.uniforms,si),o.vertexShader=`varying vec3 vTWorld;
varying vec3 vTNormal;
${o.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
${om}`),o.fragmentShader=`uniform float uUnit;
uniform float uTile;
uniform float uFloor;
${am}
${o.fragmentShader}`.replace("#include <color_fragment>",`#include <color_fragment>
      {
        vec3 tn = normalize(vTNormal);
        if (tn.y > .5) { ${a} }
        else if (tn.y > -.5) { ${i==="pile"?a:GM(i==="island",t)} }
      }`)},n.customProgramCacheKey=()=>`terra-earth-${i}-${t.name}`,n}var WM={hatch:0,dots:1,solid:2,cross:3,stripes:4};function rh({color:i,opacity:e,pattern:t="solid",...n}){let s=new Li({color:i,transparent:!0,opacity:e,depthWrite:!1,...n}),r=WM[t];return s.onBeforeCompile=a=>{Object.assign(a.uniforms,si),a.vertexShader=`varying vec3 vTWorld;
varying vec3 vTNormal;
${a.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
vec3 objectNormal = vec3(0., 1., 0.);
${om}`),a.fragmentShader=`uniform float uTile;
uniform float uTime;
uniform float uMotion;
${am}
${a.fragmentShader}`.replace("#include <color_fragment>",`#include <color_fragment>
      {
        vec2 p = vTWorld.xz / uTile;
        float a = 1.;
        ${r===0?"a = mix(.42, 1., step(.5, fract((p.x + p.y) * .7 - uTime * .12 * uMotion)));":""}
        ${r===1?"vec2 q = fract(p * 1.5) - .5; a = mix(.5, 1., 1. - smoothstep(.2, .26, length(q)));":""}
        ${r===3?"a = mix(.35, 1., max(step(.72, fract((p.x + p.y) * .7)), step(.72, fract((p.x - p.y) * .7))));":""}
        ${r===4?"a = mix(.3, 1., step(.62, fract((p.x - p.y) * .55)));":""}
        diffuseColor.a *= a;
      }`)},s.customProgramCacheKey=()=>`terra-zone-${r}`,s}function lm(){let i=document.createElement("canvas");i.width=512,i.height=288;let e=i.getContext("2d"),t=e.createRadialGradient(256,170,20,256,150,330);t.addColorStop(0,"#f2f1ec"),t.addColorStop(.55,"#d9dcdc"),t.addColorStop(1,"#9fa9b0"),e.fillStyle=t,e.fillRect(0,0,512,288);let n=new pi(i);return n.colorSpace=Lt,n}function cm(){let i=document.createElement("canvas");i.width=4,i.height=256;let e=i.getContext("2d"),t=e.createLinearGradient(0,0,0,256),[n,s,r]=zi.sky.map(o=>`#${o.toString(16).padStart(6,"0")}`);t.addColorStop(0,n),t.addColorStop(.58,s),t.addColorStop(1,r),e.fillStyle=t,e.fillRect(0,0,4,256);let a=new pi(i);return a.colorSpace=Lt,a}var XM=[[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[0,2],[0,1]],qM=1.6,YM=(i,e,t)=>typeof i=="function"?i(e,t):i;function ah(i,e,t,n,s){if(e<0||t<0||e>=i.grid.rows||t>=i.grid.cols||i.maps.padding[e][t])return 0;let r=i.maps.action[e][t],a=n?n.maps.action[e][t]:r;return Math.max(0,a+(r-a)*YM(s,e,t))}function ZM(i,e,t,n){if(typeof n!="function")return n;let s=e%2?[(e-1)/2]:[e/2-1,e/2],r=t%2?[(t-1)/2]:[t/2-1,t/2],a=0,o=0;for(let c of s)for(let l of r)c>=0&&l>=0&&c<i.grid.rows&&l<i.grid.cols&&(a+=n(c,l),o++);return o?a/o:1}function $M(i,e,t,n,s){let r=e%2?[(e-1)/2]:[e/2-1,e/2],a=t%2?[(t-1)/2]:[t/2-1,t/2],o=1/0;for(let c of r)for(let l of a)o=Math.min(o,ah(i,c,l,n,s));return o}function Gr(i,e=null,t=1,n=null){let s=i.grid.rows*2+1,r=i.grid.cols*2+1,a=new Float64Array(s*r),o=qM/2;for(let c=0;c<s;c++)for(let l=0;l<r;l++)a[c*r+l]=$M(i,c,l,e,t);for(let c=0;c<s;c++){let l=c*r;for(let h=1;h<r;h++)a[l+h]=Math.min(a[l+h],a[l+h-1]+o);for(let h=r-2;h>=0;h--)a[l+h]=Math.min(a[l+h],a[l+h+1]+o)}for(let c=1;c<s;c++)for(let l=0;l<r;l++){let h=c*r+l;a[h]=Math.min(a[h],a[h-r]+o)}for(let c=s-2;c>=0;c--)for(let l=0;l<r;l++){let h=c*r+l;a[h]=Math.min(a[h],a[h+r]+o)}if(e&&(typeof t=="function"||t>0&&t<1)){let c=n?.start??Gr(i,e,0),l=n?.end??Gr(i);for(let h=0;h<s;h++)for(let d=0;d<r;d++){let u=h*r+d,f=ZM(i,h,d,t);a[u]=Math.min(a[u],c.heights[u]+(l.heights[u]-c.heights[u])*f)}}return{rows:s,cols:r,heights:a}}function JM(i,e=null){let t=[],n=[],s=new Map,r=i.grid.cols*2+1,a=(o,c)=>{let l=o*r+c;return s.has(l)||(s.set(l,t.length),t.push([o,c])),s.get(l)};for(let o=0;o<i.grid.rows;o++)for(let c=0;c<i.grid.cols;c++){if(ah(i,o,c,e,0)<=0&&ah(i,o,c,e,1)<=0)continue;let l=a(o*2+1,c*2+1),h=XM.map(([u,f])=>a(o*2+u,c*2+f)),d=h.flatMap((u,f)=>[l,u,h[(f+1)%h.length]]);n.push({row:o,col:c,center:l,ring:h,triangles:d})}return{nodes:t,cells:n}}var oh=class extends tt{constructor(e,{previous:t=null,unitHeight:n=e.grid.tile_size_m*.48,layerSettings:s={},visibility:r={},palette:a=zi,roughness:o=0}={}){super(),this.frame=e,this.previous=t,this.unitHeight=n,this.endHeights=Gr(e),this.startHeights=t?Gr(e,t,0):this.endHeights,this.endpoints={start:this.startHeights,end:this.endHeights},this.progress=1,this.topology=JM(e,t),this.activeCells=[];let{rows:c,cols:l,tile_size_m:h}=e.grid,d=new Float32Array(this.topology.nodes.length*3),u=new Float32Array(d.length),f=new Pe;for(let p=0;p<this.topology.nodes.length;p++){let[m,M]=this.topology.nodes[p],S=(T,b)=>b?0:((Math.sin(m*12.9898+M*78.233+T)*43758.5453%1+1)%1-.5)*2*o*h;d[p*3]=(M/2-l/2)*h+S(1.7,M===0||M===l*2),d[p*3+2]=(m/2-c/2)*h+S(5.3,m===0||m===c*2);let y=(m*37+M*61+m*M*7)%29/29;f.set(a.loose).multiplyScalar(.95+y*.1),f.toArray(u,p*3)}this.positions=new Ut(d,3).setUsage(xi);let g=new ut;g.setAttribute("position",this.positions),g.setAttribute("color",new Ut(u,3)),this.surface=new et(g,ki("pile",{vertexColors:!0,flatShading:!0,polygonOffset:!0,polygonOffsetFactor:-1,polygonOffsetUnits:-2},a)),this.surface.name="connected-soil-piles",this.surface.castShadow=!0,this.surface.receiveShadow=!0,this.surface.userData.soilPiles=this,this.add(this.surface),this.layerSettings=s,this.overlays={};for(let[p,m]of Object.entries(s)){let M=new ut;M.setAttribute("position",this.positions);let S=rh({color:m.color,opacity:m.opacity*.7,pattern:m.pattern,polygonOffset:!0,polygonOffsetFactor:-2}),y=new et(M,S);y.position.y=h*(.008+Object.keys(this.overlays).length*.003),y.renderOrder=3+Object.keys(this.overlays).length,y.visible=!!r[p],y.frustumCulled=!1,y.userData.skipAO=!0,this.overlays[p]=y,this.add(y)}let _=new ut;_.setAttribute("position",this.positions),this.gridLines=new ns(_,new Ni({color:7426351,transparent:!0,opacity:.25,depthWrite:!1})),this.gridLines.position.y=h*.022,this.gridLines.renderOrder=12,this.gridLines.visible=!!r.grid,this.gridLines.frustumCulled=!1,this.add(this.gridLines),this.update(1)}nodeHeight(e,t){return(this.heights.heights[e*this.heights.cols+t]??0)*this.unitHeight}endpointHeight(e,t,n=!1){let s=n?this.startHeights:this.endHeights;return(s.heights[(e*2+1)*s.cols+t*2+1]??0)*this.unitHeight}update(e=1){this.progress=e,this.heights=typeof e=="function"?Gr(this.frame,this.previous,e,this.endpoints):e<=0?this.startHeights:e>=1?this.endHeights:Gr(this.frame,this.previous,e,this.endpoints);let t=this.positions.array;for(let r=0;r<this.topology.nodes.length;r++){let[a,o]=this.topology.nodes[r];t[r*3+1]=this.nodeHeight(a,o)}this.positions.needsUpdate=!0;let n=this.topology.cells.filter(r=>ah(this.frame,r.row,r.col,this.previous,e)>0),s=n.length!==this.activeCells.length||n.some((r,a)=>r!==this.activeCells[a]);if(this.activeCells=n,this.surface.visible=n.length>0,s||!this.surface.geometry.index){this.surface.geometry.setIndex(n.flatMap(r=>r.triangles));for(let[r,a]of Object.entries(this.overlays)){let o=this.layerSettings[r],c=this.frame.maps[o.map];a.geometry.setIndex(n.filter(l=>c!=null&&o.test(c[l.row][l.col])).flatMap(l=>l.triangles))}this.gridLines.geometry.setIndex(n.flatMap(r=>r.ring.flatMap((a,o)=>[a,r.ring[(o+1)%r.ring.length]])))}this.surface.geometry.computeVertexNormals(),this.surface.geometry.computeBoundingSphere()}setLayer(e,t){e==="grid"?this.gridLines.visible=t:this.overlays[e]&&(this.overlays[e].visible=t&&this.frame.maps[this.layerSettings[e].map]!=null)}cellForHit(e){let t=this.activeCells[Math.floor(e.faceIndex/8)];return t?{row:t.row,col:t.col}:null}dispose(){this.traverse(e=>{e.geometry?.dispose(),e.material&&e.material.dispose()}),this.clear()}};function vd(i,e=0){let t=(i^Math.imul(e+1,2654435761))>>>0;return t=Math.imul(t^t>>>16,2246822507),t=Math.imul(t^t>>>13,3266489909),(t^t>>>16)>>>0}var Gs=(i,e)=>vd(i,e)/4294967295,hm=(i,e,t)=>Math.max(e,Math.min(t,i));function um(i){if(!Array.isArray(i)||!i.length||!Array.isArray(i[0])||!i[0].length)throw new Error("Obstacle padding must be a nonempty rectangular array.");let e=i.length,t=i[0].length;if(e>128||t>128||i.some(n=>!Array.isArray(n)||n.length!==t||n.some(s=>![0,1,!1,!0].includes(s))))throw new Error("Obstacle padding must contain aligned 0/1 cells, at most 128 \xD7 128.");return{rows:e,cols:t}}function jM(i,e,t){let n=new Uint8Array(e*t),s=[];for(let r=0;r<e;r++)for(let a=0;a<t;a++){let o=r*t+a;if(!i[r][a]||n[o])continue;let c=[[r,a]];n[o]=1;let l=r,h=r,d=a,u=a;for(let g=0;g<c.length;g++){let[_,p]=c[g];l=Math.min(l,_),h=Math.max(h,_),d=Math.min(d,p),u=Math.max(u,p);for(let[m,M]of[[_-1,p],[_,p-1],[_,p+1],[_+1,p]]){if(m<0||M<0||m>=e||M>=t)continue;let S=m*t+M;i[m][M]&&!n[S]&&(n[S]=1,c.push([m,M]))}}let f=2166136261;for(let[g,_]of[...c].sort((p,m)=>p[0]-m[0]||p[1]-m[1]))f=Math.imul(f^g*131+_,16777619)>>>0;s.push({cells:c,minRow:l,maxRow:h,minCol:d,maxCol:u,seed:f})}return s}function KM(i,e,t){let n=new Uint16Array(t),s=null,r=0;for(let a=0;a<e;a++){let o=[];for(let c=0;c<t;c++)n[c]=i[a*t+c]?n[c]+1:0;for(let c=0;c<=t;c++){let l=c<t?n[c]:0,h=c;for(;o.length&&o[o.length-1].height>l;){let d=o.pop(),u=d.height*(c-d.start),f={row:a-d.height+1,col:d.start,rows:d.height,cols:c-d.start};(u>r||u===r&&(f.row<s.row||f.row===s.row&&f.col<s.col))&&(s=f,r=u),h=d.start}l&&(!o.length||o[o.length-1].height<l)&&o.push({start:h,height:l})}}return s}function QM(i,e,t,n){let s={...i};function r(a){if(a.row<0||a.col<0||a.row+a.rows>t||a.col+a.cols>n)return!1;for(let o=a.row;o<a.row+a.rows;o++)for(let c=a.col;c<a.col+a.cols;c++)if(!e[o*n+c])return!1;return!0}for(;;){let a=s,o=[{...a,row:a.row-1,rows:a.rows+1},{...a,col:a.col-1,cols:a.cols+1},{...a,rows:a.rows+1},{...a,cols:a.cols+1}].filter(r).sort((c,l)=>l.rows*l.cols-c.rows*c.cols||c.row-l.row||c.col-l.col);if(!o.length)return s;s=o[0]}}function eS(i){let{rows:e,cols:t}=um(i),n=[],s=jM(i,e,t);for(let[r,a]of s.entries()){let o=a.maxRow-a.minRow+1,c=a.maxCol-a.minCol+1,l=new Uint8Array(o*c);for(let[f,g]of a.cells)l[(f-a.minRow)*c+g-a.minCol]=1;let h=l.slice(),d=a.cells.length,u=a.cells.length/(o*c);for(;d;){let f=QM(KM(h,o,c),l,o,c);for(let y=f.row;y<f.row+f.rows;y++)for(let T=f.col;T<f.col+f.cols;T++){let b=y*c+T;h[b]&&(h[b]=0,d--)}let g={row:f.row+a.minRow,col:f.col+a.minCol,rows:f.rows,cols:f.cols},_=vd(a.seed,g.row*131+g.col),p=Math.min(f.rows,f.cols),m=Math.max(f.rows,f.cols),M=(a.minRow+a.minCol+o+c)%3;if(u>=.9&&p>=3&&m>=6&&f.rows*f.cols>=24&&M!==0){let y=f.cols>=f.rows,T=Math.min(3,Math.floor(p/3),Math.max(1,Math.round(p/(m*.37))));for(let b=0;b<T;b++){let P=Math.floor(p*b/T),x=Math.floor(p*(b+1)/T),E={...g};y?(E.row+=P,E.rows=x-P):(E.col+=P,E.cols=x-P),n.push({...E,kind:"container",component:r,seed:vd(_,b)})}}else n.push({...g,kind:"boulder",component:r,seed:_})}}return n}function tS(i,e,t,n,s=!0){let a=[],o=[],c=[],l=new Pe().setHex([10131340,10721928,9278606][n%3]),h=new Pe(8824919),d=new R,u=new R,f=new R;for(let p=0;p<3;p++){let m=[];for(let M=0;M<9;M++){let S=(M+Gs(n,M)*.13)*Math.PI*2/9,y=p===2?.44+Gs(n,M+20)*.22:.85+Gs(n,M+p*9+40)*.14;m.push(new R(Math.cos(S)*i/2*y,p===0?0:t*(p===1?.38+Gs(n,M+70)*.1:.76+Gs(n,M+90)*.18),Math.sin(S)*e/2*y))}a.push(m)}function g(p,m,M){f.crossVectors(d.subVectors(m,p),u.subVectors(M,p)).normalize();let S=Math.abs(f.y),y=l.clone().multiplyScalar(.84+Gs(n,o.length)*.2+S*.12);s&&S>.72&&Gs(n,o.length+7)<.45&&y.lerp(h,.55);for(let T of[p,m,M])o.push(T.x,T.y,T.z),c.push(y.r,y.g,y.b)}for(let p=0;p<9;p++){let m=(p+1)%9;for(let M=0;M<2;M++)g(a[M][p],a[M+1][p],a[M+1][m]),g(a[M][p],a[M+1][m],a[M][m]);g(new R(0,0,0),a[0][p],a[0][m]),g(a[2][p],new R(0,t,0),a[2][m])}let _=new ut;return _.setAttribute("position",new rt(o,3)),_.setAttribute("color",new rt(c,3)),_.computeVertexNormals(),_.computeBoundingBox(),_}function Wr(i,e,t,n,s,r,a,o,c){let l=new et(e,t);return l.position.set(n,s,r),l.scale.set(a,o,c),l.castShadow=!0,l.receiveShadow=!0,i.add(l),l}function nS(i,e,t,n,s){let r=Math.max(e.rows,e.cols)*t,a=Math.min(e.rows,e.cols)*t,o=r*.94,c=Math.min(a*.88,o*.42),l=hm(c*.94,t*.7,t*3.6),h=new tt;h.rotation.y=e.rows>e.cols?Math.PI/2:0,i.add(h);let d=n+t*.12;s.box||(s.box=new Bt(1,1,1)),s.metal||(s.metal=new Ke({color:7831675,roughness:.64,metalness:.25})),s.foundation||(s.foundation=new Ke({color:9343364,roughness:1}));let u=new Ke({color:(s.paper?[9277839,8357252,10001045]:[10772291,5340795,6455185])[e.seed%3],roughness:.77,metalness:.15}),f=u.clone();f.color.multiplyScalar(1.12),Wr(h,s.box,s.foundation,0,d/2,0,o+t*.06,d,c+t*.09),Wr(h,s.box,u,0,d+l/2,0,o,l,c),Wr(h,s.box,f,0,d+l+t*.025,0,o,t*.05,c);let g=Math.max(4,Math.round(o/(t*.45))),_=new jt(s.box,f,g*2),p=new ft;_.castShadow=!0,_.receiveShadow=!0;for(let m=0;m<2;m++)for(let M=0;M<g;M++)p.position.set(o*(-.46+.92*M/(g-1)),d+l/2,(m?1:-1)*(c/2+t*.013)),p.scale.set(t*.065,l*.94,t*.033),p.updateMatrix(),_.setMatrixAt(m*g+M,p.matrix);h.add(_);for(let m of[-1,1]){Wr(h,s.box,f,o/2+t*.02,d+l*.5,m*c*.237,t*.04,l*.88,c*.45),Wr(h,s.box,s.metal,o/2+t*.047,d+l*.5,m*c*.17,t*.028,l*.79,t*.038);for(let M of[-1,1])Wr(h,s.box,s.metal,M*(o/2-t*.045),d+l/2,m*(c/2-t*.036),t*.09,l,t*.075)}}function dm(i,e={}){typeof e=="number"&&(e={tile:e});let t=e.tile??i.grid.tile_size_m,n=e.unitHeight??t*.48;if(!Number.isFinite(t)||t<=0||!Number.isFinite(n)||n<=0)throw new Error("Obstacle display scale must be finite and positive.");let{rows:s,cols:r}=um(i.maps.padding);if(s!==i.grid.rows||r!==i.grid.cols||!Array.isArray(i.maps.action)||i.maps.action.length!==s||i.maps.action.some(l=>!Array.isArray(l)||l.length!==r||l.some(h=>!Number.isFinite(h))))throw new Error("Obstacle terrain must match the frame grid and contain finite heights.");let a=eS(i.maps.padding),o=new tt,c={paper:e.style!=="diorama"};o.name="Terra obstacle props",o.userData.footprints=a;for(let l of a){let h=new tt;h.name=`${l.kind}-${l.row}-${l.col}`,h.userData.footprint={...l};let d=1/0,u=-1/0;for(let f=l.row;f<l.row+l.rows;f++)for(let g=l.col;g<l.col+l.cols;g++){let _=i.maps.action[f][g]*n;d=Math.min(d,_),u=Math.max(u,_)}if(h.position.set((l.col+l.cols/2-r/2)*t,d+t*.008,(l.row+l.rows/2-s/2)*t),l.kind==="container")nS(h,l,t,u-d,c);else{c.stone||(c.stone=new Ke({vertexColors:!0,roughness:1,flatShading:!0}));let f=hm(Math.min(l.rows,l.cols)*t*.62,t*.52,t*3.4)+u-d,g=new et(tS(l.cols*t*.96,l.rows*t*.96,f,l.seed,!c.paper),c.stone);g.castShadow=!0,g.receiveShadow=!0,h.add(g)}o.add(h)}return o}function mm(i){let e=i>>>0||1;return()=>(e=Math.imul(e^e>>>15,739982445)+1831565813>>>0,e^=e>>>13,(e>>>0)/4294967295)}function gm(i,e,t,n,s=!1){let r=Math.min(n,e*.98,t*.98),a=[],o=[[e-r,t-r,0],[-e+r,t-r,Math.PI/2],[-e+r,-t+r,Math.PI],[e-r,-t+r,Math.PI*1.5]];for(let[c,l,h]of o)for(let d=0;d<=6;d++){let u=h+d/6*Math.PI/2;a.push(new Z(c+Math.cos(u)*r,l+Math.sin(u)*r))}return s&&a.reverse(),i?(i.setFromPoints(a),i):a}function iS(i,e,t,n,s){let r=mm(s),a=gm(null,i,e,t),o=[a],c=[[.9,.34],[.66,.72],[.26,1]];for(let[m,M]of c)o.push(a.map(S=>{let y=.9+r()*.2;return new R(S.x*m*y,-n*M*(.85+r()*.3),S.y*m*y)}));o[0]=a.map(m=>new R(m.x,0,m.y));let l=[],h=[],d=new Pe,u=[9206374,8219740,9864302,7299410],f=(m,M,S)=>{d.setHex(u[Math.floor(r()*u.length)]);for(let y of[m,M,S])l.push(y.x,y.y,y.z),h.push(d.r,d.g,d.b)};for(let m=0;m<o.length-1;m++)for(let M=0;M<a.length;M++){let S=(M+1)%a.length,y=o[m][M],T=o[m][S],b=o[m+1][M],P=o[m+1][S];f(y,b,T),f(T,b,P)}let g=new R(0,-n*1.25,0),_=o[o.length-1];for(let m=0;m<_.length;m++)f(_[m],g,_[(m+1)%_.length]);let p=new ut;return p.setAttribute("position",new rt(l,3)),p.setAttribute("color",new rt(h,3)),p.computeVertexNormals(),p}function sS(i,e=8){if(typeof document>"u")return null;let t=document.createElement("canvas");t.width=64,t.height=8;let n=t.getContext("2d");for(let r=0;r<e;r++){n.fillStyle=i[r%i.length],n.beginPath();let a=64/e;n.moveTo(r*a,0),n.lineTo(r*a+a,0),n.lineTo(r*a+a-4,8),n.lineTo(r*a-4,8),n.fill()}let s=new pi(t);return s.colorSpace=Lt,s.wrapS=kn,s.anisotropy=4,s}function fm(i){let e=new Ke({roughness:.9,flatShading:!0,...i});return e.onBeforeCompile=t=>{t.uniforms.uTime=si.uTime,t.vertexShader=`uniform float uTime;
${t.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
      #ifdef USE_INSTANCING
      float swayPhase = instanceMatrix[3].x * .37 + instanceMatrix[3].z * .23;
      float swayHeight = max(0., position.y + .5);
      transformed.x += sin(uTime * 1.3 + swayPhase) * .05 * swayHeight;
      transformed.z += cos(uTime * 1.1 + swayPhase) * .035 * swayHeight;
      #endif`)},e.customProgramCacheKey=()=>"terra-sway",e}var Hi=class{constructor(e,t,n,s){this.mesh=new jt(t,n,s),this.mesh.count=0,this.mesh.castShadow=!0,this.mesh.receiveShadow=!0,e.add(this.mesh),this.dummy=new ft,this.color=new Pe}add(e,t,n,s,r,a,o,c=0,l=0){if(this.mesh.count>=this.mesh.instanceMatrix.count)return;let h=this.dummy;h.position.set(e,t,n),h.rotation.set(l,c,l*.6),h.scale.set(s,r,a),h.updateMatrix(),this.mesh.setMatrixAt(this.mesh.count,h.matrix),this.mesh.setColorAt(this.mesh.count,this.color.setHex(o)),this.mesh.count++}finish(){this.mesh.instanceMatrix.needsUpdate=!0,this.mesh.instanceColor&&(this.mesh.instanceColor.needsUpdate=!0),this.mesh.computeBoundingSphere()}};function Kt(i,e,t,n,s,r,a,o,c){let l=new et(c,e);return l.position.set(t,n,s),l.scale.set(r,a,o),l.castShadow=!0,l.receiveShadow=!0,i.add(l),l}function rS(i,e,t,n,s){let r=new tt;r.position.set(e,0,t),r.rotation.y=n,i.add(r);let{cube:a}=s,o=s.materials;Kt(r,o.concrete,0,.08,0,4.2,.16,2.5,a),Kt(r,o.office,0,1.4,0,4,2.5,2.3,a),Kt(r,o.trim,0,2.7,0,4.15,.14,2.45,a),Kt(r,o.trim,0,.22,0,4.1,.14,2.4,a);for(let l of[-1.2,.15])Kt(r,o.window,l,1.6,1.16,1,.75,.04,a),Kt(r,o.trim,l,1.18,1.19,1.1,.07,.08,a);Kt(r,o.door,1.35,1.15,1.16,.8,1.9,.05,a),Kt(r,o.concrete,1.35,.15,1.55,1.05,.3,.6,a),Kt(r,o.window,-2.005,1.6,0,.04,.7,1,a),Kt(r,o.metal,-1.2,2.95,-.4,.8,.36,.6,a),Kt(r,o.sign,.15,3.08,1.05,1.7,.46,.06,a);let c=new tt;return c.position.set(1.3,0,-1.95),r.add(c),Kt(c,o.loo,0,1.12,0,1.05,2.24,1.05,a),Kt(c,o.looRoof,0,2.3,0,1.12,.12,1.12,a),Kt(c,o.trim,0,1.12,.53,.7,1.8,.03,a),r}function aS(i,e,t,n,s){let r=new tt;r.position.set(e,0,t),r.rotation.y=n,i.add(r);let a=s.pipe;for(let[o,c]of[[-.55,.32],[0,.32],[.55,.32],[-.27,.8],[.27,.8],[0,1.27]]){let l=new et(a,s.materials.pipe);l.rotation.x=Math.PI/2,l.position.set(o,c,0),l.scale.set(.3,3.2,.3),l.castShadow=!0,l.receiveShadow=!0,r.add(l);let h=new et(s.ring,s.materials.pipeEnd);h.position.set(o,c,1.61),h.scale.setScalar(.3),r.add(h)}return Kt(r,s.materials.wood,0,.03,-1.1,1.8,.06,.2,s.cube),Kt(r,s.materials.wood,0,.03,1.1,1.8,.06,.2,s.cube),r}function oS(i,e,t,n,s,r){let a=new tt;a.position.set(e,0,t),a.rotation.y=n,i.add(a),Kt(a,s.materials.wood,0,.07,0,1.2,.14,1,s.cube);let o=2+Math.floor(r()*3);for(let c=0;c<o;c++){let l=Kt(a,s.materials.bag,(c%2-.5)*.52,.28+Math.floor(c/2)*.26,0,.5,.24,.86,s.bagGeometry);l.rotation.y=(r()-.5)*.2}return a}function pm(i,e=mo.paper,t=!1){let{rows:n,cols:s,tile_size_m:r}=i.grid,a=Math.max(n,s)*r,o=new tt;o.name="Terra plinth";let c=new Bt(s*r,1,n*r);c.translate(0,-.5,0);let l=new et(c,ki("soil",{polygonOffset:!0,polygonOffsetFactor:1,polygonOffsetUnits:2},e));l.name="plinth",l.receiveShadow=!0,l.castShadow=t,o.add(l);let h=null;return t&&(h=new et(new Hn(a*12,a*12),new La({color:2893344,opacity:.28})),h.rotation.x=-Math.PI/2,h.receiveShadow=!0,h.name="studio-floor",o.add(h)),o.userData.extent={hx:s*r/2,hz:n*r/2},o.setFloor=d=>{let u=Math.min(-Math.max(a*(t?.085:.06),1.8),d-r*.8);l.position.y=d,l.scale.y=d-u,si.uFloor.value=d,h&&(h.position.y=u-.002)},o.update=()=>{},o.dispose=()=>{c.dispose(),l.material.dispose(),h&&(h.geometry.dispose(),h.material.dispose()),o.removeFromParent()},o}function _m(i,{style:e="diorama"}={}){if(e==="paper")return pm(i);if(e==="studio")return pm(i,mo.studio,!0);let{rows:t,cols:n,tile_size_m:s}=i.grid,r=n*s/2,a=t*s/2,o=Math.max(t,n)*s,c=Vt.clamp(o*.2,6,22),l=new tt;l.name="Terra surroundings";let h=mm(t*7919+n*104729+Math.round(s*1e3)),d=r+c,u=a+c,f=c*1.1,g={cube:new Bt(1,1,1),pipe:new Qn(1,1,1,10,1,!0),ring:new Ia(.72,1,10),bagGeometry:new Bt(1,1,1),materials:{concrete:new Ke({color:12170925,roughness:1}),office:new Ke({color:15986918,roughness:.8}),trim:new Ke({color:4157338,roughness:.7}),window:new Ke({color:10475238,roughness:.15,metalness:.1,emissive:1915460,emissiveIntensity:.3}),door:new Ke({color:3102072,roughness:.7}),metal:new Ke({color:13225680,roughness:.5}),sign:new Ke({color:15905329,roughness:.6}),loo:new Ke({color:3842264,roughness:.6}),looRoof:new Ke({color:15397621,roughness:.6}),pipe:new Ke({color:15040058,roughness:.7,side:Sn}),pipeEnd:new Ke({color:12083499,roughness:.8,side:Sn}),wood:new Ke({color:12159573,roughness:1}),bag:new Ke({color:15327433,roughness:1})}},_=gm(new gi,d,u,f),p=new mi;p.moveTo(-r,-a),p.lineTo(-r,a),p.lineTo(r,a),p.lineTo(r,-a),p.closePath(),_.holes.push(p);let m=new Fi(_,{depth:1,bevelEnabled:!1,curveSegments:6});m.rotateX(Math.PI/2);let M=new et(m,ki("island"));M.receiveShadow=!0,M.castShadow=!1,M.name="island-turf",l.add(M);let S=new et(iS(d,u,f,Math.max(o*.16,4),t*31+n),new Ke({vertexColors:!0,flatShading:!0,roughness:1,side:Sn}));S.name="island-underside",l.add(S);let y=Math.min(4.2,a*.5),T=new et(new Hn(1,1),ki("soil",{color:13482134}));T.rotation.x=-Math.PI/2,T.scale.set(c,y,1),T.position.set(r+c/2+.01,.012,0),T.name="site-road",T.receiveShadow=!0,l.add(T);let b=y/2+.3,P=new Hi(l,g.cube,new Ke({color:15657696,roughness:.7}),400),x=new jt(g.cube,new Ke({map:sS(["#e8573a","#f7f2e8"]),color:typeof document>"u"?15226682:16777215,roughness:.7}),800);x.count=0,x.castShadow=!0,l.add(x);let E=new ft,C=.18,I=[[[-r-C,-a-C],[r+C,-a-C]],[[-r-C,a+C],[r+C,a+C]],[[-r-C,-a-C],[-r-C,a+C]],[[r+C,-a-C],[r+C,-b]],[[r+C,b],[r+C,a+C]]];for(let[[G,V],[se,ce]]of I){let fe=Math.hypot(se-G,ce-V),me=Math.max(1,Math.round(fe/2.4)),D=Math.atan2(-(ce-V),se-G);for(let Me=0;Me<=me;Me++){let Ve=Me/me;P.add(G+(se-G)*Ve,.45,V+(ce-V)*Ve,.1,.9,.1,Me%2?15657696:15226682)}for(let Me=0;Me<me;Me++)for(let Ve of[.42,.78]){let A=(Me+.5)/me;E.position.set(G+(se-G)*A,Ve,V+(ce-V)*A),E.rotation.set(0,D,0),E.scale.set(fe/me,.09,.02),E.updateMatrix(),x.setMatrixAt(x.count++,E.matrix)}}P.finish(),x.instanceMatrix.needsUpdate=!0,x.computeBoundingSphere();let L=new Er(.2,.6,8);L.translate(0,.3,0);let X=new Hi(l,L,new Ke({color:16777215,roughness:.6,flatShading:!0}),40),q=new Hi(l,new Qn(.115,.145,.1,8),new Ke({color:16777215,roughness:.4}),40);for(let G=0;G<Math.floor(c/2.2);G++)for(let V of[-1,1]){let se=r+1+G*2.2,ce=V*(y/2+.35);X.add(se,0,ce,1,1,1,15953706),q.add(se,.33,ce,1,1,1,16250090)}X.finish(),q.finish();let F=[],Y=(G,V,se)=>{if(Math.abs(G)<r+1.2+se&&Math.abs(V)<a+1.2+se)return!1;let ce=Math.max(0,Math.abs(G)-(d-f)),fe=Math.max(0,Math.abs(V)-(u-f));return Math.hypot(ce,fe)>f-se-.5||G>r&&Math.abs(V)<y/2+se+.6?!1:F.every(([me,D,Me])=>Math.hypot(G-me,V-D)>se+Me)},W=-(y/2+3.2),ie=r+Math.min(c*.55,6);c>=6&&Y(ie,W,2.3)&&Y(ie+1.3,W-1.95,.8)&&(rS(l,ie,W,0,g),F.push([ie,W,2.8],[ie+1.3,W-1.95,.9]));let ne=r+Math.min(c*.6,6.5),ge=y/2+3;c>=6&&Y(ne,ge,1.7)&&(aS(l,ne,ge,Math.PI/2+.1,g),F.push([ne,ge,2]));for(let G=0;G<3;G++){let V=-r-c*(.35+h()*.3),se=(h()-.5)*a*1.4;Y(V,se,.9)&&(oS(l,V,se,h()*Math.PI,g,h),F.push([V,se,.9]))}let ue=new Qn(.5,.7,1,6);ue.translate(0,.5,0);let xe=new Er(1,1,7);xe.translate(0,.5,0);let Ne=new ei(1,0),it=new Hi(l,ue,new Ke({color:16777215,roughness:1,flatShading:!0}),400),Xe=new Hi(l,xe,fm({color:16777215}),900),j=new Hi(l,Ne,fm({color:16777215}),700),he=new Hi(l,new Sa(1,0),new Ke({color:16777215,roughness:1,flatShading:!0}),200),le=[5214042,6069343,4620114],Te=[8238678,9224541,6989903,10930522],Fe=[15905628,15306091],ke=4*(d*u-r*a),oe=Math.min(2600,Math.round(ke/2.2));for(let G=0;G<oe;G++){let V=h(),se=Math.floor(h()*4),ce=Math.pow(h(),.8),fe,me;se<2?(fe=(V*2-1)*d,me=(se?1:-1)*(a+1.8+ce*(c-1.8))):(me=(V*2-1)*u,fe=(se===3?1:-1)*(r+1.8+ce*(c-1.8)));let D=h(),Me=.62+h()*.45,Ve=D<.45?1.1*Me:D<.8?1.3*Me:.7*Me,A=Math.min(Math.abs(Math.abs(fe)-r),Math.abs(Math.abs(me)-a));if(h()>.25+Math.min(1,A/c)*.9||!Y(fe,me,Ve))continue;F.push([fe,me,Ve]);let v=h()*Math.PI*2,U=(h()-.5)*.08;if(D<.45){let B=(3.4+h()*1.8)*Me;it.add(fe,0,me,.18*Me,B*.3,.18*Me,9067835,v);for(let k=0;k<3;k++)Xe.add(fe,B*(.22+k*.22),me,(1.25-k*.3)*Me,B*.42,(1.25-k*.3)*Me,le[(G+k)%3],v+k,U)}else if(D<.8){let B=(2.2+h()*1.4)*Me;it.add(fe,0,me,.16*Me,B*.55,.16*Me,9725247,v),j.add(fe,B*.75,me,1.25*Me,1.05*Me,1.2*Me,h()<.08?Fe[G%2]:Te[G%4],v,U),h()<.6&&j.add(fe+.45*Me,B*.98,me-.2*Me,.8*Me,.7*Me,.8*Me,Te[(G+1)%4],v+1)}else D<.93?j.add(fe,.35*Me,me,.7*Me,.5*Me,.7*Me,Te[(G+2)%4],v):he.add(fe,.12*Me,me,.55*Me,.38*Me,.5*Me,[10130828,9078399,10985879][G%3],v,U*3)}for(let G of[it,Xe,j,he])G.finish();let ee=new tt,O=new Ke({color:16777215,roughness:1,flatShading:!0,emissive:16777215,emissiveIntensity:.25}),H=new ei(1,1),Q=Math.hypot(d,u);for(let G=0;G<6;G++){let V=new tt,se=G/6*Math.PI*2+h()*.6,ce=Q*(1.25+h()*.45),fe=o*(.035+h()*.025);V.position.set(Math.cos(se)*ce,o*(.12+h()*.22),Math.sin(se)*ce);for(let me=0;me<5;me++){let D=new et(H,O);D.position.set((me-2)*fe*.9,(1-Math.abs(me-2)*.45)*fe*.35,(h()-.5)*fe*.7),D.scale.setScalar(fe*(1.1-Math.abs(me-2)*.22)),V.add(D)}V.userData={angle:se,distance:ce,height:V.position.y,speed:.006+h()*.006},ee.add(V)}return l.add(ee),l.userData.extent={hx:d,hz:u},l.setFloor=G=>{let V=Math.min(-Math.max(o*.07,2.2),G-s*.8);M.scale.y=-V,S.position.y=V,si.uFloor.value=G},l.update=G=>{for(let V of ee.children){let{angle:se,distance:ce,height:fe,speed:me}=V.userData,D=se+G*me;V.position.set(Math.cos(D)*ce,fe+Math.sin(G*.3+se*3)*o*.006,Math.sin(D)*ce)}},l.dispose=()=>{let G=new Set,V=new Set;l.traverse(se=>{se.geometry&&G.add(se.geometry),se.material&&V.add(se.material)});for(let se of G)se.dispose();for(let se of V)se.map?.dispose(),se.dispose();l.removeFromParent()},l}var xm=9.81,lS=[11039039,12157001,9723951,12881752],yd=900,lh=240,vm=new R(0,1,0),cS={vertexShader:`
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
    }`},ch=class extends tt{constructor({groundHeight:e=()=>0}={}){super(),this.name="Terra effects",this.groundHeight=e,this.dummy=new ft,this.color=new Pe;let t=new ei(1,0);this.clodMesh=new jt(t,new Ke({roughness:1,flatShading:!0}),yd),this.clodMesh.castShadow=!0,this.clodMesh.frustumCulled=!1,this.clodMesh.count=0,this.add(this.clodMesh),this.puffMesh=new jt(new ei(1,1),new Ke({roughness:1,flatShading:!0}),260),this.puffMesh.frustumCulled=!1,this.puffMesh.count=0,this.add(this.puffMesh);for(let s of[this.clodMesh,this.puffMesh])s.setColorAt(0,this.color.setHex(16777215)),s.userData.skipAO=!0;let n=new ut;n.setAttribute("position",new Ut(new Float32Array(lh*3),3).setUsage(xi)),n.setAttribute("size",new Ut(new Float32Array(lh),1).setUsage(xi)),n.setAttribute("alpha",new Ut(new Float32Array(lh),1).setUsage(xi)),this.dustMaterial=new bt({...cS,uniforms:{color:{value:new Pe(13219740)},scale:{value:600}},transparent:!0,depthWrite:!1}),this.dustPoints=new xa(n,this.dustMaterial),this.dustPoints.frustumCulled=!1,this.dustPoints.renderOrder=40,this.dustPoints.userData.skipAO=!0,this.add(this.dustPoints),this.clods=[],this.puffs=[],this.dusts=[],this.puffsEnabled=!0,this.dustEnabled=!1,this.palette=lS}setViewport(e,t){this.dustMaterial.uniforms.scale.value=e/(2*Math.tan(Vt.degToRad(t)/2))}dust(e,{count:t=6,size:n=.6,spread:s=.3,rise:r=.35,life:a=1.6,opacity:o=.22,drift:c=null}={}){if(this.dustEnabled)for(let l=0;l<t&&this.dusts.length<lh;l++){let h=new R((Math.random()-.5)*s*2,Math.random()*s*.4,(Math.random()-.5)*s*2),d=h.clone().multiplyScalar(.8).add(vm.clone().multiplyScalar(r*(.5+Math.random()*.7)));c&&d.add(c),this.dusts.push({position:e.clone().add(h),velocity:d,size:n*(.6+Math.random()*.7),age:-Math.random()*.15,life:a*(.75+Math.random()*.5),opacity:o})}}throwClods(e,t,{count:n=10,flight:s=.45,spread:r=.3,size:a=.09,settle:o=!0,jitter:c=.08}={}){for(let l=0;l<n&&this.clods.length<yd;l++){let h=e.clone().add(new R((Math.random()-.5)*c*2,(Math.random()-.5)*c,(Math.random()-.5)*c*2)),d=t.clone().add(new R((Math.random()-.5)*r*2,0,(Math.random()-.5)*r*2)),u=s*(.8+Math.random()*.4),f=Math.random()*s*.6,g=d.sub(h).multiplyScalar(1/u);g.y+=.5*xm*u,this.clods.push({position:h,velocity:g,spin:new R(Math.random()*9,Math.random()*9,Math.random()*9),rotation:new Dn(Math.random()*6,Math.random()*6,0),size:a*(.6+Math.random()*.8),age:-f,life:u+(o?1.4:0),arrive:u,settle:o,bounced:!1,color:this.palette[Math.floor(Math.random()*this.palette.length)]})}}burst(e,{count:t=12,speed:n=2.2,size:s=.07}={}){for(let r=0;r<t&&this.clods.length<yd;r++){let a=Math.random()*Math.PI*2,o=.55+Math.random()*.5,c=new R(Math.cos(a)*(1-o),o*1.4,Math.sin(a)*(1-o)).multiplyScalar(n*(.6+Math.random()*.6));this.clods.push({position:e.clone(),velocity:c,spin:new R(Math.random()*12,Math.random()*12,0),rotation:new Dn,size:s*(.6+Math.random()*.8),age:-Math.random()*.08,life:2.2,arrive:1/0,settle:!0,bounced:!1,color:this.palette[Math.floor(Math.random()*this.palette.length)]})}}puff(e,{count:t=6,size:n=.25,spread:s=.35,rise:r=.6,life:a=.9,color:o=15326402,drift:c=null}={}){if(this.puffsEnabled)for(let l=0;l<t&&this.puffs.length<260;l++){let h=new R((Math.random()-.5)*s*2,Math.random()*s*.5,(Math.random()-.5)*s*2),d=h.clone().multiplyScalar(1.4).add(vm.clone().multiplyScalar(r*(.6+Math.random()*.6)));c&&d.add(c),this.puffs.push({position:e.clone().add(h),velocity:d,size:n*(.6+Math.random()*.7),age:-Math.random()*.12,life:a*(.75+Math.random()*.5),color:o})}}update(e){e=Math.min(Math.max(e,0),.25);for(let t=e;t>1e-6;t-=.025)this.step(Math.min(.025,t));this.draw()}step(e){this.clods=this.clods.filter(t=>{if(t.age+=e,t.age<0)return!0;if(t.age>t.life)return!1;if((t.age<t.arrive||t.settle)&&!t.resting){t.velocity.y-=xm*e,t.position.addScaledVector(t.velocity,e),t.rotation.x+=t.spin.x*e,t.rotation.y+=t.spin.y*e,t.rotation.z+=t.spin.z*e;let n=this.groundHeight(t.position.x,t.position.z)+t.size*.5;t.position.y<n&&t.velocity.y<0&&(t.position.y=n,!t.bounced&&t.velocity.y<-1.2?(t.velocity.y*=-.28,t.velocity.x*=.45,t.velocity.z*=.45,t.bounced=!0):(t.resting=!0,t.restAge=t.age))}return!(t.age>=t.arrive&&!t.settle)}),this.puffs=this.puffs.filter(t=>(t.age+=e,t.age>=0&&(t.position.addScaledVector(t.velocity,e),t.velocity.multiplyScalar(Math.exp(-e*2.2))),t.age<t.life)),this.dusts=this.dusts.filter(t=>(t.age+=e,t.age>=0&&(t.position.addScaledVector(t.velocity,e),t.velocity.multiplyScalar(Math.exp(-e*1.6))),t.age<t.life))}draw(){let e=this.dummy,t=this.clods.length;for(let a=0;a<t;a++){let o=this.clods[a],c=o.age>=0,l=o.resting?Math.max(0,1-(o.age-o.restAge)/.9):1;e.position.copy(o.position),e.rotation.copy(o.rotation),e.scale.setScalar(c?o.size*l:0),e.updateMatrix(),this.clodMesh.setMatrixAt(a,e.matrix),this.clodMesh.setColorAt(a,this.color.setHex(o.color))}this.clodMesh.count=t,this.clodMesh.instanceMatrix.needsUpdate=!0,this.clodMesh.instanceColor&&(this.clodMesh.instanceColor.needsUpdate=!0);for(let a=0;a<this.puffs.length;a++){let o=this.puffs[a],c=Math.max(0,o.age)/o.life,l=o.age<0?0:Math.sin(Math.min(1,c*1.25)*Math.PI)*(1+c*.6);e.position.copy(o.position),e.rotation.set(o.age*.7,o.age,0),e.scale.setScalar(o.size*l),e.updateMatrix(),this.puffMesh.setMatrixAt(a,e.matrix),this.puffMesh.setColorAt(a,this.color.setHex(o.color))}this.puffMesh.count=this.puffs.length,this.puffMesh.instanceMatrix.needsUpdate=!0,this.puffMesh.instanceColor&&(this.puffMesh.instanceColor.needsUpdate=!0);let n=this.dustPoints.geometry.attributes.position,s=this.dustPoints.geometry.attributes.size,r=this.dustPoints.geometry.attributes.alpha;for(let a=0;a<this.dusts.length;a++){let o=this.dusts[a],c=Math.max(0,o.age)/o.life;n.setXYZ(a,o.position.x,o.position.y,o.position.z),s.setX(a,o.age<0?0:o.size*(.55+c*1.1)),r.setX(a,o.age<0?0:o.opacity*Math.min(1,c*6)*(1-c)**1.5)}this.dustPoints.geometry.setDrawRange(0,this.dusts.length);for(let a of[n,s,r])a.needsUpdate=!0}clear(){this.clods=[],this.puffs=[],this.dusts=[],this.clodMesh.count=0,this.puffMesh.count=0,this.dustPoints.geometry.setDrawRange(0,0)}dispose(){this.clear();for(let e of[this.clodMesh,this.puffMesh])e.geometry.dispose(),e.material.dispose(),e.dispose();this.dustPoints.geometry.dispose(),this.dustMaterial.dispose(),this.removeFromParent()}};var Xr={name:"CopyShader",uniforms:{tDiffuse:{value:null},opacity:{value:1}},vertexShader:`

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


		}`};var Fn=class{constructor(){this.isPass=!0,this.enabled=!0,this.needsSwap=!0,this.clear=!1,this.renderToScreen=!1}setSize(){}render(){console.error("THREE.Pass: .render() must be implemented in derived pass.")}dispose(){}},hS=new as(-1,1,1,-1,0,1),Md=class extends ut{constructor(){super(),this.setAttribute("position",new rt([-1,3,0,-1,-1,0,3,-1,0],3)),this.setAttribute("uv",new rt([0,2,0,0,2,0],2))}},uS=new Md,vs=class{constructor(e){this._mesh=new et(uS,e)}dispose(){this._mesh.geometry.dispose()}render(e){e.render(this._mesh,hS)}get material(){return this._mesh.material}set material(e){this._mesh.material=e}};var ys=class extends Fn{constructor(e,t="tDiffuse"){super(),this.textureID=t,this.uniforms=null,this.material=null,e instanceof bt?(this.uniforms=e.uniforms,this.material=e):e&&(this.uniforms=gn.clone(e.uniforms),this.material=new bt({name:e.name!==void 0?e.name:"unspecified",defines:Object.assign({},e.defines),uniforms:this.uniforms,vertexShader:e.vertexShader,fragmentShader:e.fragmentShader})),this._fsQuad=new vs(this.material)}render(e,t,n){this.uniforms[this.textureID]&&(this.uniforms[this.textureID].value=n.texture),this._fsQuad.material=this.material,this.renderToScreen?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(t),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this._fsQuad.render(e))}dispose(){this.material.dispose(),this._fsQuad.dispose()}};var go=class extends Fn{constructor(e,t){super(),this.scene=e,this.camera=t,this.clear=!0,this.needsSwap=!1,this.inverse=!1}render(e,t,n){let s=e.getContext(),r=e.state;r.buffers.color.setMask(!1),r.buffers.depth.setMask(!1),r.buffers.color.setLocked(!0),r.buffers.depth.setLocked(!0);let a,o;this.inverse?(a=0,o=1):(a=1,o=0),r.buffers.stencil.setTest(!0),r.buffers.stencil.setOp(s.REPLACE,s.REPLACE,s.REPLACE),r.buffers.stencil.setFunc(s.ALWAYS,a,4294967295),r.buffers.stencil.setClear(o),r.buffers.stencil.setLocked(!0),e.setRenderTarget(n),this.clear&&e.clear(),e.render(this.scene,this.camera),e.setRenderTarget(t),this.clear&&e.clear(),e.render(this.scene,this.camera),r.buffers.color.setLocked(!1),r.buffers.depth.setLocked(!1),r.buffers.color.setMask(!0),r.buffers.depth.setMask(!0),r.buffers.stencil.setLocked(!1),r.buffers.stencil.setFunc(s.EQUAL,1,4294967295),r.buffers.stencil.setOp(s.KEEP,s.KEEP,s.KEEP),r.buffers.stencil.setLocked(!0)}},hh=class extends Fn{constructor(){super(),this.needsSwap=!1}render(e){e.state.buffers.stencil.setLocked(!1),e.state.buffers.stencil.setTest(!1)}};var uh=class{constructor(e,t){if(this.renderer=e,this._pixelRatio=e.getPixelRatio(),t===void 0){let n=e.getSize(new Z);this._width=n.width,this._height=n.height,t=new Ht(this._width*this._pixelRatio,this._height*this._pixelRatio,{type:nn}),t.texture.name="EffectComposer.rt1"}else this._width=t.width,this._height=t.height;this.renderTarget1=t,this.renderTarget2=t.clone(),this.renderTarget2.texture.name="EffectComposer.rt2",this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2,this.renderToScreen=!0,this.passes=[],this.copyPass=new ys(Xr),this.copyPass.material.blending=zt,this.timer=new Va}swapBuffers(){let e=this.readBuffer;this.readBuffer=this.writeBuffer,this.writeBuffer=e}addPass(e){this.passes.push(e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}insertPass(e,t){this.passes.splice(t,0,e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}removePass(e){let t=this.passes.indexOf(e);t!==-1&&this.passes.splice(t,1)}isLastEnabledPass(e){for(let t=e+1;t<this.passes.length;t++)if(this.passes[t].enabled)return!1;return!0}render(e){this.timer.update(),e===void 0&&(e=this.timer.getDelta());let t=this.renderer.getRenderTarget(),n=!1;for(let s=0,r=this.passes.length;s<r;s++){let a=this.passes[s];if(a.enabled!==!1){if(a.renderToScreen=this.renderToScreen&&this.isLastEnabledPass(s),a.render(this.renderer,this.writeBuffer,this.readBuffer,e,n),a.needsSwap){if(n){let o=this.renderer.getContext(),c=this.renderer.state.buffers.stencil;c.setFunc(o.NOTEQUAL,1,4294967295),this.copyPass.render(this.renderer,this.writeBuffer,this.readBuffer,e),c.setFunc(o.EQUAL,1,4294967295)}this.swapBuffers()}go!==void 0&&(a instanceof go?n=!0:a instanceof hh&&(n=!1))}}this.renderer.setRenderTarget(t)}reset(e){if(e===void 0){let t=this.renderer.getSize(new Z);this._pixelRatio=this.renderer.getPixelRatio(),this._width=t.width,this._height=t.height,e=this.renderTarget1.clone(),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.renderTarget1=e,this.renderTarget2=e.clone(),this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2}setSize(e,t){this._width=e,this._height=t;let n=this._width*this._pixelRatio,s=this._height*this._pixelRatio;this.renderTarget1.setSize(n,s),this.renderTarget2.setSize(n,s);for(let r=0;r<this.passes.length;r++)this.passes[r].setSize(n,s)}setPixelRatio(e){this._pixelRatio=e,this.setSize(this._width,this._height)}dispose(){this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.copyPass.dispose()}};var dh=class extends Fn{constructor(e,t,n=null,s=null,r=null){super(),this.scene=e,this.camera=t,this.overrideMaterial=n,this.clearColor=s,this.clearAlpha=r,this.clear=!0,this.clearDepth=!1,this.needsSwap=!1,this.isRenderPass=!0,this._oldClearColor=new Pe}render(e,t,n){let s=e.autoClear;e.autoClear=!1;let r,a;this.overrideMaterial!==null&&(a=this.scene.overrideMaterial,this.scene.overrideMaterial=this.overrideMaterial),this.clearColor!==null&&(e.getClearColor(this._oldClearColor),e.setClearColor(this.clearColor,e.getClearAlpha())),this.clearAlpha!==null&&(r=e.getClearAlpha(),e.setClearAlpha(this.clearAlpha)),this.clearDepth==!0&&e.clearDepth(),e.setRenderTarget(this.renderToScreen?null:n),this.clear===!0&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),e.render(this.scene,this.camera),this.clearColor!==null&&e.setClearColor(this._oldClearColor),this.clearAlpha!==null&&e.setClearAlpha(r),this.overrideMaterial!==null&&(this.scene.overrideMaterial=a),e.autoClear=s}};var _o={name:"GTAOShader",defines:{PERSPECTIVE_CAMERA:1,SAMPLES:16,NORMAL_VECTOR_TYPE:1,DEPTH_SWIZZLING:"x",SCREEN_SPACE_RADIUS:0,SCREEN_SPACE_RADIUS_SCALE:100,SCENE_CLIP_BOX:0},uniforms:{tNormal:{value:null},tDepth:{value:null},tNoise:{value:null},resolution:{value:new Z},cameraNear:{value:null},cameraFar:{value:null},cameraProjectionMatrix:{value:new st},cameraProjectionMatrixInverse:{value:new st},cameraWorldMatrix:{value:new st},radius:{value:.25},distanceExponent:{value:1},thickness:{value:1},distanceFallOff:{value:1},scale:{value:1},sceneBoxMin:{value:new R(-1,-1,-1)},sceneBoxMax:{value:new R(1,1,1)}},vertexShader:`

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
		}`},xo={name:"GTAODepthShader",defines:{PERSPECTIVE_CAMERA:1},uniforms:{tDepth:{value:null},cameraNear:{value:null},cameraFar:{value:null}},vertexShader:`
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
		}`};function ym(i=5){let e=Math.floor(i)%2===0?Math.floor(i)+1:Math.floor(i),t=dS(e),n=t.length,s=new Uint8Array(n*4);for(let a=0;a<n;++a){let o=t[a],c=2*Math.PI*o/n,l=new R(Math.cos(c),Math.sin(c),0).normalize();s[a*4]=(l.x*.5+.5)*255,s[a*4+1]=(l.y*.5+.5)*255,s[a*4+2]=127,s[a*4+3]=255}let r=new Ui(s,e,e);return r.wrapS=kn,r.wrapT=kn,r.needsUpdate=!0,r}function dS(i){let e=Math.floor(i)%2===0?Math.floor(i)+1:Math.floor(i),t=e*e,n=Array(t).fill(0),s=Math.floor(e/2),r=e-1;for(let a=1;a<=t;){if(s===-1&&r===e?(r=e-2,s=0):(r===e&&(r=0),s<0&&(s=e-1)),n[s*e+r]!==0){r-=2,s++;continue}else n[s*e+r]=a++;r++,s--}return n}var vo={name:"PoissonDenoiseShader",defines:{SAMPLES:16,SAMPLE_VECTORS:Sd(16,2,1),NORMAL_VECTOR_TYPE:1,DEPTH_VALUE_SOURCE:0},uniforms:{tDiffuse:{value:null},tNormal:{value:null},tDepth:{value:null},tNoise:{value:null},resolution:{value:new Z},cameraProjectionMatrixInverse:{value:new st},lumaPhi:{value:5},depthPhi:{value:5},normalPhi:{value:5},radius:{value:4},index:{value:0}},vertexShader:`

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
		}`};function Sd(i,e,t){let n=fS(i,e,t),s="vec3[SAMPLES](";for(let r=0;r<i;r++){let a=n[r];s+=`vec3(${a.x}, ${a.y}, ${a.z})${r<i-1?",":")"}`}return s}function fS(i,e,t){let n=[];for(let s=0;s<i;s++){let r=2*Math.PI*e*s/i,a=Math.pow(s/(i-1),t);n.push(new R(Math.cos(r),Math.sin(r),a))}return n}var ph=class{constructor(e=Math){this.grad3=[[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]],this.grad4=[[0,1,1,1],[0,1,1,-1],[0,1,-1,1],[0,1,-1,-1],[0,-1,1,1],[0,-1,1,-1],[0,-1,-1,1],[0,-1,-1,-1],[1,0,1,1],[1,0,1,-1],[1,0,-1,1],[1,0,-1,-1],[-1,0,1,1],[-1,0,1,-1],[-1,0,-1,1],[-1,0,-1,-1],[1,1,0,1],[1,1,0,-1],[1,-1,0,1],[1,-1,0,-1],[-1,1,0,1],[-1,1,0,-1],[-1,-1,0,1],[-1,-1,0,-1],[1,1,1,0],[1,1,-1,0],[1,-1,1,0],[1,-1,-1,0],[-1,1,1,0],[-1,1,-1,0],[-1,-1,1,0],[-1,-1,-1,0]],this.p=[];for(let t=0;t<256;t++)this.p[t]=Math.floor(e.random()*256);this.perm=[];for(let t=0;t<512;t++)this.perm[t]=this.p[t&255];this.simplex=[[0,1,2,3],[0,1,3,2],[0,0,0,0],[0,2,3,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,3,0],[0,2,1,3],[0,0,0,0],[0,3,1,2],[0,3,2,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,3,2,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,0,3],[0,0,0,0],[1,3,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,3,0,1],[2,3,1,0],[1,0,2,3],[1,0,3,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,3,1],[0,0,0,0],[2,1,3,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,1,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,0,1,2],[3,0,2,1],[0,0,0,0],[3,1,2,0],[2,1,0,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,1,0,2],[0,0,0,0],[3,2,0,1],[3,2,1,0]]}noise(e,t){let n,s,r,a=.5*(Math.sqrt(3)-1),o=(e+t)*a,c=Math.floor(e+o),l=Math.floor(t+o),h=(3-Math.sqrt(3))/6,d=(c+l)*h,u=c-d,f=l-d,g=e-u,_=t-f,p,m;g>_?(p=1,m=0):(p=0,m=1);let M=g-p+h,S=_-m+h,y=g-1+2*h,T=_-1+2*h,b=c&255,P=l&255,x=this.perm[b+this.perm[P]]%12,E=this.perm[b+p+this.perm[P+m]]%12,C=this.perm[b+1+this.perm[P+1]]%12,I=.5-g*g-_*_;I<0?n=0:(I*=I,n=I*I*this._dot(this.grad3[x],g,_));let L=.5-M*M-S*S;L<0?s=0:(L*=L,s=L*L*this._dot(this.grad3[E],M,S));let X=.5-y*y-T*T;return X<0?r=0:(X*=X,r=X*X*this._dot(this.grad3[C],y,T)),70*(n+s+r)}noise3d(e,t,n){let s,r,a,o,l=(e+t+n)*.3333333333333333,h=Math.floor(e+l),d=Math.floor(t+l),u=Math.floor(n+l),f=1/6,g=(h+d+u)*f,_=h-g,p=d-g,m=u-g,M=e-_,S=t-p,y=n-m,T,b,P,x,E,C;M>=S?S>=y?(T=1,b=0,P=0,x=1,E=1,C=0):M>=y?(T=1,b=0,P=0,x=1,E=0,C=1):(T=0,b=0,P=1,x=1,E=0,C=1):S<y?(T=0,b=0,P=1,x=0,E=1,C=1):M<y?(T=0,b=1,P=0,x=0,E=1,C=1):(T=0,b=1,P=0,x=1,E=1,C=0);let I=M-T+f,L=S-b+f,X=y-P+f,q=M-x+2*f,F=S-E+2*f,Y=y-C+2*f,W=M-1+3*f,ie=S-1+3*f,ne=y-1+3*f,ge=h&255,ue=d&255,xe=u&255,Ne=this.perm[ge+this.perm[ue+this.perm[xe]]]%12,it=this.perm[ge+T+this.perm[ue+b+this.perm[xe+P]]]%12,Xe=this.perm[ge+x+this.perm[ue+E+this.perm[xe+C]]]%12,j=this.perm[ge+1+this.perm[ue+1+this.perm[xe+1]]]%12,he=.6-M*M-S*S-y*y;he<0?s=0:(he*=he,s=he*he*this._dot3(this.grad3[Ne],M,S,y));let le=.6-I*I-L*L-X*X;le<0?r=0:(le*=le,r=le*le*this._dot3(this.grad3[it],I,L,X));let Te=.6-q*q-F*F-Y*Y;Te<0?a=0:(Te*=Te,a=Te*Te*this._dot3(this.grad3[Xe],q,F,Y));let Fe=.6-W*W-ie*ie-ne*ne;return Fe<0?o=0:(Fe*=Fe,o=Fe*Fe*this._dot3(this.grad3[j],W,ie,ne)),32*(s+r+a+o)}noise4d(e,t,n,s){let r=this.grad4,a=this.simplex,o=this.perm,c=(Math.sqrt(5)-1)/4,l=(5-Math.sqrt(5))/20,h,d,u,f,g,_=(e+t+n+s)*c,p=Math.floor(e+_),m=Math.floor(t+_),M=Math.floor(n+_),S=Math.floor(s+_),y=(p+m+M+S)*l,T=p-y,b=m-y,P=M-y,x=S-y,E=e-T,C=t-b,I=n-P,L=s-x,X=E>C?32:0,q=E>I?16:0,F=C>I?8:0,Y=E>L?4:0,W=C>L?2:0,ie=I>L?1:0,ne=X+q+F+Y+W+ie,ge=a[ne][0]>=3?1:0,ue=a[ne][1]>=3?1:0,xe=a[ne][2]>=3?1:0,Ne=a[ne][3]>=3?1:0,it=a[ne][0]>=2?1:0,Xe=a[ne][1]>=2?1:0,j=a[ne][2]>=2?1:0,he=a[ne][3]>=2?1:0,le=a[ne][0]>=1?1:0,Te=a[ne][1]>=1?1:0,Fe=a[ne][2]>=1?1:0,ke=a[ne][3]>=1?1:0,oe=E-ge+l,ee=C-ue+l,O=I-xe+l,H=L-Ne+l,Q=E-it+2*l,G=C-Xe+2*l,V=I-j+2*l,se=L-he+2*l,ce=E-le+3*l,fe=C-Te+3*l,me=I-Fe+3*l,D=L-ke+3*l,Me=E-1+4*l,Ve=C-1+4*l,A=I-1+4*l,v=L-1+4*l,U=p&255,B=m&255,k=M&255,pe=S&255,_e=o[U+o[B+o[k+o[pe]]]]%32,te=o[U+ge+o[B+ue+o[k+xe+o[pe+Ne]]]]%32,re=o[U+it+o[B+Xe+o[k+j+o[pe+he]]]]%32,Se=o[U+le+o[B+Te+o[k+Fe+o[pe+ke]]]]%32,Ie=o[U+1+o[B+1+o[k+1+o[pe+1]]]]%32,ve=.6-E*E-C*C-I*I-L*L;ve<0?h=0:(ve*=ve,h=ve*ve*this._dot4(r[_e],E,C,I,L));let ye=.6-oe*oe-ee*ee-O*O-H*H;ye<0?d=0:(ye*=ye,d=ye*ye*this._dot4(r[te],oe,ee,O,H));let Be=.6-Q*Q-G*G-V*V-se*se;Be<0?u=0:(Be*=Be,u=Be*Be*this._dot4(r[re],Q,G,V,se));let qe=.6-ce*ce-fe*fe-me*me-D*D;qe<0?f=0:(qe*=qe,f=qe*qe*this._dot4(r[Se],ce,fe,me,D));let Je=.6-Me*Me-Ve*Ve-A*A-v*v;return Je<0?g=0:(Je*=Je,g=Je*Je*this._dot4(r[Ie],Me,Ve,A,v)),27*(h+d+u+f+g)}_dot(e,t,n){return e[0]*t+e[1]*n}_dot3(e,t,n,s){return e[0]*t+e[1]*n+e[2]*s}_dot4(e,t,n,s,r){return e[0]*t+e[1]*n+e[2]*s+e[3]*r}};var yo=class i extends Fn{constructor(e,t,n=512,s=512,r,a,o){super(),this.width=n,this.height=s,this.clear=!0,this.camera=t,this.scene=e,this.output=0,this._renderGBuffer=!0,this._visibilityCache=[],this.blendIntensity=1,this.pdRings=2,this.pdRadiusExponent=2,this.pdSamples=16,this.gtaoNoiseTexture=ym(),this.pdNoiseTexture=this._generateNoise(),this.gtaoRenderTarget=new Ht(this.width,this.height,{type:nn}),this.pdRenderTarget=this.gtaoRenderTarget.clone(),this.gtaoMaterial=new bt({defines:Object.assign({},_o.defines),uniforms:gn.clone(_o.uniforms),vertexShader:_o.vertexShader,fragmentShader:_o.fragmentShader,blending:zt,depthTest:!1,depthWrite:!1}),this.gtaoMaterial.defines.PERSPECTIVE_CAMERA=this.camera.isPerspectiveCamera?1:0,this.gtaoMaterial.uniforms.tNoise.value=this.gtaoNoiseTexture,this.gtaoMaterial.uniforms.resolution.value.set(this.width,this.height),this.gtaoMaterial.uniforms.cameraNear.value=this.camera.near,this.gtaoMaterial.uniforms.cameraFar.value=this.camera.far,this.normalMaterial=new Na,this.normalMaterial.blending=zt,this.pdMaterial=new bt({defines:Object.assign({},vo.defines),uniforms:gn.clone(vo.uniforms),vertexShader:vo.vertexShader,fragmentShader:vo.fragmentShader,depthTest:!1,depthWrite:!1}),this.pdMaterial.uniforms.tDiffuse.value=this.gtaoRenderTarget.texture,this.pdMaterial.uniforms.tNoise.value=this.pdNoiseTexture,this.pdMaterial.uniforms.resolution.value.set(this.width,this.height),this.pdMaterial.uniforms.lumaPhi.value=10,this.pdMaterial.uniforms.depthPhi.value=2,this.pdMaterial.uniforms.normalPhi.value=3,this.pdMaterial.uniforms.radius.value=8,this.depthRenderMaterial=new bt({defines:Object.assign({},xo.defines),uniforms:gn.clone(xo.uniforms),vertexShader:xo.vertexShader,fragmentShader:xo.fragmentShader,blending:zt}),this.depthRenderMaterial.uniforms.cameraNear.value=this.camera.near,this.depthRenderMaterial.uniforms.cameraFar.value=this.camera.far,this.copyMaterial=new bt({uniforms:gn.clone(Xr.uniforms),vertexShader:Xr.vertexShader,fragmentShader:Xr.fragmentShader,transparent:!0,depthTest:!1,depthWrite:!1,blendSrc:Ya,blendDst:Ns,blendEquation:Pn,blendSrcAlpha:qa,blendDstAlpha:Ns,blendEquationAlpha:Pn}),this.blendMaterial=new bt({uniforms:gn.clone(fh.uniforms),vertexShader:fh.vertexShader,fragmentShader:fh.fragmentShader,transparent:!0,depthTest:!1,depthWrite:!1,blending:$l,blendSrc:Ya,blendDst:Ns,blendEquation:Pn,blendSrcAlpha:qa,blendDstAlpha:Ns,blendEquationAlpha:Pn}),this._fsQuad=new vs(null),this._originalClearColor=new Pe,this.setGBuffer(r?r.depthTexture:void 0,r?r.normalTexture:void 0),a!==void 0&&this.updateGtaoMaterial(a),o!==void 0&&this.updatePdMaterial(o)}setSize(e,t){this.width=e,this.height=t,this.gtaoRenderTarget.setSize(e,t),this.normalRenderTarget.setSize(e,t),this.pdRenderTarget.setSize(e,t),this.gtaoMaterial.uniforms.resolution.value.set(e,t),this.gtaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.gtaoMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this.pdMaterial.uniforms.resolution.value.set(e,t),this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse)}dispose(){this.gtaoNoiseTexture.dispose(),this.pdNoiseTexture.dispose(),this.normalRenderTarget.dispose(),this.gtaoRenderTarget.dispose(),this.pdRenderTarget.dispose(),this.normalMaterial.dispose(),this.pdMaterial.dispose(),this.copyMaterial.dispose(),this.depthRenderMaterial.dispose(),this._fsQuad.dispose()}get gtaoMap(){return this.pdRenderTarget.texture}setGBuffer(e,t){e!==void 0?(this.depthTexture=e,this.normalTexture=t,this._renderGBuffer=!1):(this.depthTexture=new Kn,this.depthTexture.format=_i,this.depthTexture.type=fs,this.normalRenderTarget=new Ht(this.width,this.height,{minFilter:Ot,magFilter:Ot,type:nn,depthTexture:this.depthTexture}),this.normalTexture=this.normalRenderTarget.texture,this._renderGBuffer=!0);let n=this.normalTexture?1:0,s=this.depthTexture===this.normalTexture?"w":"x";this.gtaoMaterial.defines.NORMAL_VECTOR_TYPE=n,this.gtaoMaterial.defines.DEPTH_SWIZZLING=s,this.gtaoMaterial.uniforms.tNormal.value=this.normalTexture,this.gtaoMaterial.uniforms.tDepth.value=this.depthTexture,this.pdMaterial.defines.NORMAL_VECTOR_TYPE=n,this.pdMaterial.defines.DEPTH_SWIZZLING=s,this.pdMaterial.uniforms.tNormal.value=this.normalTexture,this.pdMaterial.uniforms.tDepth.value=this.depthTexture,this.depthRenderMaterial.uniforms.tDepth.value=this.normalRenderTarget.depthTexture}setSceneClipBox(e){e?(this.gtaoMaterial.needsUpdate=this.gtaoMaterial.defines.SCENE_CLIP_BOX!==1,this.gtaoMaterial.defines.SCENE_CLIP_BOX=1,this.gtaoMaterial.uniforms.sceneBoxMin.value.copy(e.min),this.gtaoMaterial.uniforms.sceneBoxMax.value.copy(e.max)):(this.gtaoMaterial.needsUpdate=this.gtaoMaterial.defines.SCENE_CLIP_BOX===0,this.gtaoMaterial.defines.SCENE_CLIP_BOX=0)}updateGtaoMaterial(e){e.radius!==void 0&&(this.gtaoMaterial.uniforms.radius.value=e.radius),e.distanceExponent!==void 0&&(this.gtaoMaterial.uniforms.distanceExponent.value=e.distanceExponent),e.thickness!==void 0&&(this.gtaoMaterial.uniforms.thickness.value=e.thickness),e.distanceFallOff!==void 0&&(this.gtaoMaterial.uniforms.distanceFallOff.value=e.distanceFallOff,this.gtaoMaterial.needsUpdate=!0),e.scale!==void 0&&(this.gtaoMaterial.uniforms.scale.value=e.scale),e.samples!==void 0&&e.samples!==this.gtaoMaterial.defines.SAMPLES&&(this.gtaoMaterial.defines.SAMPLES=e.samples,this.gtaoMaterial.needsUpdate=!0),e.screenSpaceRadius!==void 0&&(e.screenSpaceRadius?1:0)!==this.gtaoMaterial.defines.SCREEN_SPACE_RADIUS&&(this.gtaoMaterial.defines.SCREEN_SPACE_RADIUS=e.screenSpaceRadius?1:0,this.gtaoMaterial.needsUpdate=!0)}updatePdMaterial(e){let t=!1;e.lumaPhi!==void 0&&(this.pdMaterial.uniforms.lumaPhi.value=e.lumaPhi),e.depthPhi!==void 0&&(this.pdMaterial.uniforms.depthPhi.value=e.depthPhi),e.normalPhi!==void 0&&(this.pdMaterial.uniforms.normalPhi.value=e.normalPhi),e.radius!==void 0&&e.radius!==this.radius&&(this.pdMaterial.uniforms.radius.value=e.radius),e.radiusExponent!==void 0&&e.radiusExponent!==this.pdRadiusExponent&&(this.pdRadiusExponent=e.radiusExponent,t=!0),e.rings!==void 0&&e.rings!==this.pdRings&&(this.pdRings=e.rings,t=!0),e.samples!==void 0&&e.samples!==this.pdSamples&&(this.pdSamples=e.samples,t=!0),t&&(this.pdMaterial.defines.SAMPLES=this.pdSamples,this.pdMaterial.defines.SAMPLE_VECTORS=Sd(this.pdSamples,this.pdRings,this.pdRadiusExponent),this.pdMaterial.needsUpdate=!0)}render(e,t,n){switch(this._renderGBuffer&&(this._overrideVisibility(),this._renderOverride(e,this.normalMaterial,this.normalRenderTarget,7829503,1),this._restoreVisibility()),this.gtaoMaterial.uniforms.cameraNear.value=this.camera.near,this.gtaoMaterial.uniforms.cameraFar.value=this.camera.far,this.gtaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.gtaoMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this.gtaoMaterial.uniforms.cameraWorldMatrix.value.copy(this.camera.matrixWorld),this._renderPass(e,this.gtaoMaterial,this.gtaoRenderTarget,16777215,1),this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this._renderPass(e,this.pdMaterial,this.pdRenderTarget,16777215,1),this.output){case i.OUTPUT.Off:break;case i.OUTPUT.Diffuse:this.copyMaterial.uniforms.tDiffuse.value=n.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case i.OUTPUT.AO:this.copyMaterial.uniforms.tDiffuse.value=this.gtaoRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case i.OUTPUT.Denoise:this.copyMaterial.uniforms.tDiffuse.value=this.pdRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case i.OUTPUT.Depth:this.depthRenderMaterial.uniforms.cameraNear.value=this.camera.near,this.depthRenderMaterial.uniforms.cameraFar.value=this.camera.far,this._renderPass(e,this.depthRenderMaterial,this.renderToScreen?null:t);break;case i.OUTPUT.Normal:this.copyMaterial.uniforms.tDiffuse.value=this.normalRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case i.OUTPUT.Default:this.copyMaterial.uniforms.tDiffuse.value=n.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t),this.blendMaterial.uniforms.intensity.value=this.blendIntensity,this.blendMaterial.uniforms.tDiffuse.value=this.pdRenderTarget.texture,this._renderPass(e,this.blendMaterial,this.renderToScreen?null:t);break;default:console.warn("THREE.GTAOPass: Unknown output type.")}}_renderPass(e,t,n,s,r){e.getClearColor(this._originalClearColor);let a=e.getClearAlpha(),o=e.autoClear;e.setRenderTarget(n),e.autoClear=!1,s!=null&&(e.setClearColor(s),e.setClearAlpha(r||0),e.clear()),this._fsQuad.material=t,this._fsQuad.render(e),e.autoClear=o,e.setClearColor(this._originalClearColor),e.setClearAlpha(a)}_renderOverride(e,t,n,s,r){e.getClearColor(this._originalClearColor);let a=e.getClearAlpha(),o=e.autoClear;e.setRenderTarget(n),e.autoClear=!1,s=t.clearColor||s,r=t.clearAlpha||r,s!=null&&(e.setClearColor(s),e.setClearAlpha(r||0),e.clear()),this.scene.overrideMaterial=t,e.render(this.scene,this.camera),this.scene.overrideMaterial=null,e.autoClear=o,e.setClearColor(this._originalClearColor),e.setClearAlpha(a)}_overrideVisibility(){let e=this.scene,t=this._visibilityCache;e.traverse(function(n){(n.isPoints||n.isLine||n.isLine2)&&n.visible&&(n.visible=!1,t.push(n))})}_restoreVisibility(){let e=this._visibilityCache;for(let t=0;t<e.length;t++)e[t].visible=!0;e.length=0}_generateNoise(e=64){let t=new ph,n=e*e*4,s=new Uint8Array(n);for(let a=0;a<e;a++)for(let o=0;o<e;o++){let c=a,l=o;s[(a*e+o)*4]=(t.noise(c,l)*.5+.5)*255,s[(a*e+o)*4+1]=(t.noise(c+e,l)*.5+.5)*255,s[(a*e+o)*4+2]=(t.noise(c,l+e)*.5+.5)*255,s[(a*e+o)*4+3]=(t.noise(c+e,l+e)*.5+.5)*255}let r=new Ui(s,e,e,bn,hn);return r.wrapS=kn,r.wrapT=kn,r.needsUpdate=!0,r}};yo.OUTPUT={Off:-1,Default:0,Diffuse:1,Depth:2,Normal:3,AO:4,Denoise:5};var Mo={name:"OutputShader",uniforms:{tDiffuse:{value:null},toneMappingExposure:{value:1}},vertexShader:`
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

		}`};var mh=class extends Fn{constructor(){super(),this.isOutputPass=!0,this.uniforms=gn.clone(Mo.uniforms),this.material=new Ar({name:Mo.name,uniforms:this.uniforms,vertexShader:Mo.vertexShader,fragmentShader:Mo.fragmentShader}),this._fsQuad=new vs(this.material),this._outputColorSpace=null,this._toneMapping=null}render(e,t,n){this.uniforms.tDiffuse.value=n.texture,this.uniforms.toneMappingExposure.value=e.toneMappingExposure,(this._outputColorSpace!==e.outputColorSpace||this._toneMapping!==e.toneMapping)&&(this._outputColorSpace=e.outputColorSpace,this._toneMapping=e.toneMapping,this.material.defines={},ht.getTransfer(this._outputColorSpace)===pt&&(this.material.defines.SRGB_TRANSFER=""),this._toneMapping===Za?this.material.defines.LINEAR_TONE_MAPPING="":this._toneMapping===$a?this.material.defines.REINHARD_TONE_MAPPING="":this._toneMapping===Ja?this.material.defines.CINEON_TONE_MAPPING="":this._toneMapping===hs?this.material.defines.ACES_FILMIC_TONE_MAPPING="":this._toneMapping===Ka?this.material.defines.AGX_TONE_MAPPING="":this._toneMapping===Fs?this.material.defines.NEUTRAL_TONE_MAPPING="":this._toneMapping===ja&&(this.material.defines.CUSTOM_TONE_MAPPING=""),this.material.needsUpdate=!0),this.renderToScreen===!0?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(t),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this._fsQuad.render(e))}dispose(){this.material.dispose(),this._fsQuad.dispose()}};var Mm={name:"FXAAShader",uniforms:{tDiffuse:{value:null},resolution:{value:new Z(1/1024,1/512)}},vertexShader:`

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

		}`};var gh=class extends ys{constructor(){super(Mm)}setSize(e,t){this.material.uniforms.resolution.value.set(1/e,1/t)}};var bd=class extends yo{_overrideVisibility(){super._overrideVisibility();let e=this._visibilityCache;this.scene.traverse(t=>{(t.isSprite||t.isLineSegments2||t.userData.skipAO)&&t.visible&&(t.visible=!1,e.push(t))})}_renderOverride(e,...t){let n=this.scene.background;this.scene.background=null,super._renderOverride(e,...t),this.scene.background=n}},pS={uniforms:{tDiffuse:{value:null},tDepth:{value:null},tNormal:{value:null},resolution:{value:new Z(1,1)},cameraNear:{value:.1},cameraFar:{value:1e3},thickness:{value:1},strength:{value:.92},vignette:{value:.16}},vertexShader:"varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }",fragmentShader:`
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
    }`},_h=class{constructor(e,t,n){this.renderer=e,this.scene=t,this.camera=n;let s=e.getDrawingBufferSize(new Z),r=new Ht(s.x,s.y,{type:nn,samples:4});this.composer=new uh(e,r),this.composer.addPass(new dh(t,n)),this.ao=new bd(t,n,s.x,s.y),this.ao.blendIntensity=.9,this.composer.addPass(this.ao),this.ink=new ys(pS),this.ink.uniforms.tDepth.value=this.ao.depthTexture,this.ink.uniforms.tNormal.value=this.ao.normalTexture,this.composer.addPass(this.ink),this.composer.addPass(new mh),this.composer.addPass(new gh)}configure({span:e,tile:t}){this.ao.updateGtaoMaterial({radius:Math.max(t*1.6,e*.018),distanceExponent:1.4,thickness:1.2,scale:1.05,samples:16}),this.ao.updatePdMaterial({lumaPhi:10,depthPhi:2,normalPhi:3,radius:6,rings:2,samples:12})}setSize(e,t,n){this.composer.setPixelRatio(n),this.composer.setSize(e,t);let s=this.renderer.getDrawingBufferSize(new Z);this.ink.uniforms.resolution.value.copy(s),this.ink.uniforms.thickness.value=1.35*n}setLook({vignette:e=0,ink:t=.92}={}){this.ink.uniforms.vignette.value=e,this.ink.uniforms.strength.value=t}setSamples(e){for(let t of[this.composer.renderTarget1,this.composer.renderTarget2])t.samples!==e&&(t.samples=e,t.dispose())}render(){this.ink.uniforms.cameraNear.value=this.camera.near,this.ink.uniforms.cameraFar.value=this.camera.far,this.composer.render()}dispose(){this.ao.dispose(),this.composer.dispose()}};var Ws={dig:{color:15113984,opacity:.55,map:"target",pattern:"hatch",test:i=>i<0},dump:{color:40563,opacity:.5,map:"target",pattern:"dots",test:i=>i>0},restricted:{color:13983232,opacity:.42,map:"dumpability_static",pattern:"cross",test:i=>!i},dumpability:{color:29362,opacity:.28,map:"dumpability",pattern:"solid",test:i=>!!i},interaction:{color:5682409,opacity:.26,map:"interaction",pattern:"solid",test:i=>!!i}},Ow=new st,Vi=new ft,Sm=new Pe,bm=i=>i*i*(3-2*i),mS=i=>i<.5?4*i*i*i:1-(-2*i+2)**3/2,gS=i=>1+(1.4+1)*(i-1)**3+1.4*(i-1)**2,xh=(i,e,t)=>i+(e-i)*t,bi=(i,e,t)=>Math.min(t,Math.max(e,i)),_S=["dig","dump","transfer"],Em=["studio","paper","diorama"],xS={dig:{color:14256668,opacity:.42},dump:{color:3116906,opacity:.2,pattern:"solid"}},vS=i=>i==="studio"?Object.fromEntries(Object.entries(Ws).map(([e,t])=>[e,{...t,...xS[e]}])):Ws,wm="terra-viewer3d-quality",Tm="terra-viewer3d-presentation";function Am(i){if(!i)return;let e=new Set,t=new Set;i.traverse(n=>{if(n.geometry&&e.add(n.geometry),n.material)for(let s of Array.isArray(n.material)?n.material:[n.material])t.add(s)});for(let n of e)n.dispose();for(let n of t)n.dispose();i.removeFromParent(),i.clear()}function Rm(i){try{return localStorage.getItem(i)}catch{return null}}function Cm(i,e){try{localStorage.setItem(i,e)}catch{}}var vh=class{constructor(e,{onPick:t,onCameraChange:n,onError:s,onQualityChange:r}={}){this.element=e,this.onPick=t,this.onCameraChange=n,this.onError=s,this.onQualityChange=r,this.scene=new Ds,this.renderer=new Vc({antialias:!0,alpha:!1,preserveDrawingBuffer:!0,powerPreference:"high-performance"}),this.pixelRatio=Math.min(window.devicePixelRatio||1,2),this.renderer.setPixelRatio(this.pixelRatio),this.renderer.shadowMap.enabled=!0,this.renderer.shadowMap.type=Us,this.renderer.toneMapping=hs,this.renderer.toneMappingExposure=1,this.renderer.outputColorSpace=Lt,e.appendChild(this.renderer.domElement),this.renderer.domElement.addEventListener("webglcontextlost",h=>{h.preventDefault(),this.onError?.(new Error("The graphics context was lost. Reload the viewer to reconnect to the scene. Your live episode remains on the server."))});let a=new Fr(this.renderer);this.scene.environment=a.fromScene(new Yc,.04).texture,this.scene.environmentIntensity=.28,a.dispose(),this.camera=new Qt(32,1,.1,2e3),this.controls=new qc(this.camera,this.renderer.domElement),this.controls.enableDamping=!0,this.controls.dampingFactor=.085,this.controls.maxPolarAngle=Math.PI*.47,this.controls.minPolarAngle=.001,this.controls.screenSpacePanning=!0,this.controls.addEventListener("start",()=>{this.tween=null,this.follow&&(this.follow=!1,this.onCameraChange?.({follow:!1}))}),this.hemi=new Ba(13624575,10122832,.8),this.scene.add(this.hemi),this.sun=new Cr(16771529,2.7),this.sun.castShadow=!0,this.sun.shadow.mapSize.set(2048,2048),this.sun.shadow.bias=-4e-4,this.sun.shadow.radius=3,this.scene.add(this.sun),this.scene.add(this.sun.target),this.fill=new Cr(11127295,.35),this.scene.add(this.fill),this.world=new tt,this.scene.add(this.world),this.machines=new Map,this.heightScale=1,this.visibility={dig:!0,dump:!0,restricted:!1,dumpability:!1,interaction:!0,grid:!1,tags:!0},this.raycaster=new Ga,this.pointer=new Z,this.selected=null,this.reducedMotion=window.matchMedia("(prefers-reduced-motion: reduce)").matches,this.effects=new ch({groundHeight:(h,d)=>this.groundAt(h,d)}),this.scene.add(this.effects);let o=Rm(wm);this.quality=o==="fast"?"fast":"high",this.perf=o?null:{frames:0,elapsed:0};let c=Rm(Tm);this.presentation=Em.includes(c)?c:"studio";try{this.post=new _h(this.renderer,this.scene,this.camera)}catch(h){console.warn("Post-processing unavailable",h),this.post=null,this.quality="fast"}this.lineMaterials=new Set,this.applyLook();let l=null;e.addEventListener("pointerdown",h=>{l={x:h.clientX,y:h.clientY,button:h.button}}),e.addEventListener("pointerup",h=>{l?.button===0&&Math.hypot(h.clientX-l.x,h.clientY-l.y)<5&&this.pick(h),l=null}),this.resizeObserver=new ResizeObserver(()=>this.resize()),this.resizeObserver.observe(e),this.resize(),this.clock={last:performance.now(),idle:0},this.renderer.setAnimationLoop(h=>{this.update(h),this.controls.update(),this.render()})}render(){this.quality==="high"&&this.post?this.post.render():this.renderer.render(this.scene,this.camera)}measure(e){if(!this.perf||!this.frame||this.quality!=="high"||document.hidden||(++this.perf.frames>20&&(this.perf.elapsed+=e),this.perf.frames<110))return;let t=this.perf.elapsed/(this.perf.frames-20);this.perf=null,t>1/24&&(this.quality="fast",this.onQualityChange?.("fast"))}setQuality(e){return this.quality=e==="fast"||!this.post?"fast":"high",Cm(wm,this.quality),this.quality}applyLook(){let e=this.presentation,t=e==="paper",n=e==="studio",s=e==="diorama";this.palette=mo[e],this.scene.background=t?new Pe(12,12,12):n?this.backdrop||(this.backdrop=lm()):this.sky||(this.sky=cm()),this.scene.fog=s&&this.span?new da(new Pe(zi.sky[1]),this.span*3.2,this.span*7.5):null,this.renderer.toneMapping=t?Fs:hs,this.renderer.toneMappingExposure=1,this.scene.environmentIntensity=n?.5:.28,this.hemi.color.set(t?16777215:n?14805747:13624575),this.hemi.groundColor.set(t?9275520:n?8022616:10122832),this.hemi.intensity=t?.9:n?.6:.8,this.sun.color.set(t?16777215:n?16773340:16771529),this.sun.intensity=t?2.3:n?2.9:2.7,this.sun.shadow.radius=n?5:3,this.fill.color.set(t?16777215:n?13031925:11127295),this.fill.intensity=t?.45:n?.4:.35,this.effects.puffsEnabled=s,this.effects.dustEnabled=n,this.effects.palette=this.palette.clods,si.uMotion.value=s?1:0,this.post?.setLook({vignette:s?.16:n?.12:0,ink:n?0:.92}),this.element.ownerDocument?.body&&(this.element.ownerDocument.body.dataset.presentation=this.presentation)}setPresentation(e){if(this.presentation=Em.includes(e)?e:"studio",Cm(Tm,this.presentation),this.applyLook(),this.frame){let t=this.camera.position.clone(),n=this.controls.target.clone();this.setFrame(this.frame,{reset:!0}),this.tween=null,this.camera.position.copy(t),this.controls.target.copy(n),this.controls.update()}return this.presentation}resize(){let e=this.element.clientWidth||1,t=this.element.clientHeight||1;this.renderer.setSize(e,t,!1),this.camera.aspect=e/t,this.camera.updateProjectionMatrix(),this.post?.setSize(e,t,this.pixelRatio);let n=this.renderer.getDrawingBufferSize(new Z);this.effects.setViewport(n.y,this.camera.fov);for(let s of this.lineMaterials??[])s.resolution.copy(n)}point(e,t,n=0){let{rows:s,cols:r,tile_size_m:a}=this.frame.grid;return new R((t+.5-r/2)*a,n*this.unitHeight,(e+.5-s/2)*a)}heightAt(e,t,n=this.frame){e=bi(Math.round(e),0,n.grid.rows-1),t=bi(Math.round(t),0,n.grid.cols-1);let s=n.maps.action[e][t];return s>0&&!n.maps.padding[e][t]?this.piles.endpointHeight(e,t,n===this.piles.previous):s*this.unitHeight}displayHeightAt(e,t){e=bi(Math.round(e),0,this.frame.grid.rows-1),t=bi(Math.round(t),0,this.frame.grid.cols-1);let n=this.displayHeights?.[e]?.[t]??this.frame.maps.action[e][t];return n>0&&!this.frame.maps.padding[e][t]?this.piles.nodeHeight(e*2+1,t*2+1):n*this.unitHeight}surfacePoint(e,t){let n=this.point(e,t);return n.y=this.displayHeightAt(e,t),n}groundAt(e,t){if(!this.frame)return 0;let{rows:n,cols:s,tile_size_m:r}=this.frame.grid,a=Math.floor(e/r+s/2),o=Math.floor(t/r+n/2);return o<0||a<0||o>=n||a>=s?0:this.frame.maps.padding[o][a]?this.obstacleTop??0:this.displayHeightAt(o,a)}buildWorld(e){this.terrain&&this.disposeWorld();let{rows:t,cols:n,tile_size_m:s}=e.grid,r=t*n;this.span=Math.max(t,n)*s,si.uTile.value=s,this.camera.near=Math.max(s*.025,this.span/200),this.camera.far=this.span*20,this.camera.updateProjectionMatrix(),this.controls.minDistance=Math.max(s*2,this.span*.08),this.controls.maxDistance=this.span*4.5,this.applyLook();let a=this.span*.5+Vt.clamp(this.span*.2,6,22)+2;this.sun.position.set(-this.span*.75,this.span*1.35,-this.span*.45),Object.assign(this.sun.shadow.camera,{left:-a,right:a,top:a,bottom:-a,near:.1,far:this.span*4}),this.sun.shadow.camera.updateProjectionMatrix(),this.sun.shadow.normalBias=s*.04,this.fill.position.set(this.span*.8,this.span*.6,this.span*.9),this.post?.configure({span:this.span,tile:s});let o=ki("soil",{polygonOffset:!0,polygonOffsetFactor:1,polygonOffsetUnits:2},this.palette),c=ki("soil",{color:16777215,polygonOffset:!0,polygonOffsetFactor:-1,polygonOffsetUnits:-2},this.palette),l=new Ke({color:9204051,roughness:1});for(let h of[o,c,l])h.shadowSide=Jn;this.terrain=new jt(new Bt(1,1,1),[o,o,c,l,o,o],r),this.terrain.instanceMatrix.setUsage(xi),this.terrain.castShadow=!0,this.terrain.receiveShadow=!0,this.world.add(this.terrain),this.environment=_m(e,{style:this.presentation}),this.world.add(this.environment),this.layers={},this.boundaries={},this.boundaryEntries=new Map,this.layerSettings=vS(this.presentation);for(let[h,d]of Object.entries(this.layerSettings)){let u=new Hn(1,1);u.rotateX(-Math.PI/2);let f=rh({color:d.color,opacity:d.opacity,pattern:d.pattern,polygonOffset:!0,polygonOffsetFactor:-2}),g=new jt(u,f,r);if(g.instanceMatrix.setUsage(xi),g.visible=this.visibility[h],g.renderOrder=3+Object.keys(this.layers).length,g.frustumCulled=!1,g.userData.skipAO=!0,this.layers[h]=g,this.world.add(g),h==="dig"||h==="dump"||h==="interaction"){let _=new zr({color:new Pe(d.color).multiplyScalar(.82),linewidth:2.6,transparent:!0,opacity:.95,depthWrite:!1});_.resolution.copy(this.renderer.getDrawingBufferSize(new Z)),this.lineMaterials.add(_);let p=new jc(new ks,_);p.renderOrder=14,p.visible=this.visibility[h],p.frustumCulled=!1,this.boundaries[h]=p,this.world.add(p)}}this.gridLines=new ns(new ut,new Ni({color:7033138,transparent:!0,opacity:.22,depthWrite:!1})),this.gridLines.renderOrder=10,this.gridLines.visible=this.visibility.grid,this.gridLines.frustumCulled=!1,this.world.add(this.gridLines),this.selection=new ns(new ba(new Bt(s*.99,s*.04,s*.99)),new Ni({color:16776160,depthTest:!1})),this.selection.renderOrder=20,this.selection.visible=!1,this.world.add(this.selection)}disposeWorld(){this.clearMotion(),this.effects.clear();for(let n of this.machines.values())this.scene.remove(n.root),n.dispose();this.machines.clear();let e=new Set,t=new Set;this.world.traverse(n=>{if(n.geometry&&e.add(n.geometry),n.material)for(let s of Array.isArray(n.material)?n.material:[n.material])t.add(s)});for(let n of e)n.dispose();for(let n of t)n.map?.dispose(),n.dispose();this.world.clear(),this.selected=null,this.piles=null,this.obstacleProps=null,this.environment=null,this.lineMaterials.clear()}setFrame(e,{animate:t=!1,duration:n=650,reset:s=!1}={}){let r=this.frame,a=!r||r.grid.rows!==e.grid.rows||r.grid.cols!==e.grid.cols||r.grid.tile_size_m!==e.grid.tile_size_m;this.clearMotion(),this.frame=e,this.unitHeight=e.grid.tile_size_m*.48*this.heightScale,si.uUnit.value=this.unitHeight,(a||s)&&this.buildWorld(e);let o=po(r,e),c=t&&!s&&!a&&!this.reducedMotion&&r&&!r.done&&e.step===r.step+1,l=0;for(let d of e.maps.action)for(let u of d)l=Math.min(l,u);if(this.finalFloor=l*this.unitHeight-e.grid.tile_size_m*.85,c)for(let d of r.maps.action)for(let u of d)l=Math.min(l,u);this.setFloor(l*this.unitHeight-e.grid.tile_size_m*.85),Am(this.obstacleProps),this.obstacleProps=dm(e,{unitHeight:this.unitHeight,style:this.presentation}),this.world.add(this.obstacleProps),Am(this.piles),this.piles=new oh(e,{previous:c?r:null,unitHeight:this.unitHeight,layerSettings:this.layerSettings,visibility:this.visibility,palette:this.palette,roughness:this.presentation==="studio"?.16:0}),this.world.add(this.piles),this.piles.update(c?0:1),this.populate(e);let h=new Set(e.agents.map(d=>d.id));for(let[d,u]of this.machines)h.has(d)||(this.scene.remove(u.root),u.dispose(),this.machines.delete(d));for(let d of e.agents){let u=this.machines.get(d.id);u&&(u.agent.type!==d.type||u.agent.action_type!==d.action_type||u.agent.width!==d.width||u.agent.height!==d.height||u.agent.reach.some((f,g)=>f!==d.reach[g]))&&(this.scene.remove(u.root),u.dispose(),this.machines.delete(d.id),u=null),u||(u=em(d,e.grid.tile_size_m,{style:this.presentation}),u.setTags(this.visibility.tags),this.machines.set(d.id,u),this.scene.add(u.root)),c||(u.lastMove=null),this.poseMachine(u,d,e,1)}if(c){let d=this.actorWork(r,e),u=new Map;for(let _ of d.values()){let p=this.machines.get(_.id);(_.kind==="dig"||_.kind==="dump")&&p?.plan&&(_.plan=p.plan({kind:_.kind,from:_.from,to:_.to,cells:_.cells.map(m=>this.workCell(m,r,e))}));for(let[m,M]of _.plan?.timing??[])u.set(m,M);_.events=this.planEvents(_),_.fired=new Set}let f=[...d.values()].some(_=>_S.includes(_.kind)),g=f?n*1.3:n;this.motion={previous:r,frame:e,facts:o,actors:d,timing:u,start:performance.now(),duration:bi(g,100,900)};for(let _ of o.changed)this.updateCell(_.row,_.col,r.maps.action[_.row][_.col]);this.dirtyInstances(),this.update(performance.now())}this.selected&&this.highlight(this.selected.row,this.selected.col),(a||s)&&this.home({instant:!0})}populate(e){let{rows:t,cols:n,tile_size_m:s}=e.grid,r=[];this.boundaryEntries.clear(),this.gridEntries=new Map,this.displayHeights=e.maps.action.map(l=>[...l]);let a=this.palette.dug.map(l=>new Pe(l)),o=new Pe(this.palette.sand),c=new Pe(this.palette.loose);for(let l=0;l<t;l++)for(let h=0;h<n;h++){let d=l*n+h,u=e.maps.action[l][h];this.updateCell(l,h,u);let f=(l*71+h*29+l*h%47)%31/31;Sm.copy(u<0?a[Math.min(a.length-1,-u-1)]:u>0?c:o).multiplyScalar(.98+f*.04),this.terrain.setColorAt(d,Sm),this.visibility.grid&&(this.gridEntries.set(d,r.length),r.push(...this.flatGridCell(l,h,u)))}this.dirtyInstances(),this.terrain.instanceColor.needsUpdate=!0,this.terrain.computeBoundingSphere(),this.gridLines.geometry.dispose(),this.gridLines.geometry=new ut,this.gridLines.geometry.setAttribute("position",new rt(r,3));for(let[l,h]of Object.entries(this.layers))h.visible=this.visibility[l]&&e.maps[Ws[l].map]!=null;for(let[l,h]of Object.entries(this.boundaries)){let d=[],u=Ws[l].test,f=e.maps[Ws[l].map],g=(_,p)=>f!=null&&_>=0&&_<t&&p>=0&&p<n&&!e.maps.padding[_][p]&&u(f[_][p]);for(let _=0;_<t;_++)for(let p=0;p<n;p++)if(g(_,p)){let m=_*n+p,M=[[_-1,p,[[0,0],[0,1],[0,2]]],[_+1,p,[[2,0],[2,1],[2,2]]],[_,p-1,[[0,0],[1,0],[2,0]]],[_,p+1,[[0,2],[1,2],[2,2]]]];for(let[S,y,T]of M)if(!g(S,y)){this.boundaryEntries.has(m)||this.boundaryEntries.set(m,[]);for(let b of[T[0],T[1],T[1],T[2]]){let P=_*2+b[0],x=p*2+b[1];this.boundaryEntries.get(m).push({name:l,y:d.length+1,row2:P,col2:x}),d.push((x/2-n/2)*s,this.boundaryHeight(_,p,P,x),(P/2-t/2)*s)}}}h.geometry.dispose(),h.geometry=new ks,d.length&&h.geometry.setPositions(d),h.visible=this.visibility[l]&&d.length>0}}flatGridCell(e,t,n){let s=this.frame.grid.tile_size_m,r=this.point(e,t,n),a=r.y+s*.022,o=n>0&&!this.frame.maps.padding[e][t]?0:s/2,c=r.x,l=r.z;return[c-o,a,l-o,c+o,a,l-o,c+o,a,l-o,c+o,a,l+o,c+o,a,l+o,c-o,a,l+o,c-o,a,l+o,c-o,a,l-o]}setFloor(e){this.floor=e,this.environment?.setFloor(e)}boundaryHeight(e,t,n,s){let r=this.displayHeights[e][t];return(r>0?this.piles.nodeHeight(n,s):r*this.unitHeight)+this.frame.grid.tile_size_m*.035}updateCell(e,t,n){let s=this.frame,r=s.grid.tile_size_m,a=e*s.grid.cols+t,o=n>0&&!s.maps.padding[e][t],c=this.point(e,t,n);this.displayHeights[e][t]=n;let l=Math.max(r*.02,(o?0:c.y)-this.floor);Vi.rotation.set(0,0,0),Vi.position.set(c.x,this.floor+l/2,c.z),Vi.scale.set(r,l,r),Vi.updateMatrix(),this.terrain.setMatrixAt(a,Vi.matrix);let h=0;for(let[d,u]of Object.entries(this.layers)){let f=Ws[d],g=s.maps[f.map],_=d==="dig"&&this.presentation==="studio"&&g!=null&&n<=g[e][t],p=!o&&!_&&g!=null&&f.test(g[e][t])&&!s.maps.padding[e][t];Vi.position.set(c.x,c.y+r*(.008+h*.003),c.z),Vi.scale.set(p?r:0,1,p?r:0),Vi.updateMatrix(),u.setMatrixAt(a,Vi.matrix),h++}this.gridEntries.has(a)&&this.gridLines.geometry.attributes.position.array.set(this.flatGridCell(e,t,n),this.gridEntries.get(a))}dirtyInstances(){this.terrain.instanceMatrix.needsUpdate=!0;for(let t of Object.values(this.layers))t.instanceMatrix.needsUpdate=!0;let e=this.frame.grid.cols;for(let[t,n]of this.boundaryEntries)for(let s of n){let r=this.boundaries[s.name].geometry.attributes.instanceStart?.data.array;r&&(r[s.y]=this.boundaryHeight(Math.floor(t/e),t%e,s.row2,s.col2))}for(let t of Object.values(this.boundaries)){let n=t.geometry.attributes.instanceStart?.data;n&&(n.needsUpdate=!0)}this.gridLines.geometry.attributes.position&&(this.gridLines.geometry.attributes.position.needsUpdate=!0)}poseMachine(e,t,n,s,r,a,o="",c=null){let l=r||t,h=o==="turn"?gS(s):s,d=t.position.map((f,g)=>xh(l.position[g],f,s)),u=this.point(d[0],d[1]);u.y=xh(this.displayHeightAt(...l.position),this.displayHeightAt(...t.position),s),e.root.position.copy(u),e.root.rotation.y=xd(l.base_yaw,t.base_yaw,h),e.setPose({...t,previous_loaded:l.loaded,cabin_yaw:xd(l.cabin_yaw,t.cabin_yaw,h),wheel_angle:xh(l.wheel_angle,t.wheel_angle,s)},t.id===n.current_agent,s,o,c),e.drive?.(e.root.position,e.root.rotation.y)}actorWork(e,t){let n=new Map(e.agents.map(o=>[o.id,o])),s=new Map(t.agents.map(o=>[o.id,o.loaded-(n.get(o.id)?.loaded??o.loaded)])),r=new Map(t.agents.map(o=>[o.id,[]]));for(let o of po(e,t).changed){let c=t.agents.filter(d=>o.delta<0?s.get(d.id)>0:s.get(d.id)<0),l=null,h=1/0;for(let d of c.length?c:t.agents){let u=Math.hypot(d.position[0]-o.row,d.position[1]-o.col);u<h&&(h=u,l=d)}r.get(l.id).push(o)}let a=new Map;for(let o of t.agents){let c=n.get(o.id);if(!c)continue;let l=r.get(o.id),h=s.get(o.id),d=t.agents.find(p=>p.id!==o.id&&s.get(p.id)>0&&!r.get(p.id).some(m=>m.delta<0)),u=o.position.some((p,m)=>p!==c.position[m]),f=o.cabin_yaw!==c.cabin_yaw,g=f||o.base_yaw!==c.base_yaw||o.wheel_angle!==c.wheel_angle||o.shovel_lifted!==c.shovel_lifted,_="";l.some(p=>p.delta<0)&&h>0?_="dig":l.some(p=>p.delta>0)&&h<0?_="dump":h<0&&d?_="transfer":l.length?_="terrain":u?_="move":g&&(_="turn"),a.set(o.id,{id:o.id,kind:_,cells:l,load:h,from:c,to:o,swing:_==="turn"&&f&&o.base_yaw===c.base_yaw,recipient:_==="transfer"?d:null})}return a}workCell(e,t,n){let s=this.point(e.row,e.col),r=n.maps.padding[e.row][e.col],a=(o,c)=>o>0&&!r?this.piles.endpointHeight(e.row,e.col,c):o*this.unitHeight;return{key:e.row*n.grid.cols+e.col,row:e.row,col:e.col,delta:e.delta,x:s.x,z:s.z,before:a(t.maps.action[e.row][e.col],!0),after:a(n.maps.action[e.row][e.col],!1)}}planEvents(e){let t=[],{kind:n,plan:s,from:r,to:a}=e;if(!n||n==="terrain")return t;let o=a.position.some((l,h)=>l!==r.position[h]);t.push({at:0,once:"exhaust-start"}),o&&t.push({from:.05,to:.9,stream:"tracks",rate:16});let c=l=>e.cells.filter(h=>l<0?h.delta<0:h.delta>0);if(n==="dig"){let l=s?.events.bite??(a.type===2?.38:.32),[h,d]=s?.events.drag??[l,l+.22];t.push({at:l,once:"bite",cells:c(-1)}),t.push({from:h,to:d,stream:"scoop",cells:c(-1),rate:s?34:70}),s?.events.breakout&&t.push({at:s.events.breakout,once:"spill"})}else if(n==="dump"){let[l,h]=s?.events.pour??(a.type===1?[.32,.72]:a.type===2?[.34,.62]:[.44,.74]);t.push({from:l,to:h,stream:a.type===1?"bed":"pour",cells:c(1),rate:s?230:60}),t.push({at:Math.min(.95,(l+h)/2+.1),once:"landing",cells:c(1)})}else n==="transfer"&&t.push({from:.44,to:.72,stream:"transfer",rate:55});return t}centroid(e){let t=new R;if(!e?.length)return null;for(let n of e)t.add(this.surfacePoint(n.row,n.col));return t.multiplyScalar(1/e.length)}runEvents(e,t,n){let s=e.frame.grid.tile_size_m,r=this.effects,a=this.presentation==="studio";for(let o of e.actors.values()){let c=this.machines.get(o.id);if(!(!c||!o.events.length)){c.root.updateMatrixWorld(!0);for(let[l,h]of o.events.entries())if(h.once){if(o.fired.has(l)||t<h.at)continue;if(o.fired.add(l),h.once==="exhaust-start")a||r.puff(c.exhaust(),{count:4,size:s*.28,rise:1.4,spread:s*.15,color:6185835,life:1.1});else if(h.once==="bite"){let d=a?c.teeth():this.centroid(h.cells)??c.tip();r.burst(d,{count:a?9:14,speed:a?1.6:2.4,size:s*(a?.06:.09)}),r.puff(d,{count:7,size:s*.38,spread:s*.6,rise:.5}),r.dust(d,{count:6,size:s*1.1,spread:s*.5,life:1.6})}else if(h.once==="landing"){let d=this.centroid(h.cells);d&&(r.puff(d,{count:8,size:s*.42,spread:s*.7,rise:.4}),r.dust(d,{count:9,size:s*1.5,spread:s*.8,life:2}))}else h.once==="spill"&&r.throwClods(c.lip(),c.lip().setY(this.groundAt(c.lip().x,c.lip().z)),{count:5,flight:.4,spread:s*.3,size:s*.06})}else if(t>=h.from&&t<=h.to){h.carry=(h.carry??0)+h.rate*n;let d=Math.floor(h.carry);for(h.carry-=d;d-- >0;)this.emitStream(h,c,e,s)}}}}emitStream(e,t,n,s){let r=this.effects,a=this.presentation==="studio",o=c=>c?.length?this.surfacePoint(...Object.values(c[Math.floor(Math.random()*c.length)]).slice(0,2)):null;if(e.stream==="tracks"){let c=n.frame.agents.find(h=>h.id===t.agent.id),l=new R(-t.agent.height*s*.45,0,(Math.random()<.5?-1:1)*t.agent.width*s*.35).applyAxisAngle(new R(0,1,0),t.root.rotation.y).add(t.root.position);c&&Math.random()<.5&&(r.puff(l,{count:1,size:s*.3,spread:s*.2,rise:.35,life:.8}),Math.random()<.4&&r.dust(l,{count:1,size:s*.9,spread:s*.3,life:1.3,opacity:.16})),Math.random()<.25&&r.puff(t.exhaust(),{count:1,size:s*.2,rise:1.3,spread:s*.08,color:6975351,life:1})}else if(e.stream==="scoop")if(a){let c=t.teeth(),l=c.clone().add(new R((Math.random()-.5)*s*1.2,0,(Math.random()-.5)*s*1.2));l.y=this.groundAt(l.x,l.z),r.throwClods(c.clone().add(new R(0,s*.12,0)),l,{count:1,flight:.28,spread:s*.15,size:s*.055,jitter:s*.2}),Math.random()<.12&&r.dust(c,{count:1,size:s*.8,spread:s*.3,life:1.2,opacity:.14})}else{let c=o(e.cells);c&&r.throwClods(c,t.tip(),{count:1,flight:.22,spread:0,size:s*.08,settle:!1,jitter:s*.3})}else if(e.stream==="pour"||e.stream==="bed"){let c=o(e.cells);if(!c)return;let l=e.stream==="bed"?t.bedLip():a?t.lip():t.tip();r.throwClods(l,c,{count:1,flight:a?.42:.34,spread:s*.45,size:s*(a?.06:.095),jitter:s*(a?.22:.12)}),a&&Math.random()<.06&&r.dust(l,{count:1,size:s*.9,spread:s*.3,life:1.4,opacity:.12})}else if(e.stream==="transfer"){let c=this.machines.get(n.actors.get(t.agent.id)?.recipient?.id);if(!c)return;c.root.updateMatrixWorld(!0),r.throwClods(t.tip(),c.tip(),{count:1,flight:.3,spread:s*.2,size:s*.09,settle:!1,jitter:s*.1})}}update(e){let t=Math.min(.1,Math.max(0,(e-(this.clock?.last??e))/1e3));if(this.clock&&(this.clock.last=e),si.uTime.value=e/1e3,!this.frame)return;this.measure(t),this.environment?.update(e/1e3);let n=new Map;if(this.motion){let{previous:r,frame:a,facts:o,start:c,duration:l,actors:h,timing:d}=this.motion,u=bi((e-c)/l,0,1),f=bm(u),g=a.grid.cols,_=(p,m)=>{let M=d.get(p*g+m);return M?bm(bi((u-M[0])/(M[1]-M[0]),0,1)):f};o.changed.length&&this.piles.update(d.size?_:f);for(let p of o.changed)this.updateCell(p.row,p.col,xh(r.maps.action[p.row][p.col],a.maps.action[p.row][p.col],_(p.row,p.col)));for(let p of a.agents){let m=r.agents.find(y=>y.id===p.id),M=h.get(p.id),S=M?.kind==="terrain"?"":M?.kind??"";if(!S&&[...h.values()].some(y=>y.recipient?.id===p.id)&&(S="receive"),(S==="turn"||S==="move")&&(S=m&&p.position.some((y,T)=>y!==m.position[T])?"move":"turn"),this.poseMachine(this.machines.get(p.id),p,a,M?.plan?u:f,m,r,S,M?.plan),m&&p.position.some((y,T)=>y!==m.position[T])){let y=new Z(Math.cos(p.base_yaw),Math.sin(p.base_yaw)),T=new Z(p.position[1]-m.position[1],p.position[0]-m.position[0]);n.set(p.id,{move:u,direction:Math.sign(y.x*T.x-y.y*T.y)||1})}}o.changed.length&&(this.dirtyInstances(),this.selected&&this.highlight(this.selected.row,this.selected.col)),this.reducedMotion||this.runEvents(this.motion,u,t),u>=1&&(this.clearMotion(),this.floor!==this.finalFloor&&(this.setFloor(this.finalFloor),this.populate(this.frame)))}let s=e/1e3;for(let r of this.machines.values())r.tick?.(s,{...n.get(r.agent.id)||{},reducedMotion:this.reducedMotion});if(this.clock.idle-=t,!this.reducedMotion&&this.clock.idle<=0){this.clock.idle=1.3+Math.random()*.8;let r=this.machines.get(this.frame.current_agent),a=this.frame.grid.tile_size_m;r&&!this.frame.done&&(r.root.updateMatrixWorld(!0),this.effects.puff(r.exhaust(),{count:1,size:a*.18,rise:1.1,spread:a*.05,color:7764867,life:1.2}))}if(this.effects.update(t),this.tween){let{from:r,to:a,start:o,duration:c}=this.tween,l=mS(bi((e-o)/c,0,1));this.camera.position.lerpVectors(r.position,a.position,l),this.controls.target.lerpVectors(r.target,a.target,l),l>=1&&(this.tween=null)}if(this.follow){let r=this.machines.get(this.frame.current_agent);if(r){let a=r.root.position.clone();a.y+=this.frame.grid.tile_size_m;let o=a.sub(this.controls.target).multiplyScalar(.055);this.camera.position.add(o),this.controls.target.add(o)}}}clearMotion(){if(this.motion)for(let e of this.motion.frame.agents){let t=this.machines.get(e.id);t&&t.tick?.(performance.now()/1e3,{reducedMotion:this.reducedMotion})}this.motion=null}setLayer(e,t){if(this.visibility[e]=t,e==="tags"){for(let n of this.machines.values())n.setTags(t);return}this.frame&&(this.piles?.setLayer(e,t),e==="grid"?(this.gridLines.visible=t,this.populate(this.frame)):this.layers[e]&&(this.layers[e].visible=t&&this.frame.maps[Ws[e].map]!=null),this.boundaries[e]&&(this.boundaries[e].visible=t))}setHeight(e){this.heightScale=e,this.frame&&this.setFrame(this.frame)}flyTo(e,t,{instant:n=!1}={}){if(n||this.reducedMotion){this.tween=null,this.camera.position.copy(e),this.controls.target.copy(t),this.controls.update();return}this.tween={from:{position:this.camera.position.clone(),target:this.controls.target.clone()},to:{position:e,target:t},start:performance.now(),duration:750}}home({instant:e=!1}={}){if(!this.frame)return;this.follow=!1;let t=this.camera.aspect,n=this.presentation==="diorama"?1.75:1.45,s=this.span*(t<1?n/t:n);this.flyTo(new R(s*.72,s*.66,s*.84),new R(0,-this.span*.04,0),{instant:e}),this.onCameraChange?.({view:"home",follow:!1})}top(){this.frame&&(this.follow=!1,this.camera.up.set(0,1,0),this.flyTo(new R(0,this.span*(this.presentation==="diorama"?1.92:1.62)/Math.min(this.camera.aspect,1),this.span*.001),new R(0,0,0)),this.onCameraChange?.({view:"top",follow:!1}))}setFollow(e){if(this.follow=e,this.tween=null,e&&this.frame){let t=this.machines.get(this.frame.current_agent);if(t){let n=this.frame.agents.find(d=>d.id===this.frame.current_agent),s=this.frame.grid.tile_size_m,r=t.root.position.clone();r.y+=s;let a=Vt.degToRad(this.camera.fov),o=2*Math.atan(Math.tan(a/2)*this.camera.aspect),c=Math.max(n.reach[1],Math.hypot(n.width,n.height)*.65)*s,l=Math.max(this.span*.5,c*1.15/Math.sin(Math.min(a,o)/2)),h=this.camera.position.clone().sub(this.controls.target).normalize().multiplyScalar(l);this.flyTo(r.clone().add(h),r)}}this.onCameraChange?.({follow:e})}pick(e){if(!this.terrain)return;let t=this.renderer.domElement.getBoundingClientRect();this.pointer.set((e.clientX-t.left)/t.width*2-1,-(e.clientY-t.top)/t.height*2+1),this.camera.updateMatrixWorld(),this.world.updateMatrixWorld(!0),this.raycaster.setFromCamera(this.pointer,this.camera);let n=[this.terrain,this.obstacleProps];this.piles.surface.visible&&n.push(this.piles.surface);let s=this.raycaster.intersectObjects(n.filter(Boolean),!0)[0],r;if(s?.object===this.piles.surface)r=this.piles.cellForHit(s);else if(s?.object===this.terrain&&s.instanceId!==void 0)r={row:Math.floor(s.instanceId/this.frame.grid.cols),col:s.instanceId%this.frame.grid.cols};else if(s){let{rows:a,cols:o,tile_size_m:c}=this.frame.grid;r={row:bi(Math.floor(s.point.z/c+a/2),0,a-1),col:bi(Math.floor(s.point.x/c+o/2),0,o-1)}}r&&(this.highlight(r.row,r.col),this.onPick?.(r))}highlight(e,t){if(e>=this.frame.grid.rows||t>=this.frame.grid.cols){this.selected=null,this.selection.visible=!1;return}this.selected={row:e,col:t},this.selection.position.copy(this.surfacePoint(e,t)),this.selection.position.y+=this.frame.grid.tile_size_m*.03,this.selection.visible=!0}capture({scale:e=2}={}){let t=this.element.clientWidth||1,n=this.element.clientHeight||1,s=Math.min(Math.max(e,this.pixelRatio),this.renderer.capabilities.maxTextureSize/Math.max(t,n));this.renderer.setPixelRatio(s),this.renderer.setSize(t,n,!1),this.post?.setSamples(0),this.post?.setSize(t,n,s);let r=this.renderer.getDrawingBufferSize(new Z);for(let a of this.lineMaterials)a.resolution.copy(r);try{return this.render(),{url:this.renderer.domElement.toDataURL("image/png"),width:r.x,height:r.y}}finally{this.renderer.setPixelRatio(this.pixelRatio),this.post?.setSamples(4),this.resize()}}};/*!
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
 */var De=i=>document.getElementById(i),Xs=i=>new Intl.NumberFormat(void 0,{maximumFractionDigits:2}).format(i),wt=(i,e)=>{De(i).textContent=e},Tt,ct,Rt=0,Wi="replay",Fm=null,Ms=!1,Pt=!1,Ei=!1,So=null,Ed,yh=0,Om=De("terra-replay"),Bm=!!Om?.textContent.trim();function qs(i){De("loading").hidden=!0,wt("error-message",i?.message||String(i)),De("error").hidden=!1}function wi(i,e=3100){clearTimeout(Ed),wt("event",i),De("event").classList.add("visible"),Ed=setTimeout(()=>De("event").classList.remove("visible"),e)}function qi(){return ct?.frames[Rt]}function zm(){return!!ct&&Wi==="manual"&&!Ms&&!Pt&&!Ei&&Rt===ct.frames.length-1&&!qi().done}function ri(){let i=zm(),e=qi();document.querySelectorAll("[data-action]").forEach(n=>{n.disabled=!i}),De("reset").disabled=Pt||Wi!=="manual"||Ms,De("export").disabled=!ct||Pt,De("screenshot").disabled=!Tt||!ct,De("open-file").disabled=Pt,De("previous").disabled=!ct||Rt<=0||Pt,De("next").disabled=!ct||Rt>=ct.frames.length-1||Pt,De("play").disabled=!ct||ct.frames.length<2||Pt,De("seek").disabled=!ct||ct.frames.length<2||Pt,De("play").textContent=Ei?"\u2161":"\u25B6",De("play").setAttribute("aria-label",Ei?"Pause replay":"Play replay"),De("resume-live").hidden=!Fm||Bm||!Ms&&!(Wi==="manual"&&ct&&Rt<ct.frames.length-1),De("resume-live").disabled=Pt;let t=ct&&Rt<ct.frames.length-1;wt("manual-status",Pt?"STEPPING\u2026":Ms||Wi!=="manual"?"REPLAY ONLY":t?"HISTORY":e?.done?"ENDED":Ei?"PLAYBACK":"LIVE"),wt("session-mode",Ms?"Imported replay":Wi==="manual"?t?"Manual \xB7 history":"Manual session":"Replay session")}function Ad(){if(!So||!qi())return;let{row:i,col:e}=So,t=qi(),{maps:n}=t;if(i>=t.grid.rows||e>=t.grid.cols){So=null;return}let s=De("cell-inspector");s.replaceChildren();let r=document.createElement("span");r.className="eyebrow",r.textContent="CELL INSPECTOR",s.append(r);let a=document.createElement("div");a.className="cell-heading",a.textContent=`ROW ${i}  \xB7  COL ${e}`,s.append(a);let o=document.createElement("div");o.className="cell-data",s.append(o);let c=(d,u,f)=>n[d]==null?"Unavailable":n[d][i][e]?u:f,l=n.target[i][e],h=[["Raw soil height",`${n.action[i][e]} units`],["Target",l<0?`Dig ${-l}`:l>0?`Dump ${l}`:"Neutral"],["Obstacle",n.padding[i][e]?"Yes":"No"],["Static dumping",c("dumpability_static","Allowed","Prohibited")],["Dumpable now",c("dumpability","Yes","No")],["Workspace",c("interaction","Inside","Outside")],["Traversability feature",n.traversability==null?"Unavailable":{"-1":"Occupied (\u22121)",0:"Clear (0)",1:"Blocked (1)"}[Number(n.traversability[i][e])]]];for(let[d,u]of h){let f=document.createElement("span"),g=document.createElement("strong");f.textContent=d,g.textContent=u,o.append(f,g)}}function yS(){let i=qi(),e=i.agents.find(o=>o.id===i.current_agent),t=sm(i);wt("title",ct.metadata.title),wt("source",ct.metadata.source),wt("grid-spec",`${i.grid.rows} \xD7 ${i.grid.cols} \xB7 ${Xs(i.grid.tile_size_m)} m / cell`),wt("agent-count",`${i.agents.length} machine${i.agents.length===1?"":"s"}`),wt("agent-id",String(e.id+1).padStart(2,"0")),wt("machine-name",md[e.type]),wt("embodiment",e.action_type===1?"Wheeled":"Tracked"),De("load").replaceChildren(document.createTextNode(Xs(e.loaded)));let n=document.createElement("small");n.textContent=" units",De("load").append(n),wt("reward",rm(i.reward)),wt("outcome",i.task_done?"Task complete":i.done?"Episode ended \xB7 task incomplete":`Ready \xB7 machine ${e.id+1} acts next`),De("outcome").classList.toggle("done",i.done);let s=De("agent-list");s.replaceChildren(),s.hidden=i.agents.length<=1;for(let o of i.agents){let c=document.createElement("span");c.className=`agent-tag${o.id===i.current_agent?" active":""}`,c.textContent=`${String(o.id+1).padStart(2,"0")} ${md[o.type]} \xB7 ${o.loaded}`,s.append(c)}wt("cut-units",`${Xs(t.cut)} units`),wt("fill-units",`${Xs(t.fill)} units`),wt("scene-caption",`${Xs(i.grid.cols*i.grid.tile_size_m)} \xD7 ${Xs(i.grid.rows*i.grid.tile_size_m)} m worksite \xB7 illustrative soil mounds`),wt("step",i.step),wt("frame-count",`${Rt+1} / ${ct.frames.length}`),wt("action-label",_d(i,ct.frames[Rt-1])),De("seek").max=String(ct.frames.length-1),De("seek").value=String(Rt),De("seek").setAttribute("aria-valuetext",`Snapshot ${Rt+1} of ${ct.frames.length}, step ${i.step}`);let r=De("left-action"),a=De("right-action");r.dataset.action=e.action_type===1?"2":"3",a.dataset.action=e.action_type===1?"3":"2",r.querySelector(".turn-label").textContent=e.action_type===1?"Steer left":"Turn left",a.querySelector(".turn-label").textContent=e.action_type===1?"Steer right":"Turn right",r.title=`${e.action_type===1?"Steer left":"Turn anticlockwise"} \xB7 Left or A`,a.title=`${e.action_type===1?"Steer right":"Turn clockwise"} \xB7 Right or D`,wt("work-label",e.type===2?e.shovel_lifted?"Lower shovel / dump":"Lift shovel":e.loaded>0?"Dump / transfer soil":e.type===1?"Dump (empty)":"Dig soil");for(let[o,c]of[["interaction","interaction"],["restricted","dumpability_static"],["dumpability","dumpability"]]){let l=document.querySelector(`[data-layer="${o}"]`);l.disabled=i.maps[c]==null,l.closest("label").title=i.maps[c]==null?"This diagnostic layer is unavailable in the recording.":""}Ad(),ri()}function Xi(i,{animate:e=!1,reset:t=!1,announce:n=!1}={}){if(!ct)return;let s=Rt;Rt=Math.max(0,Math.min(ct.frames.length-1,i));let r=qi();if(Tt.setFrame(r,{animate:e&&Rt===s+1,reset:t,duration:Math.min(650,800/Number(De("speed").value))}),yS(),n&&Rt>0){let a=ct.frames[Rt-1];r.step>a.step&&!a.done?wi(`${_d(r,a)} \xB7 ${po(a,r).message}`):wi("Episode boundary \xB7 initial snapshot")}else t&&(clearTimeout(Ed),De("event").classList.remove("visible"))}function Gi(i){Ei=i,yh=performance.now(),ri()}function Pm(){!ct||ct.frames.length<2||Pt||(!Ei&&Rt===ct.frames.length-1&&Xi(0),Gi(!Ei))}function km(i){Ei&&!document.hidden&&i-yh>=900/Number(De("speed").value)&&(yh=i,Rt<ct.frames.length-1&&Xi(Rt+1,{animate:!0,announce:!0}),Rt>=ct.frames.length-1&&Gi(!1)),requestAnimationFrame(km)}async function Mh(i,e){let t=await fetch(i,e===void 0?{cache:"no-store"}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(e)}),n=await t.json().catch(()=>{throw new Error(`The server returned an unreadable response (${t.status}).`)});if(!t.ok)throw new Error(n.error||`Request failed (${t.status}).`);return n}function bo(i,{local:e=!1}={}){if(im(i.replay),!["manual","replay"].includes(i.mode))throw new Error("Unknown viewer session mode.");ct=i.replay,Wi=i.mode,Ms=e,Rt=0,So=null,Ei=!1,e||(Fm=i.mode),De("cell-inspector").replaceChildren();let t=document.createElement("span");t.className="eyebrow",t.textContent="CELL INSPECTOR";let n=document.createElement("span");n.className="cell-hint",n.textContent="Click the terrain to inspect a cell",De("cell-inspector").append(t,n),Xi(Wi==="manual"&&!e?ct.frames.length-1:0,{reset:!0}),De("loading").hidden=!0,De("error").hidden=!0}async function Im(i){if(zm()){Pt=!0,ri();try{let{frame:e}=await Mh("/api/action",{action:i});gd(e),ct.frames.push(e),Xi(ct.frames.length-1,{animate:!0,announce:!0})}catch(e){qs(e)}finally{Pt=!1,ri()}}}async function Dm(){if(!(Pt||Wi!=="manual"||Ms)){Pt=!0,Gi(!1),ri();try{bo(await Mh("/api/reset",{})),wi("Episode reset \xB7 ready to play")}catch(i){qs(i)}finally{Pt=!1,ri()}}}function Hm(i,e,t=!1){let n=document.createElement("a");n.href=i,n.download=e,document.body.append(n),n.click(),n.remove(),t&&setTimeout(()=>URL.revokeObjectURL(i),1e3)}function MS(){if(!ct)return;let i=new Blob([JSON.stringify(ct)],{type:"application/json"});Hm(URL.createObjectURL(i),"terra-replay.json",!0),wi(`Exported ${ct.frames.length} recorded snapshots`)}function SS(){document.querySelectorAll("[data-action]").forEach(i=>i.addEventListener("click",e=>{Im(Number(i.dataset.action)),e.detail>0&&De("viewport").focus({preventScroll:!0})})),De("reset").addEventListener("click",i=>{Dm(),i.detail>0&&De("viewport").focus({preventScroll:!0})}),De("previous").addEventListener("click",()=>{Gi(!1),Xi(Rt-1)}),De("next").addEventListener("click",()=>{Gi(!1),Xi(Rt+1,{animate:!0,announce:!0})}),De("play").addEventListener("click",Pm),De("seek").addEventListener("input",()=>{Gi(!1),Xi(Number(De("seek").value))}),De("speed").addEventListener("change",()=>{yh=performance.now()}),De("camera-home").addEventListener("click",()=>Tt?.home()),De("brand-home").addEventListener("click",i=>{i.preventDefault(),Tt?.home()}),De("camera-top").addEventListener("click",()=>Tt?.top()),De("camera-follow").addEventListener("click",()=>Tt?.setFollow(!Tt.follow)),De("quality").addEventListener("click",Um),De("presentation").addEventListener("click",Lm),De("height-scale").addEventListener("input",()=>{let i=Number(De("height-scale").value);wt("height-value",`${Xs(i)}\xD7`),Tt?.setHeight(i)}),document.querySelectorAll("[data-layer]").forEach(i=>i.addEventListener("change",()=>{Tt?.setLayer(i.dataset.layer,i.checked),Nm()})),De("layers-toggle").addEventListener("click",()=>{let i=[...document.querySelectorAll("[data-layer]")].filter(t=>!t.disabled),e=!i.some(t=>t.checked);for(let t of i)t.checked=e,Tt?.setLayer(t.dataset.layer,e);Nm()}),De("export").addEventListener("click",MS),De("screenshot").addEventListener("click",()=>{try{let i=Tt.capture();Hm(i.url,`terra-step-${qi().step}.png`),wi(`Scene captured \xB7 ${i.width} \xD7 ${i.height} PNG`)}catch(i){qs(i)}}),De("open-file").addEventListener("click",()=>De("replay-file").click()),De("replay-file").addEventListener("change",async i=>{let e=i.target.files[0];if(e){if(Pt){i.target.value="",wi("Wait for the current action before opening a replay.");return}Pt=!0,Gi(!1),ri();try{if(e.size>256*1024*1024)throw new Error("Please use a JSON recording smaller than 256 MB. Large recordings can be opened through Python with --replay.");let t=JSON.parse(await e.text());bo({mode:"replay",replay:t},{local:!0}),wi(`Opened ${e.name}`)}catch(t){qs(t)}finally{i.target.value="",Pt=!1,ri()}}}),De("resume-live").addEventListener("click",async()=>{if(!Pt){Pt=!0,Gi(!1),ri();try{bo(await Mh("/api/session"))}catch(i){qs(i)}finally{Pt=!1,ri()}}}),De("dismiss-error").addEventListener("click",()=>{De("error").hidden=!0}),document.addEventListener("keydown",i=>{if(i.ctrlKey||i.metaKey||i.altKey||i.repeat||["INPUT","SELECT","TEXTAREA","BUTTON"].includes(i.target.tagName)||i.target.isContentEditable||!De("error").hidden)return;let e=i.key.toLowerCase();if(e==="g"){i.preventDefault(),Um();return}if(e==="p"){i.preventDefault(),Lm();return}if(e==="h"){i.preventDefault(),Tt?.home();return}if(e==="t"){i.preventDefault(),Tt?.top();return}if(e==="f"){i.preventDefault(),Tt?.setFollow(!Tt.follow);return}if(!ct||Pt)return;if(Wi!=="manual"||Ms||Rt<ct.frames.length-1||Ei){e===" "&&(i.preventDefault(),Pm()),(e==="arrowleft"||e==="arrowright")&&(i.preventDefault(),Gi(!1),Xi(Rt+(e==="arrowright"?1:-1)));return}if(e==="r"){i.preventDefault(),Dm();return}let t=qi().agents.find(a=>a.id===qi().current_agent),n=t.action_type===1?2:3,s=t.action_type===1?3:2,r={arrowup:0,w:0,arrowdown:1,s:1,arrowleft:n,a:n,arrowright:s,d:s,q:5,e:4," ":6,n:7};e in r&&(i.preventDefault(),Im(r[e]))})}function wd(i){De("quality").setAttribute("aria-pressed",String(i==="high"))}var Td={studio:["Studio","Studio style \xB7 earth block on a studio floor"],paper:["Paper","Paper style \xB7 plain figure look"],diorama:["Diorama","Diorama style \xB7 stylized island"]};function Vm(i){De("presentation").querySelector("span").textContent=Td[i][0]}function Lm(){if(!Tt)return;let i=Object.keys(Td),e=Tt.setPresentation(i[(i.indexOf(Tt.presentation)+1)%i.length]);Vm(e),Ad(),wi(Td[e][1])}function Um(){if(!Tt)return;let i=Tt.setQuality(Tt.quality==="high"?"fast":"high");wd(i),wi(i==="high"?"Rich lighting on \xB7 ambient occlusion and outlines":"Fast graphics \xB7 plain lighting")}function Nm(){wt("layers-toggle",[...document.querySelectorAll("[data-layer]")].some(i=>i.checked&&!i.disabled)?"Hide all":"Show all")}async function bS(){SS(),ri();try{Tt=new vh(De("viewport"),{onPick:i=>{So=i,Ad()},onCameraChange:({view:i,follow:e})=>{i&&(De("camera-home").classList.toggle("selected",i==="home"),De("camera-top").classList.toggle("selected",i==="top")),e!==void 0&&De("camera-follow").setAttribute("aria-pressed",String(e))},onError:qs,onQualityChange:i=>{wd(i),wi("Switched to fast graphics for smoother motion \xB7 press G to restore")}}),wd(Tt.quality),Vm(Tt.presentation),window.terraViewer={scene:Tt,show:(i,e)=>Xi(i,e)},Bm?bo({mode:"replay",replay:JSON.parse(Om.textContent)},{local:!0}):bo(await Mh("/api/session")),requestAnimationFrame(km)}catch(i){qs(i),wt("session-mode","Unavailable"),wt("title","Open a Terra worksite"),wt("source","Check the error message to continue.")}}bS();})();
