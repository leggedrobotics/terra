(()=>{/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */var rs={LEFT:0,MIDDLE:1,RIGHT:2,ROTATE:0,DOLLY:1,PAN:2},as={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},of=0,Qh=1,lf=2;var Ds=1,cf=2,Ar=3,ji=0,Qt=1,xi=2,zt=0,As=1,eu=2,tu=3,iu=4,Fl=5;var Ti=100,hf=101,uf=102,df=103,ff=104,Ls=200,pf=201,mf=202,gf=203,Zo=204,Jo=205,Ba=206,_f=207,za=208,xf=209,vf=210,yf=211,Mf=212,bf=213,Sf=214,Ko=0,jo=1,Qo=2,Rs=3,el=4,tl=5,il=6,nl=7,Ol=0,Ef=1,wf=2,rn=0,ka=1,Ha=2,Va=3,os=4,Ga=5,Wa=6,Ns=7;var nu=300,ls=301,Us=302,Bl=303,zl=304,Xa=306,zi=1e3,mn=1001,sl=1002,Ot=1003,Tf=1004;var qa=1005;var jt=1006,kl=1007;var cs=1008;var li=1009,su=1010,ru=1011,Rr=1012,Hl=1013,an=1014,Vi=1015,ei=1016,Vl=1017,Gl=1018,hs=1020,au=35902,ou=35899,lu=1021,cu=1022,vi=1023,_n=1026,xn=1027,Wl=1028,Xl=1029,us=1030,ql=1031;var Yl=1033,Ya=33776,$a=33777,Za=33778,Ja=33779,$l=35840,Zl=35841,Jl=35842,Kl=35843,jl=36196,Ql=37492,ec=37496,tc=37488,ic=37489,Ka=37490,nc=37491,sc=37808,rc=37809,ac=37810,oc=37811,lc=37812,cc=37813,hc=37814,uc=37815,dc=37816,fc=37817,pc=37818,mc=37819,gc=37820,_c=37821,xc=36492,vc=36494,yc=36495,Mc=36283,bc=36284,ja=36285,Sc=36286;var ta=2300,rl=2301,$o=2302,zh=2303,kh=2400,Hh=2401,Vh=2402;var Af=3200;var Cr=0,Rf=1,Un="",Ft="srgb",ia="srgb-linear",na="linear",ft="srgb";var Es=7680;var Gh=519,Cf=512,Pf=513,If=514,Ec=515,Df=516,Lf=517,wc=518,Nf=519,al=35044,Pr=35048;var hu="300 es",Ki=2e3,dr=2001;function Am(n){for(let e=n.length-1;e>=0;--e)if(n[e]>=65535)return!0;return!1}function Rm(n){return ArrayBuffer.isView(n)&&!(n instanceof DataView)}function sa(n){return document.createElementNS("http://www.w3.org/1999/xhtml",n)}function Uf(){let n=sa("canvas");return n.style.display="block",n}var Md={},fr=null;function ra(...n){let e="THREE."+n.shift();fr?fr("log",e,...n):console.log(e,...n)}function Ff(n){let e=n[0];if(typeof e=="string"&&e.startsWith("TSL:")){let t=n[1];t&&t.isStackTrace?n[0]+=" "+t.getLocation():n[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return n}function Ye(...n){n=Ff(n);let e="THREE."+n.shift();if(fr)fr("warn",e,...n);else{let t=n[0];t&&t.isStackTrace?console.warn(t.getError(e)):console.warn(e,...n)}}function $e(...n){n=Ff(n);let e="THREE."+n.shift();if(fr)fr("error",e,...n);else{let t=n[0];t&&t.isStackTrace?console.error(t.getError(e)):console.error(e,...n)}}function Ts(...n){let e=n.join(" ");e in Md||(Md[e]=!0,Ye(...n))}function Of(n,e,t){return new Promise(function(i,s){function r(){switch(n.clientWaitSync(e,n.SYNC_FLUSH_COMMANDS_BIT,0)){case n.WAIT_FAILED:s();break;case n.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:i()}}setTimeout(r,t)})}var Bf={[Ko]:jo,[Qo]:il,[el]:nl,[Rs]:tl,[jo]:Ko,[il]:Qo,[nl]:el,[tl]:Rs},Qi=class{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});let i=this._listeners;i[e]===void 0&&(i[e]=[]),i[e].indexOf(t)===-1&&i[e].push(t)}hasEventListener(e,t){let i=this._listeners;return i===void 0?!1:i[e]!==void 0&&i[e].indexOf(t)!==-1}removeEventListener(e,t){let i=this._listeners;if(i===void 0)return;let s=i[e];if(s!==void 0){let r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){let t=this._listeners;if(t===void 0)return;let i=t[e.type];if(i!==void 0){e.target=this;let s=i.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,e);e.target=null}}},ai=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],bd=1234567,hr=Math.PI/180,pr=180/Math.PI;function gn(){let n=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(ai[n&255]+ai[n>>8&255]+ai[n>>16&255]+ai[n>>24&255]+"-"+ai[e&255]+ai[e>>8&255]+"-"+ai[e>>16&15|64]+ai[e>>24&255]+"-"+ai[t&63|128]+ai[t>>8&255]+"-"+ai[t>>16&255]+ai[t>>24&255]+ai[i&255]+ai[i>>8&255]+ai[i>>16&255]+ai[i>>24&255]).toLowerCase()}function Ke(n,e,t){return Math.max(e,Math.min(t,n))}function uu(n,e){return(n%e+e)%e}function Cm(n,e,t,i,s){return i+(n-e)*(s-i)/(t-e)}function Pm(n,e,t){return n!==e?(t-n)/(e-n):0}function jr(n,e,t){return(1-t)*n+t*e}function Im(n,e,t,i){return jr(n,e,1-Math.exp(-t*i))}function Dm(n,e=1){return e-Math.abs(uu(n,e*2)-e)}function Lm(n,e,t){return n<=e?0:n>=t?1:(n=(n-e)/(t-e),n*n*(3-2*n))}function Nm(n,e,t){return n<=e?0:n>=t?1:(n=(n-e)/(t-e),n*n*n*(n*(n*6-15)+10))}function Um(n,e){return n+Math.floor(Math.random()*(e-n+1))}function Fm(n,e){return n+Math.random()*(e-n)}function Om(n){return n*(.5-Math.random())}function Bm(n){n!==void 0&&(bd=n);let e=bd+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function zm(n){return n*hr}function km(n){return n*pr}function Hm(n){return(n&n-1)===0&&n!==0}function Vm(n){return Math.pow(2,Math.ceil(Math.log(n)/Math.LN2))}function Gm(n){return Math.pow(2,Math.floor(Math.log(n)/Math.LN2))}function Wm(n,e,t,i,s){let r=Math.cos,a=Math.sin,o=r(t/2),c=a(t/2),l=r((e+i)/2),h=a((e+i)/2),d=r((e-i)/2),u=a((e-i)/2),f=r((i-e)/2),g=a((i-e)/2);switch(s){case"XYX":n.set(o*h,c*d,c*u,o*l);break;case"YZY":n.set(c*u,o*h,c*d,o*l);break;case"ZXZ":n.set(c*d,c*u,o*h,o*l);break;case"XZX":n.set(o*h,c*g,c*f,o*l);break;case"YXY":n.set(c*f,o*h,c*g,o*l);break;case"ZYZ":n.set(c*g,c*f,o*h,o*l);break;default:Ye("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function Ji(n,e){switch(e.constructor){case Float32Array:return n;case Uint32Array:return n/4294967295;case Uint16Array:return n/65535;case Uint8Array:return n/255;case Int32Array:return Math.max(n/2147483647,-1);case Int16Array:return Math.max(n/32767,-1);case Int8Array:return Math.max(n/127,-1);default:throw new Error("THREE.MathUtils: Invalid component type.")}}function _t(n,e){switch(e.constructor){case Float32Array:return n;case Uint32Array:return Math.round(n*4294967295);case Uint16Array:return Math.round(n*65535);case Uint8Array:return Math.round(n*255);case Int32Array:return Math.round(n*2147483647);case Int16Array:return Math.round(n*32767);case Int8Array:return Math.round(n*127);default:throw new Error("THREE.MathUtils: Invalid component type.")}}var Vt={DEG2RAD:hr,RAD2DEG:pr,generateUUID:gn,clamp:Ke,euclideanModulo:uu,mapLinear:Cm,inverseLerp:Pm,lerp:jr,damp:Im,pingpong:Dm,smoothstep:Lm,smootherstep:Nm,randInt:Um,randFloat:Fm,randFloatSpread:Om,seededRandom:Bm,degToRad:zm,radToDeg:km,isPowerOfTwo:Hm,ceilPowerOfTwo:Vm,floorPowerOfTwo:Gm,setQuaternionFromProperEuler:Wm,normalize:_t,denormalize:Ji},_u=class _u{constructor(e=0,t=0){this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("THREE.Vector2: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("THREE.Vector2: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){let t=this.x,i=this.y,s=e.elements;return this.x=s[0]*t+s[3]*i+s[6],this.y=s[1]*t+s[4]*i+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=Ke(this.x,e.x,t.x),this.y=Ke(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=Ke(this.x,e,t),this.y=Ke(this.y,e,t),this}clampLength(e,t){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Ke(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let i=this.dot(e)/t;return Math.acos(Ke(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,i=this.y-e.y;return t*t+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){let i=Math.cos(t),s=Math.sin(t),r=this.x-e.x,a=this.y-e.y;return this.x=r*i-a*s+e.x,this.y=r*s+a*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}};_u.prototype.isVector2=!0;var te=_u,Ai=class{constructor(e=0,t=0,i=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=i,this._w=s}static slerpFlat(e,t,i,s,r,a,o){let c=i[s+0],l=i[s+1],h=i[s+2],d=i[s+3],u=r[a+0],f=r[a+1],g=r[a+2],x=r[a+3];if(d!==x||c!==u||l!==f||h!==g){let p=c*u+l*f+h*g+d*x;p<0&&(u=-u,f=-f,g=-g,x=-x,p=-p);let m=1-o;if(p<.9995){let M=Math.acos(p),b=Math.sin(M);m=Math.sin(m*M)/b,o=Math.sin(o*M)/b,c=c*m+u*o,l=l*m+f*o,h=h*m+g*o,d=d*m+x*o}else{c=c*m+u*o,l=l*m+f*o,h=h*m+g*o,d=d*m+x*o;let M=1/Math.sqrt(c*c+l*l+h*h+d*d);c*=M,l*=M,h*=M,d*=M}}e[t]=c,e[t+1]=l,e[t+2]=h,e[t+3]=d}static multiplyQuaternionsFlat(e,t,i,s,r,a){let o=i[s],c=i[s+1],l=i[s+2],h=i[s+3],d=r[a],u=r[a+1],f=r[a+2],g=r[a+3];return e[t]=o*g+h*d+c*f-l*u,e[t+1]=c*g+h*u+l*d-o*f,e[t+2]=l*g+h*f+o*u-c*d,e[t+3]=h*g-o*d-c*u-l*f,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,i,s){return this._x=e,this._y=t,this._z=i,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){let i=e._x,s=e._y,r=e._z,a=e._order,o=Math.cos,c=Math.sin,l=o(i/2),h=o(s/2),d=o(r/2),u=c(i/2),f=c(s/2),g=c(r/2);switch(a){case"XYZ":this._x=u*h*d+l*f*g,this._y=l*f*d-u*h*g,this._z=l*h*g+u*f*d,this._w=l*h*d-u*f*g;break;case"YXZ":this._x=u*h*d+l*f*g,this._y=l*f*d-u*h*g,this._z=l*h*g-u*f*d,this._w=l*h*d+u*f*g;break;case"ZXY":this._x=u*h*d-l*f*g,this._y=l*f*d+u*h*g,this._z=l*h*g+u*f*d,this._w=l*h*d-u*f*g;break;case"ZYX":this._x=u*h*d-l*f*g,this._y=l*f*d+u*h*g,this._z=l*h*g-u*f*d,this._w=l*h*d+u*f*g;break;case"YZX":this._x=u*h*d+l*f*g,this._y=l*f*d+u*h*g,this._z=l*h*g-u*f*d,this._w=l*h*d-u*f*g;break;case"XZY":this._x=u*h*d-l*f*g,this._y=l*f*d-u*h*g,this._z=l*h*g+u*f*d,this._w=l*h*d+u*f*g;break;default:Ye("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){let i=t/2,s=Math.sin(i);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(e){let t=e.elements,i=t[0],s=t[4],r=t[8],a=t[1],o=t[5],c=t[9],l=t[2],h=t[6],d=t[10],u=i+o+d;if(u>0){let f=.5/Math.sqrt(u+1);this._w=.25/f,this._x=(h-c)*f,this._y=(r-l)*f,this._z=(a-s)*f}else if(i>o&&i>d){let f=2*Math.sqrt(1+i-o-d);this._w=(h-c)/f,this._x=.25*f,this._y=(s+a)/f,this._z=(r+l)/f}else if(o>d){let f=2*Math.sqrt(1+o-i-d);this._w=(r-l)/f,this._x=(s+a)/f,this._y=.25*f,this._z=(c+h)/f}else{let f=2*Math.sqrt(1+d-i-o);this._w=(a-s)/f,this._x=(r+l)/f,this._y=(c+h)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let i=e.dot(t)+1;return i<1e-8?(i=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=i):(this._x=0,this._y=-e.z,this._z=e.y,this._w=i)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=i),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(Ke(this.dot(e),-1,1)))}rotateTowards(e,t){let i=this.angleTo(e);if(i===0)return this;let s=Math.min(1,t/i);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){let i=e._x,s=e._y,r=e._z,a=e._w,o=t._x,c=t._y,l=t._z,h=t._w;return this._x=i*h+a*o+s*l-r*c,this._y=s*h+a*c+r*o-i*l,this._z=r*h+a*l+i*c-s*o,this._w=a*h-i*o-s*c-r*l,this._onChangeCallback(),this}slerp(e,t){let i=e._x,s=e._y,r=e._z,a=e._w,o=this.dot(e);o<0&&(i=-i,s=-s,r=-r,a=-a,o=-o);let c=1-t;if(o<.9995){let l=Math.acos(o),h=Math.sin(l);c=Math.sin(c*l)/h,t=Math.sin(t*l)/h,this._x=this._x*c+i*t,this._y=this._y*c+s*t,this._z=this._z*c+r*t,this._w=this._w*c+a*t,this._onChangeCallback()}else this._x=this._x*c+i*t,this._y=this._y*c+s*t,this._z=this._z*c+r*t,this._w=this._w*c+a*t,this.normalize();return this}slerpQuaternions(e,t,i){return this.copy(e).slerp(t,i)}random(){let e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),i=Math.random(),s=Math.sqrt(1-i),r=Math.sqrt(i);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},xu=class xu{constructor(e=0,t=0,i=0){this.x=e,this.y=t,this.z=i}set(e,t,i){return i===void 0&&(i=this.z),this.x=e,this.y=t,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("THREE.Vector3: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("THREE.Vector3: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Sd.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Sd.setFromAxisAngle(e,t))}applyMatrix3(e){let t=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*i+r[6]*s,this.y=r[1]*t+r[4]*i+r[7]*s,this.z=r[2]*t+r[5]*i+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){let t=this.x,i=this.y,s=this.z,r=e.elements,a=1/(r[3]*t+r[7]*i+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*i+r[8]*s+r[12])*a,this.y=(r[1]*t+r[5]*i+r[9]*s+r[13])*a,this.z=(r[2]*t+r[6]*i+r[10]*s+r[14])*a,this}applyQuaternion(e){let t=this.x,i=this.y,s=this.z,r=e.x,a=e.y,o=e.z,c=e.w,l=2*(a*s-o*i),h=2*(o*t-r*s),d=2*(r*i-a*t);return this.x=t+c*l+a*d-o*h,this.y=i+c*h+o*l-r*d,this.z=s+c*d+r*h-a*l,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){let t=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*i+r[8]*s,this.y=r[1]*t+r[5]*i+r[9]*s,this.z=r[2]*t+r[6]*i+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=Ke(this.x,e.x,t.x),this.y=Ke(this.y,e.y,t.y),this.z=Ke(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=Ke(this.x,e,t),this.y=Ke(this.y,e,t),this.z=Ke(this.z,e,t),this}clampLength(e,t){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Ke(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this.z=e.z+(t.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){let i=e.x,s=e.y,r=e.z,a=t.x,o=t.y,c=t.z;return this.x=s*c-r*o,this.y=r*a-i*c,this.z=i*o-s*a,this}projectOnVector(e){let t=e.lengthSq();if(t===0)return this.set(0,0,0);let i=e.dot(this)/t;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return hh.copy(this).projectOnVector(e),this.sub(hh)}reflect(e){return this.sub(hh.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){let t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;let i=this.dot(e)/t;return Math.acos(Ke(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let t=this.x-e.x,i=this.y-e.y,s=this.z-e.z;return t*t+i*i+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,i){let s=Math.sin(t)*e;return this.x=s*Math.sin(i),this.y=Math.cos(t)*e,this.z=s*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,i){return this.x=e*Math.sin(t),this.y=i,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){let t=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=i,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let e=Math.random()*Math.PI*2,t=Math.random()*2-1,i=Math.sqrt(1-t*t);return this.x=i*Math.cos(e),this.y=t,this.z=i*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}};xu.prototype.isVector3=!0;var A=xu,hh=new A,Sd=new Ai,vu=class vu{constructor(e,t,i,s,r,a,o,c,l){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,i,s,r,a,o,c,l)}set(e,t,i,s,r,a,o,c,l){let h=this.elements;return h[0]=e,h[1]=s,h[2]=o,h[3]=t,h[4]=r,h[5]=c,h[6]=i,h[7]=a,h[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){let t=this.elements,i=e.elements;return t[0]=i[0],t[1]=i[1],t[2]=i[2],t[3]=i[3],t[4]=i[4],t[5]=i[5],t[6]=i[6],t[7]=i[7],t[8]=i[8],this}extractBasis(e,t,i){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){let t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let i=e.elements,s=t.elements,r=this.elements,a=i[0],o=i[3],c=i[6],l=i[1],h=i[4],d=i[7],u=i[2],f=i[5],g=i[8],x=s[0],p=s[3],m=s[6],M=s[1],b=s[4],v=s[7],T=s[2],w=s[5],C=s[8];return r[0]=a*x+o*M+c*T,r[3]=a*p+o*b+c*w,r[6]=a*m+o*v+c*C,r[1]=l*x+h*M+d*T,r[4]=l*p+h*b+d*w,r[7]=l*m+h*v+d*C,r[2]=u*x+f*M+g*T,r[5]=u*p+f*b+g*w,r[8]=u*m+f*v+g*C,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){let e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],c=e[6],l=e[7],h=e[8];return t*a*h-t*o*l-i*r*h+i*o*c+s*r*l-s*a*c}invert(){let e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],c=e[6],l=e[7],h=e[8],d=h*a-o*l,u=o*c-h*r,f=l*r-a*c,g=t*d+i*u+s*f;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);let x=1/g;return e[0]=d*x,e[1]=(s*l-h*i)*x,e[2]=(o*i-s*a)*x,e[3]=u*x,e[4]=(h*t-s*c)*x,e[5]=(s*r-o*t)*x,e[6]=f*x,e[7]=(i*c-l*t)*x,e[8]=(a*t-i*r)*x,this}transpose(){let e,t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){let t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,i,s,r,a,o){let c=Math.cos(r),l=Math.sin(r);return this.set(i*c,i*l,-i*(c*a+l*o)+a+e,-s*l,s*c,-s*(-l*a+c*o)+o+t,0,0,1),this}scale(e,t){return Ts("Matrix3: .scale() is deprecated. Use .makeScale() instead."),this.premultiply(uh.makeScale(e,t)),this}rotate(e){return Ts("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."),this.premultiply(uh.makeRotation(-e)),this}translate(e,t){return Ts("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."),this.premultiply(uh.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){let t=Math.cos(e),i=Math.sin(e);return this.set(t,-i,0,i,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){let t=this.elements,i=e.elements;for(let s=0;s<9;s++)if(t[s]!==i[s])return!1;return!0}fromArray(e,t=0){for(let i=0;i<9;i++)this.elements[i]=e[i+t];return this}toArray(e=[],t=0){let i=this.elements;return e[t]=i[0],e[t+1]=i[1],e[t+2]=i[2],e[t+3]=i[3],e[t+4]=i[4],e[t+5]=i[5],e[t+6]=i[6],e[t+7]=i[7],e[t+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}};vu.prototype.isMatrix3=!0;var je=vu,uh=new je,Ed=new je().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),wd=new je().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function Xm(){let n={enabled:!0,workingColorSpace:ia,spaces:{},convert:function(s,r,a){return this.enabled===!1||r===a||!r||!a||(this.spaces[r].transfer===ft&&(s.r=Pn(s.r),s.g=Pn(s.g),s.b=Pn(s.b)),this.spaces[r].primaries!==this.spaces[a].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===ft&&(s.r=ur(s.r),s.g=ur(s.g),s.b=ur(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===Un?na:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,a){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[a].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return Ts("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),n.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return Ts("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),n.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],i=[.3127,.329];return n.define({[ia]:{primaries:e,whitePoint:i,transfer:na,toXYZ:Ed,fromXYZ:wd,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Ft},outputColorSpaceConfig:{drawingBufferColorSpace:Ft}},[Ft]:{primaries:e,whitePoint:i,transfer:ft,toXYZ:Ed,fromXYZ:wd,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Ft}}}),n}var ht=Xm();function Pn(n){return n<.04045?n*.0773993808:Math.pow(n*.9478672986+.0521327014,2.4)}function ur(n){return n<.0031308?n*12.92:1.055*Math.pow(n,.41666)-.055}var Xs,ol=class{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let i;if(e instanceof HTMLCanvasElement)i=e;else{Xs===void 0&&(Xs=sa("canvas")),Xs.width=e.width,Xs.height=e.height;let s=Xs.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),i=Xs}return i.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){let t=sa("canvas");t.width=e.width,t.height=e.height;let i=t.getContext("2d");i.drawImage(e,0,0,e.width,e.height);let s=i.getImageData(0,0,e.width,e.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=Pn(r[a]/255)*255;return i.putImageData(s,0,0),t}else if(e.data){let t=e.data.slice(0);for(let i=0;i<t.length;i++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[i]=Math.floor(Pn(t[i]/255)*255):t[i]=Pn(t[i]);return{data:t,width:e.width,height:e.height}}else return Ye("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}},qm=0,mr=class{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:qm++}),this.uuid=gn(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){let t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):typeof VideoFrame<"u"&&t instanceof VideoFrame?e.set(t.displayWidth,t.displayHeight,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){let t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];let i={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(dh(s[a].image)):r.push(dh(s[a]))}else r=dh(s);i.url=r}return t||(e.images[this.uuid]=i),i}};function dh(n){return typeof HTMLImageElement<"u"&&n instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&n instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&n instanceof ImageBitmap?ol.getDataURL(n):n.data?{data:Array.from(n.data),width:n.width,height:n.height,type:n.data.constructor.name}:(Ye("Texture: Unable to serialize Texture."),{})}var Ym=0,fh=new A,ui=class n extends Qi{constructor(e=n.DEFAULT_IMAGE,t=n.DEFAULT_MAPPING,i=mn,s=mn,r=jt,a=cs,o=vi,c=li,l=n.DEFAULT_ANISOTROPY,h=Un){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Ym++}),this.uuid=gn(),this.name="",this.source=new mr(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=i,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=l,this.format=o,this.internalFormat=null,this.type=c,this.offset=new te(0,0),this.repeat=new te(1,1),this.center=new te(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new je,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0,this.normalized=!1}get width(){return this.source.getSize(fh).x}get height(){return this.source.getSize(fh).y}get depth(){return this.source.getSize(fh).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(let t in e){let i=e[t];if(i===void 0){Ye(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}let s=this[t];if(s===void 0){Ye(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&i&&s.isVector2&&i.isVector2||s&&i&&s.isVector3&&i.isVector3||s&&i&&s.isMatrix3&&i.isMatrix3?s.copy(i):this[t]=i}}toJSON(e){let t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];let i={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),t||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==nu)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case zi:e.x=e.x-Math.floor(e.x);break;case mn:e.x=e.x<0?0:1;break;case sl:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case zi:e.y=e.y-Math.floor(e.y);break;case mn:e.y=e.y<0?0:1;break;case sl:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}};ui.DEFAULT_IMAGE=null;ui.DEFAULT_MAPPING=nu;ui.DEFAULT_ANISOTROPY=1;var yu=class yu{constructor(e=0,t=0,i=0,s=1){this.x=e,this.y=t,this.z=i,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,i,s){return this.x=e,this.y=t,this.z=i,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("THREE.Vector4: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("THREE.Vector4: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){let t=this.x,i=this.y,s=this.z,r=this.w,a=e.elements;return this.x=a[0]*t+a[4]*i+a[8]*s+a[12]*r,this.y=a[1]*t+a[5]*i+a[9]*s+a[13]*r,this.z=a[2]*t+a[6]*i+a[10]*s+a[14]*r,this.w=a[3]*t+a[7]*i+a[11]*s+a[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);let t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,i,s,r,c=e.elements,l=c[0],h=c[4],d=c[8],u=c[1],f=c[5],g=c[9],x=c[2],p=c[6],m=c[10];if(Math.abs(h-u)<.01&&Math.abs(d-x)<.01&&Math.abs(g-p)<.01){if(Math.abs(h+u)<.1&&Math.abs(d+x)<.1&&Math.abs(g+p)<.1&&Math.abs(l+f+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;let b=(l+1)/2,v=(f+1)/2,T=(m+1)/2,w=(h+u)/4,C=(d+x)/4,_=(g+p)/4;return b>v&&b>T?b<.01?(i=0,s=.707106781,r=.707106781):(i=Math.sqrt(b),s=w/i,r=C/i):v>T?v<.01?(i=.707106781,s=0,r=.707106781):(s=Math.sqrt(v),i=w/s,r=_/s):T<.01?(i=.707106781,s=.707106781,r=0):(r=Math.sqrt(T),i=C/r,s=_/r),this.set(i,s,r,t),this}let M=Math.sqrt((p-g)*(p-g)+(d-x)*(d-x)+(u-h)*(u-h));return Math.abs(M)<.001&&(M=1),this.x=(p-g)/M,this.y=(d-x)/M,this.z=(u-h)/M,this.w=Math.acos((l+f+m-1)/2),this}setFromMatrixPosition(e){let t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=Ke(this.x,e.x,t.x),this.y=Ke(this.y,e.y,t.y),this.z=Ke(this.z,e.z,t.z),this.w=Ke(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=Ke(this.x,e,t),this.y=Ke(this.y,e,t),this.z=Ke(this.z,e,t),this.w=Ke(this.w,e,t),this}clampLength(e,t){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Ke(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this.z=e.z+(t.z-e.z)*i,this.w=e.w+(t.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}};yu.prototype.isVector4=!0;var gt=yu,ll=class extends Qi{constructor(e=1,t=1,i={}){super(),i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:jt,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},i),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=i.depth,this.scissor=new gt(0,0,e,t),this.scissorTest=!1,this.viewport=new gt(0,0,e,t),this.textures=[];let s={width:e,height:t,depth:i.depth},r=new ui(s),a=i.count;for(let o=0;o<a;o++)this.textures[o]=r.clone(),this.textures[o].isRenderTargetTexture=!0,this.textures[o].renderTarget=this;this._setTextureOptions(i),this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=i.depthTexture,this.samples=i.samples,this.multiview=i.multiview,this.useArrayDepthTexture=i.useArrayDepthTexture}_setTextureOptions(e={}){let t={minFilter:jt,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let i=0;i<this.textures.length;i++)this.textures[i].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,i=1){if(this.width!==e||this.height!==t||this.depth!==i){this.width=e,this.height=t,this.depth=i;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=i,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,i=e.textures.length;t<i;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;let s=Object.assign({},e.textures[t].image);this.textures[t].source=new mr(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this.multiview=e.multiview,this.useArrayDepthTexture=e.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}},Ht=class extends ll{constructor(e=1,t=1,i={}){super(e,t,i),this.isWebGLRenderTarget=!0}},aa=class extends ui{constructor(e=null,t=1,i=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:i,depth:s},this.magFilter=Ot,this.minFilter=Ot,this.wrapR=mn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}};var cl=class extends ui{constructor(e=null,t=1,i=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:i,depth:s},this.magFilter=Ot,this.minFilter=Ot,this.wrapR=mn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var Ul=class Ul{constructor(e,t,i,s,r,a,o,c,l,h,d,u,f,g,x,p){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,i,s,r,a,o,c,l,h,d,u,f,g,x,p)}set(e,t,i,s,r,a,o,c,l,h,d,u,f,g,x,p){let m=this.elements;return m[0]=e,m[4]=t,m[8]=i,m[12]=s,m[1]=r,m[5]=a,m[9]=o,m[13]=c,m[2]=l,m[6]=h,m[10]=d,m[14]=u,m[3]=f,m[7]=g,m[11]=x,m[15]=p,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Ul().fromArray(this.elements)}copy(e){let t=this.elements,i=e.elements;return t[0]=i[0],t[1]=i[1],t[2]=i[2],t[3]=i[3],t[4]=i[4],t[5]=i[5],t[6]=i[6],t[7]=i[7],t[8]=i[8],t[9]=i[9],t[10]=i[10],t[11]=i[11],t[12]=i[12],t[13]=i[13],t[14]=i[14],t[15]=i[15],this}copyPosition(e){let t=this.elements,i=e.elements;return t[12]=i[12],t[13]=i[13],t[14]=i[14],this}setFromMatrix3(e){let t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,i){return this.determinantAffine()===0?(e.set(1,0,0),t.set(0,1,0),i.set(0,0,1),this):(e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this)}makeBasis(e,t,i){return this.set(e.x,t.x,i.x,0,e.y,t.y,i.y,0,e.z,t.z,i.z,0,0,0,0,1),this}extractRotation(e){if(e.determinantAffine()===0)return this.identity();let t=this.elements,i=e.elements,s=1/qs.setFromMatrixColumn(e,0).length(),r=1/qs.setFromMatrixColumn(e,1).length(),a=1/qs.setFromMatrixColumn(e,2).length();return t[0]=i[0]*s,t[1]=i[1]*s,t[2]=i[2]*s,t[3]=0,t[4]=i[4]*r,t[5]=i[5]*r,t[6]=i[6]*r,t[7]=0,t[8]=i[8]*a,t[9]=i[9]*a,t[10]=i[10]*a,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){let t=this.elements,i=e.x,s=e.y,r=e.z,a=Math.cos(i),o=Math.sin(i),c=Math.cos(s),l=Math.sin(s),h=Math.cos(r),d=Math.sin(r);if(e.order==="XYZ"){let u=a*h,f=a*d,g=o*h,x=o*d;t[0]=c*h,t[4]=-c*d,t[8]=l,t[1]=f+g*l,t[5]=u-x*l,t[9]=-o*c,t[2]=x-u*l,t[6]=g+f*l,t[10]=a*c}else if(e.order==="YXZ"){let u=c*h,f=c*d,g=l*h,x=l*d;t[0]=u+x*o,t[4]=g*o-f,t[8]=a*l,t[1]=a*d,t[5]=a*h,t[9]=-o,t[2]=f*o-g,t[6]=x+u*o,t[10]=a*c}else if(e.order==="ZXY"){let u=c*h,f=c*d,g=l*h,x=l*d;t[0]=u-x*o,t[4]=-a*d,t[8]=g+f*o,t[1]=f+g*o,t[5]=a*h,t[9]=x-u*o,t[2]=-a*l,t[6]=o,t[10]=a*c}else if(e.order==="ZYX"){let u=a*h,f=a*d,g=o*h,x=o*d;t[0]=c*h,t[4]=g*l-f,t[8]=u*l+x,t[1]=c*d,t[5]=x*l+u,t[9]=f*l-g,t[2]=-l,t[6]=o*c,t[10]=a*c}else if(e.order==="YZX"){let u=a*c,f=a*l,g=o*c,x=o*l;t[0]=c*h,t[4]=x-u*d,t[8]=g*d+f,t[1]=d,t[5]=a*h,t[9]=-o*h,t[2]=-l*h,t[6]=f*d+g,t[10]=u-x*d}else if(e.order==="XZY"){let u=a*c,f=a*l,g=o*c,x=o*l;t[0]=c*h,t[4]=-d,t[8]=l*h,t[1]=u*d+x,t[5]=a*h,t[9]=f*d-g,t[2]=g*d-f,t[6]=o*h,t[10]=x*d+u}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose($m,e,Zm)}lookAt(e,t,i){let s=this.elements;return Ei.subVectors(e,t),Ei.lengthSq()===0&&(Ei.z=1),Ei.normalize(),qn.crossVectors(i,Ei),qn.lengthSq()===0&&(Math.abs(i.z)===1?Ei.x+=1e-4:Ei.z+=1e-4,Ei.normalize(),qn.crossVectors(i,Ei)),qn.normalize(),go.crossVectors(Ei,qn),s[0]=qn.x,s[4]=go.x,s[8]=Ei.x,s[1]=qn.y,s[5]=go.y,s[9]=Ei.y,s[2]=qn.z,s[6]=go.z,s[10]=Ei.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){let i=e.elements,s=t.elements,r=this.elements,a=i[0],o=i[4],c=i[8],l=i[12],h=i[1],d=i[5],u=i[9],f=i[13],g=i[2],x=i[6],p=i[10],m=i[14],M=i[3],b=i[7],v=i[11],T=i[15],w=s[0],C=s[4],_=s[8],E=s[12],P=s[1],I=s[5],L=s[9],X=s[13],W=s[2],U=s[6],z=s[10],H=s[14],Q=s[3],ie=s[7],q=s[11],Z=s[15];return r[0]=a*w+o*P+c*W+l*Q,r[4]=a*C+o*I+c*U+l*ie,r[8]=a*_+o*L+c*z+l*q,r[12]=a*E+o*X+c*H+l*Z,r[1]=h*w+d*P+u*W+f*Q,r[5]=h*C+d*I+u*U+f*ie,r[9]=h*_+d*L+u*z+f*q,r[13]=h*E+d*X+u*H+f*Z,r[2]=g*w+x*P+p*W+m*Q,r[6]=g*C+x*I+p*U+m*ie,r[10]=g*_+x*L+p*z+m*q,r[14]=g*E+x*X+p*H+m*Z,r[3]=M*w+b*P+v*W+T*Q,r[7]=M*C+b*I+v*U+T*ie,r[11]=M*_+b*L+v*z+T*q,r[15]=M*E+b*X+v*H+T*Z,this}multiplyScalar(e){let t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){let e=this.elements,t=e[0],i=e[4],s=e[8],r=e[12],a=e[1],o=e[5],c=e[9],l=e[13],h=e[2],d=e[6],u=e[10],f=e[14],g=e[3],x=e[7],p=e[11],m=e[15],M=c*f-l*u,b=o*f-l*d,v=o*u-c*d,T=a*f-l*h,w=a*u-c*h,C=a*d-o*h;return t*(x*M-p*b+m*v)-i*(g*M-p*T+m*w)+s*(g*b-x*T+m*C)-r*(g*v-x*w+p*C)}determinantAffine(){let e=this.elements,t=e[0],i=e[4],s=e[8],r=e[1],a=e[5],o=e[9],c=e[2],l=e[6],h=e[10];return t*(a*h-o*l)-i*(r*h-o*c)+s*(r*l-a*c)}transpose(){let e=this.elements,t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,i){let s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=i),this}invert(){let e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],c=e[6],l=e[7],h=e[8],d=e[9],u=e[10],f=e[11],g=e[12],x=e[13],p=e[14],m=e[15],M=t*o-i*a,b=t*c-s*a,v=t*l-r*a,T=i*c-s*o,w=i*l-r*o,C=s*l-r*c,_=h*x-d*g,E=h*p-u*g,P=h*m-f*g,I=d*p-u*x,L=d*m-f*x,X=u*m-f*p,W=M*X-b*L+v*I+T*P-w*E+C*_;if(W===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let U=1/W;return e[0]=(o*X-c*L+l*I)*U,e[1]=(s*L-i*X-r*I)*U,e[2]=(x*C-p*w+m*T)*U,e[3]=(u*w-d*C-f*T)*U,e[4]=(c*P-a*X-l*E)*U,e[5]=(t*X-s*P+r*E)*U,e[6]=(p*v-g*C-m*b)*U,e[7]=(h*C-u*v+f*b)*U,e[8]=(a*L-o*P+l*_)*U,e[9]=(i*P-t*L-r*_)*U,e[10]=(g*w-x*v+m*M)*U,e[11]=(d*v-h*w-f*M)*U,e[12]=(o*E-a*I-c*_)*U,e[13]=(t*I-i*E+s*_)*U,e[14]=(x*b-g*T-p*M)*U,e[15]=(h*T-d*b+u*M)*U,this}scale(e){let t=this.elements,i=e.x,s=e.y,r=e.z;return t[0]*=i,t[4]*=s,t[8]*=r,t[1]*=i,t[5]*=s,t[9]*=r,t[2]*=i,t[6]*=s,t[10]*=r,t[3]*=i,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){let e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,i,s))}makeTranslation(e,t,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,i,0,0,0,1),this}makeRotationX(e){let t=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,t,-i,0,0,i,t,0,0,0,0,1),this}makeRotationY(e){let t=Math.cos(e),i=Math.sin(e);return this.set(t,0,i,0,0,1,0,0,-i,0,t,0,0,0,0,1),this}makeRotationZ(e){let t=Math.cos(e),i=Math.sin(e);return this.set(t,-i,0,0,i,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){let i=Math.cos(t),s=Math.sin(t),r=1-i,a=e.x,o=e.y,c=e.z,l=r*a,h=r*o;return this.set(l*a+i,l*o-s*c,l*c+s*o,0,l*o+s*c,h*o+i,h*c-s*a,0,l*c-s*o,h*c+s*a,r*c*c+i,0,0,0,0,1),this}makeScale(e,t,i){return this.set(e,0,0,0,0,t,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,t,i,s,r,a){return this.set(1,i,r,0,e,1,a,0,t,s,1,0,0,0,0,1),this}compose(e,t,i){let s=this.elements,r=t._x,a=t._y,o=t._z,c=t._w,l=r+r,h=a+a,d=o+o,u=r*l,f=r*h,g=r*d,x=a*h,p=a*d,m=o*d,M=c*l,b=c*h,v=c*d,T=i.x,w=i.y,C=i.z;return s[0]=(1-(x+m))*T,s[1]=(f+v)*T,s[2]=(g-b)*T,s[3]=0,s[4]=(f-v)*w,s[5]=(1-(u+m))*w,s[6]=(p+M)*w,s[7]=0,s[8]=(g+b)*C,s[9]=(p-M)*C,s[10]=(1-(u+x))*C,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,i){let s=this.elements;e.x=s[12],e.y=s[13],e.z=s[14];let r=this.determinantAffine();if(r===0)return i.set(1,1,1),t.identity(),this;let a=qs.set(s[0],s[1],s[2]).length(),o=qs.set(s[4],s[5],s[6]).length(),c=qs.set(s[8],s[9],s[10]).length();r<0&&(a=-a),Yi.copy(this);let l=1/a,h=1/o,d=1/c;return Yi.elements[0]*=l,Yi.elements[1]*=l,Yi.elements[2]*=l,Yi.elements[4]*=h,Yi.elements[5]*=h,Yi.elements[6]*=h,Yi.elements[8]*=d,Yi.elements[9]*=d,Yi.elements[10]*=d,t.setFromRotationMatrix(Yi),i.x=a,i.y=o,i.z=c,this}makePerspective(e,t,i,s,r,a,o=Ki,c=!1){let l=this.elements,h=2*r/(t-e),d=2*r/(i-s),u=(t+e)/(t-e),f=(i+s)/(i-s),g,x;if(c)g=r/(a-r),x=a*r/(a-r);else if(o===Ki)g=-(a+r)/(a-r),x=-2*a*r/(a-r);else if(o===dr)g=-a/(a-r),x=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return l[0]=h,l[4]=0,l[8]=u,l[12]=0,l[1]=0,l[5]=d,l[9]=f,l[13]=0,l[2]=0,l[6]=0,l[10]=g,l[14]=x,l[3]=0,l[7]=0,l[11]=-1,l[15]=0,this}makeOrthographic(e,t,i,s,r,a,o=Ki,c=!1){let l=this.elements,h=2/(t-e),d=2/(i-s),u=-(t+e)/(t-e),f=-(i+s)/(i-s),g,x;if(c)g=1/(a-r),x=a/(a-r);else if(o===Ki)g=-2/(a-r),x=-(a+r)/(a-r);else if(o===dr)g=-1/(a-r),x=-r/(a-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return l[0]=h,l[4]=0,l[8]=0,l[12]=u,l[1]=0,l[5]=d,l[9]=0,l[13]=f,l[2]=0,l[6]=0,l[10]=g,l[14]=x,l[3]=0,l[7]=0,l[11]=0,l[15]=1,this}equals(e){let t=this.elements,i=e.elements;for(let s=0;s<16;s++)if(t[s]!==i[s])return!1;return!0}fromArray(e,t=0){for(let i=0;i<16;i++)this.elements[i]=e[i+t];return this}toArray(e=[],t=0){let i=this.elements;return e[t]=i[0],e[t+1]=i[1],e[t+2]=i[2],e[t+3]=i[3],e[t+4]=i[4],e[t+5]=i[5],e[t+6]=i[6],e[t+7]=i[7],e[t+8]=i[8],e[t+9]=i[9],e[t+10]=i[10],e[t+11]=i[11],e[t+12]=i[12],e[t+13]=i[13],e[t+14]=i[14],e[t+15]=i[15],e}};Ul.prototype.isMatrix4=!0;var rt=Ul,qs=new A,Yi=new rt,$m=new A(0,0,0),Zm=new A(1,1,1),qn=new A,go=new A,Ei=new A,Td=new rt,Ad=new Ai,Ri=class n{constructor(e=0,t=0,i=0,s=n.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=i,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,i,s=this._order){return this._x=e,this._y=t,this._z=i,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,i=!0){let s=e.elements,r=s[0],a=s[4],o=s[8],c=s[1],l=s[5],h=s[9],d=s[2],u=s[6],f=s[10];switch(t){case"XYZ":this._y=Math.asin(Ke(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,f),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(u,l),this._z=0);break;case"YXZ":this._x=Math.asin(-Ke(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,f),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-d,r),this._z=0);break;case"ZXY":this._x=Math.asin(Ke(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(-d,f),this._z=Math.atan2(-a,l)):(this._y=0,this._z=Math.atan2(c,r));break;case"ZYX":this._y=Math.asin(-Ke(d,-1,1)),Math.abs(d)<.9999999?(this._x=Math.atan2(u,f),this._z=Math.atan2(c,r)):(this._x=0,this._z=Math.atan2(-a,l));break;case"YZX":this._z=Math.asin(Ke(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-h,l),this._y=Math.atan2(-d,r)):(this._x=0,this._y=Math.atan2(o,f));break;case"XZY":this._z=Math.asin(-Ke(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(u,l),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-h,f),this._y=0);break;default:Ye("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,i){return Td.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Td,t,i)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Ad.setFromEuler(this),this.setFromQuaternion(Ad,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};Ri.DEFAULT_ORDER="XYZ";var gr=class{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}},Jm=0,Rd=new A,Ys=new Ai,wn=new rt,_o=new A,Hr=new A,Km=new A,jm=new Ai,Cd=new A(1,0,0),Pd=new A(0,1,0),Id=new A(0,0,1),Dd={type:"added"},Qm={type:"removed"},$s={type:"childadded",child:null},ph={type:"childremoved",child:null},pt=class n extends Qi{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:Jm++}),this.uuid=gn(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=n.DEFAULT_UP.clone();let e=new A,t=new Ri,i=new Ai,s=new A(1,1,1);function r(){i.setFromEuler(t,!1)}function a(){t.setFromQuaternion(i,void 0,!1)}t._onChange(r),i._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new rt},normalMatrix:{value:new je}}),this.matrix=new rt,this.matrixWorld=new rt,this.matrixAutoUpdate=n.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=n.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new gr,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return Ys.setFromAxisAngle(e,t),this.quaternion.multiply(Ys),this}rotateOnWorldAxis(e,t){return Ys.setFromAxisAngle(e,t),this.quaternion.premultiply(Ys),this}rotateX(e){return this.rotateOnAxis(Cd,e)}rotateY(e){return this.rotateOnAxis(Pd,e)}rotateZ(e){return this.rotateOnAxis(Id,e)}translateOnAxis(e,t){return Rd.copy(e).applyQuaternion(this.quaternion),this.position.add(Rd.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(Cd,e)}translateY(e){return this.translateOnAxis(Pd,e)}translateZ(e){return this.translateOnAxis(Id,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(wn.copy(this.matrixWorld).invert())}lookAt(e,t,i){e.isVector3?_o.copy(e):_o.set(e,t,i);let s=this.parent;this.updateWorldMatrix(!0,!1),Hr.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?wn.lookAt(Hr,_o,this.up):wn.lookAt(_o,Hr,this.up),this.quaternion.setFromRotationMatrix(wn),s&&(wn.extractRotation(s.matrixWorld),Ys.setFromRotationMatrix(wn),this.quaternion.premultiply(Ys.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?($e("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Dd),$s.child=e,this.dispatchEvent($s),$s.child=null):$e("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}let t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(Qm),ph.child=e,this.dispatchEvent(ph),ph.child=null),this}removeFromParent(){let e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),wn.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),wn.multiply(e.parent.matrixWorld)),e.applyMatrix4(wn),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Dd),$s.child=e,this.dispatchEvent($s),$s.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let i=0,s=this.children.length;i<s;i++){let a=this.children[i].getObjectByProperty(e,t);if(a!==void 0)return a}}getObjectsByProperty(e,t,i=[]){this[e]===t&&i.push(this);let s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(e,t,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Hr,e,Km),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Hr,jm,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);let t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);let t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);let t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].traverseVisible(e)}traverseAncestors(e){let t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let e=this.pivot;if(e!==null){let t=e.x,i=e.y,s=e.z,r=this.matrix.elements;r[12]+=t-r[0]*t-r[4]*i-r[8]*s,r[13]+=i-r[1]*t-r[5]*i-r[9]*s,r[14]+=s-r[2]*t-r[6]*i-r[10]*s}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);let t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].updateMatrixWorld(e)}updateWorldMatrix(e,t,i=!1){let s=this.parent;if(e===!0&&s!==null&&s.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||i)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,i=!0),t===!0){let r=this.children;for(let a=0,o=r.length;a<o;a++)r[a].updateWorldMatrix(!1,!0,i)}}toJSON(e){let t=e===void 0||typeof e=="string",i={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),this.static!==!1&&(s.static=this.static),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.pivot!==null&&(s.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(s.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(s.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(o=>({...o})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(o,c){return o[c.uuid]===void 0&&(o[c.uuid]=c.toJSON(e)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);let o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){let c=o.shapes;if(Array.isArray(c))for(let l=0,h=c.length;l<h;l++){let d=c[l];r(e.shapes,d)}else r(e.shapes,c)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let o=[];for(let c=0,l=this.material.length;c<l;c++)o.push(r(e.materials,this.material[c]));s.material=o}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){let c=this.animations[o];s.animations.push(r(e.animations,c))}}if(t){let o=a(e.geometries),c=a(e.materials),l=a(e.textures),h=a(e.images),d=a(e.shapes),u=a(e.skeletons),f=a(e.animations),g=a(e.nodes);o.length>0&&(i.geometries=o),c.length>0&&(i.materials=c),l.length>0&&(i.textures=l),h.length>0&&(i.images=h),d.length>0&&(i.shapes=d),u.length>0&&(i.skeletons=u),f.length>0&&(i.animations=f),g.length>0&&(i.nodes=g)}return i.object=s,i;function a(o){let c=[];for(let l in o){let h=o[l];delete h.metadata,c.push(h)}return c}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot!==null?e.pivot.clone():null,this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let i=0;i<e.children.length;i++){let s=e.children[i];this.add(s.clone())}return this}};pt.DEFAULT_UP=new A(0,1,0);pt.DEFAULT_MATRIX_AUTO_UPDATE=!0;pt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var et=class extends pt{constructor(){super(),this.isGroup=!0,this.type="Group"}},eg={type:"move"},_r=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new et,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new et,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new A,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new A),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new et,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new A,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new A,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){let t=this._hand;if(t)for(let i of e.hand.values())this._getHandJoint(t,i)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,i){let s=null,r=null,a=null,o=this._targetRay,c=this._grip,l=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(l&&e.hand){a=!0;for(let x of e.hand.values()){let p=t.getJointPose(x,i),m=this._getHandJoint(l,x);p!==null&&(m.matrix.fromArray(p.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=p.radius),m.visible=p!==null}let h=l.joints["index-finger-tip"],d=l.joints["thumb-tip"],u=h.position.distanceTo(d.position),f=.02,g=.005;l.inputState.pinching&&u>f+g?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!l.inputState.pinching&&u<=f-g&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else c!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,i),r!==null&&(c.matrix.fromArray(r.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),c.matrixWorldNeedsUpdate=!0,r.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(r.linearVelocity)):c.hasLinearVelocity=!1,r.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(r.angularVelocity)):c.hasAngularVelocity=!1,c.eventsEnabled&&c.dispatchEvent({type:"gripUpdated",data:e,target:this})));o!==null&&(s=t.getPose(e.targetRaySpace,i),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(eg)))}return o!==null&&(o.visible=s!==null),c!==null&&(c.visible=r!==null),l!==null&&(l.visible=a!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){let i=new et;i.matrixAutoUpdate=!1,i.visible=!1,e.joints[t.jointName]=i,e.add(i)}return e.joints[t.jointName]}},zf={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Yn={h:0,s:0,l:0},xo={h:0,s:0,l:0};function mh(n,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?n+(e-n)*6*t:t<1/2?e:t<2/3?n+(e-n)*6*(2/3-t):n}var Le=class{constructor(e,t,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,i)}set(e,t,i){if(t===void 0&&i===void 0){let s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,i);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Ft){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,ht.colorSpaceToWorking(this,t),this}setRGB(e,t,i,s=ht.workingColorSpace){return this.r=e,this.g=t,this.b=i,ht.colorSpaceToWorking(this,s),this}setHSL(e,t,i,s=ht.workingColorSpace){if(e=uu(e,1),t=Ke(t,0,1),i=Ke(i,0,1),t===0)this.r=this.g=this.b=i;else{let r=i<=.5?i*(1+t):i+t-i*t,a=2*i-r;this.r=mh(a,r,e+1/3),this.g=mh(a,r,e),this.b=mh(a,r,e-1/3)}return ht.colorSpaceToWorking(this,s),this}setStyle(e,t=Ft){function i(r){r!==void 0&&parseFloat(r)<1&&Ye("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r,a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:Ye("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){let r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(a===6)return this.setHex(parseInt(r,16),t);Ye("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Ft){let i=zf[e.toLowerCase()];return i!==void 0?this.setHex(i,t):Ye("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Pn(e.r),this.g=Pn(e.g),this.b=Pn(e.b),this}copyLinearToSRGB(e){return this.r=ur(e.r),this.g=ur(e.g),this.b=ur(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Ft){return ht.workingToColorSpace(oi.copy(this),e),Math.round(Ke(oi.r*255,0,255))*65536+Math.round(Ke(oi.g*255,0,255))*256+Math.round(Ke(oi.b*255,0,255))}getHexString(e=Ft){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=ht.workingColorSpace){ht.workingToColorSpace(oi.copy(this),t);let i=oi.r,s=oi.g,r=oi.b,a=Math.max(i,s,r),o=Math.min(i,s,r),c,l,h=(o+a)/2;if(o===a)c=0,l=0;else{let d=a-o;switch(l=h<=.5?d/(a+o):d/(2-a-o),a){case i:c=(s-r)/d+(s<r?6:0);break;case s:c=(r-i)/d+2;break;case r:c=(i-s)/d+4;break}c/=6}return e.h=c,e.s=l,e.l=h,e}getRGB(e,t=ht.workingColorSpace){return ht.workingToColorSpace(oi.copy(this),t),e.r=oi.r,e.g=oi.g,e.b=oi.b,e}getStyle(e=Ft){ht.workingToColorSpace(oi.copy(this),e);let t=oi.r,i=oi.g,s=oi.b;return e!==Ft?`color(${e} ${t.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(i*255)},${Math.round(s*255)})`}offsetHSL(e,t,i){return this.getHSL(Yn),this.setHSL(Yn.h+e,Yn.s+t,Yn.l+i)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,i){return this.r=e.r+(t.r-e.r)*i,this.g=e.g+(t.g-e.g)*i,this.b=e.b+(t.b-e.b)*i,this}lerpHSL(e,t){this.getHSL(Yn),e.getHSL(xo);let i=jr(Yn.h,xo.h,t),s=jr(Yn.s,xo.s,t),r=jr(Yn.l,xo.l,t);return this.setHSL(i,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){let t=this.r,i=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*i+r[6]*s,this.g=r[1]*t+r[4]*i+r[7]*s,this.b=r[2]*t+r[5]*i+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},oi=new Le;Le.NAMES=zf;var oa=class n{constructor(e,t=1,i=1e3){this.isFog=!0,this.name="",this.color=new Le(e),this.near=t,this.far=i}clone(){return new n(this.color,this.near,this.far)}toJSON(){return{type:"Fog",name:this.name,color:this.color.getHex(),near:this.near,far:this.far}}},Cs=class extends pt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Ri,this.environmentIntensity=1,this.environmentRotation=new Ri,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){let t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}},$i=new A,Tn=new A,gh=new A,An=new A,Zs=new A,Js=new A,Ld=new A,_h=new A,xh=new A,vh=new A,yh=new gt,Mh=new gt,bh=new gt,pn=class n{constructor(e=new A,t=new A,i=new A){this.a=e,this.b=t,this.c=i}static getNormal(e,t,i,s){s.subVectors(i,t),$i.subVectors(e,t),s.cross($i);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,i,s,r){$i.subVectors(s,t),Tn.subVectors(i,t),gh.subVectors(e,t);let a=$i.dot($i),o=$i.dot(Tn),c=$i.dot(gh),l=Tn.dot(Tn),h=Tn.dot(gh),d=a*l-o*o;if(d===0)return r.set(0,0,0),null;let u=1/d,f=(l*c-o*h)*u,g=(a*h-o*c)*u;return r.set(1-f-g,g,f)}static containsPoint(e,t,i,s){return this.getBarycoord(e,t,i,s,An)===null?!1:An.x>=0&&An.y>=0&&An.x+An.y<=1}static getInterpolation(e,t,i,s,r,a,o,c){return this.getBarycoord(e,t,i,s,An)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(r,An.x),c.addScaledVector(a,An.y),c.addScaledVector(o,An.z),c)}static getInterpolatedAttribute(e,t,i,s,r,a){return yh.setScalar(0),Mh.setScalar(0),bh.setScalar(0),yh.fromBufferAttribute(e,t),Mh.fromBufferAttribute(e,i),bh.fromBufferAttribute(e,s),a.setScalar(0),a.addScaledVector(yh,r.x),a.addScaledVector(Mh,r.y),a.addScaledVector(bh,r.z),a}static isFrontFacing(e,t,i,s){return $i.subVectors(i,t),Tn.subVectors(e,t),$i.cross(Tn).dot(s)<0}set(e,t,i){return this.a.copy(e),this.b.copy(t),this.c.copy(i),this}setFromPointsAndIndices(e,t,i,s){return this.a.copy(e[t]),this.b.copy(e[i]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,i,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return $i.subVectors(this.c,this.b),Tn.subVectors(this.a,this.b),$i.cross(Tn).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return n.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return n.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,i,s,r){return n.getInterpolation(e,this.a,this.b,this.c,t,i,s,r)}containsPoint(e){return n.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return n.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){let i=this.a,s=this.b,r=this.c,a,o;Zs.subVectors(s,i),Js.subVectors(r,i),_h.subVectors(e,i);let c=Zs.dot(_h),l=Js.dot(_h);if(c<=0&&l<=0)return t.copy(i);xh.subVectors(e,s);let h=Zs.dot(xh),d=Js.dot(xh);if(h>=0&&d<=h)return t.copy(s);let u=c*d-h*l;if(u<=0&&c>=0&&h<=0)return a=c/(c-h),t.copy(i).addScaledVector(Zs,a);vh.subVectors(e,r);let f=Zs.dot(vh),g=Js.dot(vh);if(g>=0&&f<=g)return t.copy(r);let x=f*l-c*g;if(x<=0&&l>=0&&g<=0)return o=l/(l-g),t.copy(i).addScaledVector(Js,o);let p=h*g-f*d;if(p<=0&&d-h>=0&&f-g>=0)return Ld.subVectors(r,s),o=(d-h)/(d-h+(f-g)),t.copy(s).addScaledVector(Ld,o);let m=1/(p+x+u);return a=x*m,o=u*m,t.copy(i).addScaledVector(Zs,a).addScaledVector(Js,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},di=class{constructor(e=new A(1/0,1/0,1/0),t=new A(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,i=e.length;t<i;t+=3)this.expandByPoint(Zi.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,i=e.count;t<i;t++)this.expandByPoint(Zi.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,i=e.length;t<i;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){let i=Zi.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(i),this.max.copy(e).add(i),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);let i=e.geometry;if(i!==void 0){let r=i.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let a=0,o=r.count;a<o;a++)e.isMesh===!0?e.getVertexPosition(a,Zi):Zi.fromBufferAttribute(r,a),Zi.applyMatrix4(e.matrixWorld),this.expandByPoint(Zi);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),vo.copy(e.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),vo.copy(i.boundingBox)),vo.applyMatrix4(e.matrixWorld),this.union(vo)}let s=e.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Zi),Zi.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,i;return e.normal.x>0?(t=e.normal.x*this.min.x,i=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,i=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,i+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,i+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,i+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,i+=e.normal.z*this.min.z),t<=-e.constant&&i>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Vr),yo.subVectors(this.max,Vr),Ks.subVectors(e.a,Vr),js.subVectors(e.b,Vr),Qs.subVectors(e.c,Vr),$n.subVectors(js,Ks),Zn.subVectors(Qs,js),ys.subVectors(Ks,Qs);let t=[0,-$n.z,$n.y,0,-Zn.z,Zn.y,0,-ys.z,ys.y,$n.z,0,-$n.x,Zn.z,0,-Zn.x,ys.z,0,-ys.x,-$n.y,$n.x,0,-Zn.y,Zn.x,0,-ys.y,ys.x,0];return!Sh(t,Ks,js,Qs,yo)||(t=[1,0,0,0,1,0,0,0,1],!Sh(t,Ks,js,Qs,yo))?!1:(Mo.crossVectors($n,Zn),t=[Mo.x,Mo.y,Mo.z],Sh(t,Ks,js,Qs,yo))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Zi).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Zi).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Rn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Rn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Rn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Rn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Rn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Rn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Rn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Rn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Rn),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}},Rn=[new A,new A,new A,new A,new A,new A,new A,new A],Zi=new A,vo=new di,Ks=new A,js=new A,Qs=new A,$n=new A,Zn=new A,ys=new A,Vr=new A,yo=new A,Mo=new A,Ms=new A;function Sh(n,e,t,i,s){for(let r=0,a=n.length-3;r<=a;r+=3){Ms.fromArray(n,r);let o=s.x*Math.abs(Ms.x)+s.y*Math.abs(Ms.y)+s.z*Math.abs(Ms.z),c=e.dot(Ms),l=t.dot(Ms),h=i.dot(Ms);if(Math.max(-Math.max(c,l,h),Math.min(c,l,h))>o)return!1}return!0}var kt=new A,bo=new te,tg=0,Yt=class extends Qi{constructor(e,t,i=!1){if(super(),Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:tg++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=i,this.usage=al,this.updateRanges=[],this.gpuType=Vi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,i){e*=this.itemSize,i*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[i+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,i=this.count;t<i;t++)bo.fromBufferAttribute(this,t),bo.applyMatrix3(e),this.setXY(t,bo.x,bo.y);else if(this.itemSize===3)for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix3(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyMatrix4(e){for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix4(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyNormalMatrix(e){for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.applyNormalMatrix(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}transformDirection(e){for(let t=0,i=this.count;t<i;t++)kt.fromBufferAttribute(this,t),kt.transformDirection(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let i=this.array[e*this.itemSize+t];return this.normalized&&(i=Ji(i,this.array)),i}setComponent(e,t,i){return this.normalized&&(i=_t(i,this.array)),this.array[e*this.itemSize+t]=i,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=Ji(t,this.array)),t}setX(e,t){return this.normalized&&(t=_t(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=Ji(t,this.array)),t}setY(e,t){return this.normalized&&(t=_t(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=Ji(t,this.array)),t}setZ(e,t){return this.normalized&&(t=_t(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=Ji(t,this.array)),t}setW(e,t){return this.normalized&&(t=_t(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,i){return e*=this.itemSize,this.normalized&&(t=_t(t,this.array),i=_t(i,this.array)),this.array[e+0]=t,this.array[e+1]=i,this}setXYZ(e,t,i,s){return e*=this.itemSize,this.normalized&&(t=_t(t,this.array),i=_t(i,this.array),s=_t(s,this.array)),this.array[e+0]=t,this.array[e+1]=i,this.array[e+2]=s,this}setXYZW(e,t,i,s,r){return e*=this.itemSize,this.normalized&&(t=_t(t,this.array),i=_t(i,this.array),s=_t(s,this.array),r=_t(r,this.array)),this.array[e+0]=t,this.array[e+1]=i,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==al&&(e.usage=this.usage),e}dispose(){this.dispatchEvent({type:"dispose"})}};var la=class extends Yt{constructor(e,t,i){super(new Uint16Array(e),t,i)}};var ca=class extends Yt{constructor(e,t,i){super(new Uint32Array(e),t,i)}};var nt=class extends Yt{constructor(e,t,i){super(new Float32Array(e),t,i)}},ig=new di,Gr=new A,Eh=new A,Ci=class{constructor(e=new A,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){let i=this.center;t!==void 0?i.copy(t):ig.setFromPoints(e).getCenter(i);let s=0;for(let r=0,a=e.length;r<a;r++)s=Math.max(s,i.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){let t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){let i=this.center.distanceToSquared(e);return t.copy(e),i>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Gr.subVectors(e,this.center);let t=Gr.lengthSq();if(t>this.radius*this.radius){let i=Math.sqrt(t),s=(i-this.radius)*.5;this.center.addScaledVector(Gr,s/i),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Eh.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Gr.copy(e.center).add(Eh)),this.expandByPoint(Gr.copy(e.center).sub(Eh))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}},ng=0,Oi=new rt,wh=new pt,er=new A,wi=new di,Wr=new di,qt=new A,mt=class n extends Qi{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:ng++}),this.uuid=gn(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(Am(e)?ca:la)(e,1):this.index=e,this}setIndirect(e,t=0){return this.indirect=e,this.indirectOffset=t,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,i=0){this.groups.push({start:e,count:t,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){let t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);let i=this.attributes.normal;if(i!==void 0){let r=new je().getNormalMatrix(e);i.applyNormalMatrix(r),i.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(e){return Oi.makeRotationFromQuaternion(e),this.applyMatrix4(Oi),this}rotateX(e){return Oi.makeRotationX(e),this.applyMatrix4(Oi),this}rotateY(e){return Oi.makeRotationY(e),this.applyMatrix4(Oi),this}rotateZ(e){return Oi.makeRotationZ(e),this.applyMatrix4(Oi),this}translate(e,t,i){return Oi.makeTranslation(e,t,i),this.applyMatrix4(Oi),this}scale(e,t,i){return Oi.makeScale(e,t,i),this.applyMatrix4(Oi),this}lookAt(e){return wh.lookAt(e),wh.updateMatrix(),this.applyMatrix4(wh.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(er).negate(),this.translate(er.x,er.y,er.z),this}setFromPoints(e){let t=this.getAttribute("position");if(t===void 0){let i=[];for(let s=0,r=e.length;s<r;s++){let a=e[s];i.push(a.x,a.y,a.z||0)}this.setAttribute("position",new nt(i,3))}else{let i=Math.min(e.length,t.count);for(let s=0;s<i;s++){let r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&Ye("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new di);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){$e("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new A(-1/0,-1/0,-1/0),new A(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let i=0,s=t.length;i<s;i++){let r=t[i];wi.setFromBufferAttribute(r),this.morphTargetsRelative?(qt.addVectors(this.boundingBox.min,wi.min),this.boundingBox.expandByPoint(qt),qt.addVectors(this.boundingBox.max,wi.max),this.boundingBox.expandByPoint(qt)):(this.boundingBox.expandByPoint(wi.min),this.boundingBox.expandByPoint(wi.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&$e('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Ci);let e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){$e("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new A,1/0);return}if(e){let i=this.boundingSphere.center;if(wi.setFromBufferAttribute(e),t)for(let r=0,a=t.length;r<a;r++){let o=t[r];Wr.setFromBufferAttribute(o),this.morphTargetsRelative?(qt.addVectors(wi.min,Wr.min),wi.expandByPoint(qt),qt.addVectors(wi.max,Wr.max),wi.expandByPoint(qt)):(wi.expandByPoint(Wr.min),wi.expandByPoint(Wr.max))}wi.getCenter(i);let s=0;for(let r=0,a=e.count;r<a;r++)qt.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared(qt));if(t)for(let r=0,a=t.length;r<a;r++){let o=t[r],c=this.morphTargetsRelative;for(let l=0,h=o.count;l<h;l++)qt.fromBufferAttribute(o,l),c&&(er.fromBufferAttribute(e,l),qt.add(er)),s=Math.max(s,i.distanceToSquared(qt))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&$e('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){$e("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let i=t.position,s=t.normal,r=t.uv,a=this.getAttribute("tangent");(a===void 0||a.count!==i.count)&&(a=new Yt(new Float32Array(4*i.count),4),this.setAttribute("tangent",a));let o=[],c=[];for(let _=0;_<i.count;_++)o[_]=new A,c[_]=new A;let l=new A,h=new A,d=new A,u=new te,f=new te,g=new te,x=new A,p=new A;function m(_,E,P){l.fromBufferAttribute(i,_),h.fromBufferAttribute(i,E),d.fromBufferAttribute(i,P),u.fromBufferAttribute(r,_),f.fromBufferAttribute(r,E),g.fromBufferAttribute(r,P),h.sub(l),d.sub(l),f.sub(u),g.sub(u);let I=1/(f.x*g.y-g.x*f.y);isFinite(I)&&(x.copy(h).multiplyScalar(g.y).addScaledVector(d,-f.y).multiplyScalar(I),p.copy(d).multiplyScalar(f.x).addScaledVector(h,-g.x).multiplyScalar(I),o[_].add(x),o[E].add(x),o[P].add(x),c[_].add(p),c[E].add(p),c[P].add(p))}let M=this.groups;M.length===0&&(M=[{start:0,count:e.count}]);for(let _=0,E=M.length;_<E;++_){let P=M[_],I=P.start,L=P.count;for(let X=I,W=I+L;X<W;X+=3)m(e.getX(X+0),e.getX(X+1),e.getX(X+2))}let b=new A,v=new A,T=new A,w=new A;function C(_){T.fromBufferAttribute(s,_),w.copy(T);let E=o[_];b.copy(E),b.sub(T.multiplyScalar(T.dot(E))).normalize(),v.crossVectors(w,E);let I=v.dot(c[_])<0?-1:1;a.setXYZW(_,b.x,b.y,b.z,I)}for(let _=0,E=M.length;_<E;++_){let P=M[_],I=P.start,L=P.count;for(let X=I,W=I+L;X<W;X+=3)C(e.getX(X+0)),C(e.getX(X+1)),C(e.getX(X+2))}this._transformed=!0}computeVertexNormals(){let e=this.index,t=this.getAttribute("position");if(t!==void 0){let i=this.getAttribute("normal");if(i===void 0||i.count!==t.count)i=new Yt(new Float32Array(t.count*3),3),this.setAttribute("normal",i);else for(let u=0,f=i.count;u<f;u++)i.setXYZ(u,0,0,0);let s=new A,r=new A,a=new A,o=new A,c=new A,l=new A,h=new A,d=new A;if(e)for(let u=0,f=e.count;u<f;u+=3){let g=e.getX(u+0),x=e.getX(u+1),p=e.getX(u+2);s.fromBufferAttribute(t,g),r.fromBufferAttribute(t,x),a.fromBufferAttribute(t,p),h.subVectors(a,r),d.subVectors(s,r),h.cross(d),o.fromBufferAttribute(i,g),c.fromBufferAttribute(i,x),l.fromBufferAttribute(i,p),o.add(h),c.add(h),l.add(h),i.setXYZ(g,o.x,o.y,o.z),i.setXYZ(x,c.x,c.y,c.z),i.setXYZ(p,l.x,l.y,l.z)}else for(let u=0,f=t.count;u<f;u+=3)s.fromBufferAttribute(t,u+0),r.fromBufferAttribute(t,u+1),a.fromBufferAttribute(t,u+2),h.subVectors(a,r),d.subVectors(s,r),h.cross(d),i.setXYZ(u+0,h.x,h.y,h.z),i.setXYZ(u+1,h.x,h.y,h.z),i.setXYZ(u+2,h.x,h.y,h.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){let e=this.attributes.normal;for(let t=0,i=e.count;t<i;t++)qt.fromBufferAttribute(e,t),qt.normalize(),e.setXYZ(t,qt.x,qt.y,qt.z)}toNonIndexed(){function e(o,c){let l=o.array,h=o.itemSize,d=o.normalized,u=new l.constructor(c.length*h),f=0,g=0;for(let x=0,p=c.length;x<p;x++){o.isInterleavedBufferAttribute?f=c[x]*o.data.stride+o.offset:f=c[x]*h;for(let m=0;m<h;m++)u[g++]=l[f++]}return new Yt(u,h,d)}if(this.index===null)return Ye("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let t=new n,i=this.index.array,s=this.attributes;for(let o in s){let c=s[o],l=e(c,i);t.setAttribute(o,l)}let r=this.morphAttributes;for(let o in r){let c=[],l=r[o];for(let h=0,d=l.length;h<d;h++){let u=l[h],f=e(u,i);c.push(f)}t.morphAttributes[o]=c}t.morphTargetsRelative=this.morphTargetsRelative;let a=this.groups;for(let o=0,c=a.length;o<c;o++){let l=a[o];t.addGroup(l.start,l.count,l.materialIndex)}return t}toJSON(){let e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.parameters!==void 0&&this._transformed===!0?"BufferGeometry":this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let c=this.parameters;for(let l in c)c[l]!==void 0&&(e[l]=c[l]);return e}e.data={attributes:{}};let t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});let i=this.attributes;for(let c in i){let l=i[c];e.data.attributes[c]=l.toJSON(e.data)}let s={},r=!1;for(let c in this.morphAttributes){let l=this.morphAttributes[c],h=[];for(let d=0,u=l.length;d<u;d++){let f=l[d];h.push(f.toJSON(e.data))}h.length>0&&(s[c]=h,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);let a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));let o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let t={};this.name=e.name;let i=e.index;i!==null&&this.setIndex(i.clone());let s=e.attributes;for(let l in s){let h=s[l];this.setAttribute(l,h.clone(t))}let r=e.morphAttributes;for(let l in r){let h=[],d=r[l];for(let u=0,f=d.length;u<f;u++)h.push(d[u].clone(t));this.morphAttributes[l]=h}this.morphTargetsRelative=e.morphTargetsRelative;let a=e.groups;for(let l=0,h=a.length;l<h;l++){let d=a[l];this.addGroup(d.start,d.count,d.materialIndex)}let o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());let c=e.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this._transformed=e._transformed,this}dispose(){this.dispatchEvent({type:"dispose"})}},ha=class{constructor(e,t){this.isInterleavedBuffer=!0,this.array=e,this.stride=t,this.count=e!==void 0?e.length/t:0,this.usage=al,this.updateRanges=[],this.version=0,this.uuid=gn()}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.array=new e.array.constructor(e.array),this.count=e.count,this.stride=e.stride,this.usage=e.usage,this}copyAt(e,t,i){e*=this.stride,i*=t.stride;for(let s=0,r=this.stride;s<r;s++)this.array[e+s]=t.array[i+s];return this}set(e,t=0){return this.array.set(e,t),this}clone(e){e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=gn()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);let t=new this.array.constructor(e.arrayBuffers[this.array.buffer._uuid]),i=new this.constructor(t,this.stride);return i.setUsage(this.usage),i}onUpload(e){return this.onUploadCallback=e,this}toJSON(e){return e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=gn()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=Array.from(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}},hi=new A,Pi=class n{constructor(e,t,i,s=!1){this.isInterleavedBufferAttribute=!0,this.name="",this.data=e,this.itemSize=t,this.offset=i,this.normalized=s}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(e){this.data.needsUpdate=e}applyMatrix4(e){for(let t=0,i=this.data.count;t<i;t++)hi.fromBufferAttribute(this,t),hi.applyMatrix4(e),this.setXYZ(t,hi.x,hi.y,hi.z);return this}applyNormalMatrix(e){for(let t=0,i=this.count;t<i;t++)hi.fromBufferAttribute(this,t),hi.applyNormalMatrix(e),this.setXYZ(t,hi.x,hi.y,hi.z);return this}transformDirection(e){for(let t=0,i=this.count;t<i;t++)hi.fromBufferAttribute(this,t),hi.transformDirection(e),this.setXYZ(t,hi.x,hi.y,hi.z);return this}getComponent(e,t){let i=this.array[e*this.data.stride+this.offset+t];return this.normalized&&(i=Ji(i,this.array)),i}setComponent(e,t,i){return this.normalized&&(i=_t(i,this.array)),this.data.array[e*this.data.stride+this.offset+t]=i,this}setX(e,t){return this.normalized&&(t=_t(t,this.array)),this.data.array[e*this.data.stride+this.offset]=t,this}setY(e,t){return this.normalized&&(t=_t(t,this.array)),this.data.array[e*this.data.stride+this.offset+1]=t,this}setZ(e,t){return this.normalized&&(t=_t(t,this.array)),this.data.array[e*this.data.stride+this.offset+2]=t,this}setW(e,t){return this.normalized&&(t=_t(t,this.array)),this.data.array[e*this.data.stride+this.offset+3]=t,this}getX(e){let t=this.data.array[e*this.data.stride+this.offset];return this.normalized&&(t=Ji(t,this.array)),t}getY(e){let t=this.data.array[e*this.data.stride+this.offset+1];return this.normalized&&(t=Ji(t,this.array)),t}getZ(e){let t=this.data.array[e*this.data.stride+this.offset+2];return this.normalized&&(t=Ji(t,this.array)),t}getW(e){let t=this.data.array[e*this.data.stride+this.offset+3];return this.normalized&&(t=Ji(t,this.array)),t}setXY(e,t,i){return e=e*this.data.stride+this.offset,this.normalized&&(t=_t(t,this.array),i=_t(i,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this}setXYZ(e,t,i,s){return e=e*this.data.stride+this.offset,this.normalized&&(t=_t(t,this.array),i=_t(i,this.array),s=_t(s,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this.data.array[e+2]=s,this}setXYZW(e,t,i,s,r){return e=e*this.data.stride+this.offset,this.normalized&&(t=_t(t,this.array),i=_t(i,this.array),s=_t(s,this.array),r=_t(r,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this.data.array[e+2]=s,this.data.array[e+3]=r,this}clone(e){if(e===void 0){ra("InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.");let t=[];for(let i=0;i<this.count;i++){let s=i*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return new Yt(new this.array.constructor(t),this.itemSize,this.normalized)}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.clone(e)),new n(e.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(e){if(e===void 0){ra("InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.");let t=[];for(let i=0;i<this.count;i++){let s=i*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:t,normalized:this.normalized}}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.toJSON(e)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}},sg=0,ki=class extends Qi{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:sg++}),this.uuid=gn(),this.name="",this.type="Material",this.blending=As,this.side=ji,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Zo,this.blendDst=Jo,this.blendEquation=Ti,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Le(0,0,0),this.blendAlpha=0,this.depthFunc=Rs,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Gh,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Es,this.stencilZFail=Es,this.stencilZPass=Es,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(let t in e){let i=e[t];if(i===void 0){Ye(`Material: parameter '${t}' has value of undefined.`);continue}let s=this[t];if(s===void 0){Ye(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(i):s&&s.isVector2&&i&&i.isVector2||s&&s.isEuler&&i&&i.isEuler||s&&s.isVector3&&i&&i.isVector3?s.copy(i):this[t]=i}}toJSON(e){let t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});let i={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,this.name!==""&&(i.name=this.name),this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(i.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(i.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(i.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(e).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(e).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(e).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(e).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(e).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.shadowSide!==null&&(i.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),this.blending!==As&&(i.blending=this.blending),this.side!==ji&&(i.side=this.side),this.vertexColors===!0&&(i.vertexColors=!0),this.opacity<1&&(i.opacity=this.opacity),this.transparent===!0&&(i.transparent=!0),this.blendSrc!==Zo&&(i.blendSrc=this.blendSrc),this.blendDst!==Jo&&(i.blendDst=this.blendDst),this.blendEquation!==Ti&&(i.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(i.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(i.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(i.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(i.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(i.blendAlpha=this.blendAlpha),this.depthFunc!==Rs&&(i.depthFunc=this.depthFunc),this.depthTest===!1&&(i.depthTest=this.depthTest),this.depthWrite===!1&&(i.depthWrite=this.depthWrite),this.colorWrite===!1&&(i.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(i.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Gh&&(i.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(i.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(i.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Es&&(i.stencilFail=this.stencilFail),this.stencilZFail!==Es&&(i.stencilZFail=this.stencilZFail),this.stencilZPass!==Es&&(i.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(i.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(i.rotation=this.rotation),this.polygonOffset===!0&&(i.polygonOffset=!0),this.polygonOffsetFactor!==0&&(i.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(i.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(i.linewidth=this.linewidth),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.dithering===!0&&(i.dithering=!0),this.alphaTest>0&&(i.alphaTest=this.alphaTest),this.alphaHash===!0&&(i.alphaHash=!0),this.alphaToCoverage===!0&&(i.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(i.premultipliedAlpha=!0),this.forceSinglePass===!0&&(i.forceSinglePass=!0),this.allowOverride===!1&&(i.allowOverride=!1),this.wireframe===!0&&(i.wireframe=!0),this.wireframeLinewidth>1&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(i.flatShading=!0),this.visible===!1&&(i.visible=!1),this.toneMapped===!1&&(i.toneMapped=!1),this.fog===!1&&(i.fog=!1),Object.keys(this.userData).length>0&&(i.userData=this.userData);function s(r){let a=[];for(let o in r){let c=r[o];delete c.metadata,a.push(c)}return a}if(t){let r=s(e.textures),a=s(e.images);r.length>0&&(i.textures=r),a.length>0&&(i.images=a)}return i}fromJSON(e,t){if(e.uuid!==void 0&&(this.uuid=e.uuid),e.name!==void 0&&(this.name=e.name),e.color!==void 0&&this.color!==void 0&&this.color.setHex(e.color),e.roughness!==void 0&&(this.roughness=e.roughness),e.metalness!==void 0&&(this.metalness=e.metalness),e.sheen!==void 0&&(this.sheen=e.sheen),e.sheenColor!==void 0&&(this.sheenColor=new Le().setHex(e.sheenColor)),e.sheenRoughness!==void 0&&(this.sheenRoughness=e.sheenRoughness),e.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(e.emissive),e.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(e.specular),e.specularIntensity!==void 0&&(this.specularIntensity=e.specularIntensity),e.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(e.specularColor),e.shininess!==void 0&&(this.shininess=e.shininess),e.clearcoat!==void 0&&(this.clearcoat=e.clearcoat),e.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=e.clearcoatRoughness),e.dispersion!==void 0&&(this.dispersion=e.dispersion),e.iridescence!==void 0&&(this.iridescence=e.iridescence),e.iridescenceIOR!==void 0&&(this.iridescenceIOR=e.iridescenceIOR),e.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessRange=e.iridescenceThicknessRange),e.transmission!==void 0&&(this.transmission=e.transmission),e.thickness!==void 0&&(this.thickness=e.thickness),e.attenuationDistance!==void 0&&(this.attenuationDistance=e.attenuationDistance),e.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(e.attenuationColor),e.anisotropy!==void 0&&(this.anisotropy=e.anisotropy),e.anisotropyRotation!==void 0&&(this.anisotropyRotation=e.anisotropyRotation),e.fog!==void 0&&(this.fog=e.fog),e.flatShading!==void 0&&(this.flatShading=e.flatShading),e.blending!==void 0&&(this.blending=e.blending),e.combine!==void 0&&(this.combine=e.combine),e.side!==void 0&&(this.side=e.side),e.shadowSide!==void 0&&(this.shadowSide=e.shadowSide),e.opacity!==void 0&&(this.opacity=e.opacity),e.transparent!==void 0&&(this.transparent=e.transparent),e.alphaTest!==void 0&&(this.alphaTest=e.alphaTest),e.alphaHash!==void 0&&(this.alphaHash=e.alphaHash),e.depthFunc!==void 0&&(this.depthFunc=e.depthFunc),e.depthTest!==void 0&&(this.depthTest=e.depthTest),e.depthWrite!==void 0&&(this.depthWrite=e.depthWrite),e.colorWrite!==void 0&&(this.colorWrite=e.colorWrite),e.blendSrc!==void 0&&(this.blendSrc=e.blendSrc),e.blendDst!==void 0&&(this.blendDst=e.blendDst),e.blendEquation!==void 0&&(this.blendEquation=e.blendEquation),e.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=e.blendSrcAlpha),e.blendDstAlpha!==void 0&&(this.blendDstAlpha=e.blendDstAlpha),e.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=e.blendEquationAlpha),e.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(e.blendColor),e.blendAlpha!==void 0&&(this.blendAlpha=e.blendAlpha),e.stencilWriteMask!==void 0&&(this.stencilWriteMask=e.stencilWriteMask),e.stencilFunc!==void 0&&(this.stencilFunc=e.stencilFunc),e.stencilRef!==void 0&&(this.stencilRef=e.stencilRef),e.stencilFuncMask!==void 0&&(this.stencilFuncMask=e.stencilFuncMask),e.stencilFail!==void 0&&(this.stencilFail=e.stencilFail),e.stencilZFail!==void 0&&(this.stencilZFail=e.stencilZFail),e.stencilZPass!==void 0&&(this.stencilZPass=e.stencilZPass),e.stencilWrite!==void 0&&(this.stencilWrite=e.stencilWrite),e.wireframe!==void 0&&(this.wireframe=e.wireframe),e.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=e.wireframeLinewidth),e.wireframeLinecap!==void 0&&(this.wireframeLinecap=e.wireframeLinecap),e.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=e.wireframeLinejoin),e.rotation!==void 0&&(this.rotation=e.rotation),e.linewidth!==void 0&&(this.linewidth=e.linewidth),e.dashSize!==void 0&&(this.dashSize=e.dashSize),e.gapSize!==void 0&&(this.gapSize=e.gapSize),e.scale!==void 0&&(this.scale=e.scale),e.polygonOffset!==void 0&&(this.polygonOffset=e.polygonOffset),e.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=e.polygonOffsetFactor),e.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=e.polygonOffsetUnits),e.dithering!==void 0&&(this.dithering=e.dithering),e.alphaToCoverage!==void 0&&(this.alphaToCoverage=e.alphaToCoverage),e.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=e.premultipliedAlpha),e.forceSinglePass!==void 0&&(this.forceSinglePass=e.forceSinglePass),e.allowOverride!==void 0&&(this.allowOverride=e.allowOverride),e.visible!==void 0&&(this.visible=e.visible),e.toneMapped!==void 0&&(this.toneMapped=e.toneMapped),e.userData!==void 0&&(this.userData=e.userData),e.vertexColors!==void 0&&(typeof e.vertexColors=="number"?this.vertexColors=e.vertexColors>0:this.vertexColors=e.vertexColors),e.size!==void 0&&(this.size=e.size),e.sizeAttenuation!==void 0&&(this.sizeAttenuation=e.sizeAttenuation),e.map!==void 0&&(this.map=t[e.map]||null),e.matcap!==void 0&&(this.matcap=t[e.matcap]||null),e.alphaMap!==void 0&&(this.alphaMap=t[e.alphaMap]||null),e.bumpMap!==void 0&&(this.bumpMap=t[e.bumpMap]||null),e.bumpScale!==void 0&&(this.bumpScale=e.bumpScale),e.normalMap!==void 0&&(this.normalMap=t[e.normalMap]||null),e.normalMapType!==void 0&&(this.normalMapType=e.normalMapType),e.normalScale!==void 0){let i=e.normalScale;Array.isArray(i)===!1&&(i=[i,i]),this.normalScale=new te().fromArray(i)}return e.displacementMap!==void 0&&(this.displacementMap=t[e.displacementMap]||null),e.displacementScale!==void 0&&(this.displacementScale=e.displacementScale),e.displacementBias!==void 0&&(this.displacementBias=e.displacementBias),e.roughnessMap!==void 0&&(this.roughnessMap=t[e.roughnessMap]||null),e.metalnessMap!==void 0&&(this.metalnessMap=t[e.metalnessMap]||null),e.emissiveMap!==void 0&&(this.emissiveMap=t[e.emissiveMap]||null),e.emissiveIntensity!==void 0&&(this.emissiveIntensity=e.emissiveIntensity),e.specularMap!==void 0&&(this.specularMap=t[e.specularMap]||null),e.specularIntensityMap!==void 0&&(this.specularIntensityMap=t[e.specularIntensityMap]||null),e.specularColorMap!==void 0&&(this.specularColorMap=t[e.specularColorMap]||null),e.envMap!==void 0&&(this.envMap=t[e.envMap]||null),e.envMapRotation!==void 0&&this.envMapRotation.fromArray(e.envMapRotation),e.envMapIntensity!==void 0&&(this.envMapIntensity=e.envMapIntensity),e.reflectivity!==void 0&&(this.reflectivity=e.reflectivity),e.refractionRatio!==void 0&&(this.refractionRatio=e.refractionRatio),e.lightMap!==void 0&&(this.lightMap=t[e.lightMap]||null),e.lightMapIntensity!==void 0&&(this.lightMapIntensity=e.lightMapIntensity),e.aoMap!==void 0&&(this.aoMap=t[e.aoMap]||null),e.aoMapIntensity!==void 0&&(this.aoMapIntensity=e.aoMapIntensity),e.gradientMap!==void 0&&(this.gradientMap=t[e.gradientMap]||null),e.clearcoatMap!==void 0&&(this.clearcoatMap=t[e.clearcoatMap]||null),e.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=t[e.clearcoatRoughnessMap]||null),e.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=t[e.clearcoatNormalMap]||null),e.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new te().fromArray(e.clearcoatNormalScale)),e.iridescenceMap!==void 0&&(this.iridescenceMap=t[e.iridescenceMap]||null),e.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=t[e.iridescenceThicknessMap]||null),e.transmissionMap!==void 0&&(this.transmissionMap=t[e.transmissionMap]||null),e.thicknessMap!==void 0&&(this.thicknessMap=t[e.thicknessMap]||null),e.anisotropyMap!==void 0&&(this.anisotropyMap=t[e.anisotropyMap]||null),e.sheenColorMap!==void 0&&(this.sheenColorMap=t[e.sheenColorMap]||null),e.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=t[e.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;let t=e.clippingPlanes,i=null;if(t!==null){let s=t.length;i=new Array(s);for(let r=0;r!==s;++r)i[r]=t[r].clone()}return this.clippingPlanes=i,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}},xr=class extends ki{constructor(e){super(),this.isSpriteMaterial=!0,this.type="SpriteMaterial",this.color=new Le(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.rotation=e.rotation,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}},tr,Xr=new A,ir=new A,nr=new A,sr=new te,qr=new te,kf=new rt,So=new A,Yr=new A,Eo=new A,Nd=new te,Th=new te,Ud=new te,ua=class extends pt{constructor(e=new xr){if(super(),this.isSprite=!0,this.type="Sprite",tr===void 0){tr=new mt;let t=new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),i=new ha(t,5);tr.setIndex([0,1,2,0,2,3]),tr.setAttribute("position",new Pi(i,3,0,!1)),tr.setAttribute("uv",new Pi(i,2,3,!1))}this.geometry=tr,this.material=e,this.center=new te(.5,.5),this.count=1}raycast(e,t){e.camera===null&&$e('Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.'),ir.setFromMatrixScale(this.matrixWorld),kf.copy(e.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(e.camera.matrixWorldInverse,this.matrixWorld),nr.setFromMatrixPosition(this.modelViewMatrix),e.camera.isPerspectiveCamera&&this.material.sizeAttenuation===!1&&ir.multiplyScalar(-nr.z);let i=this.material.rotation,s,r;i!==0&&(r=Math.cos(i),s=Math.sin(i));let a=this.center;wo(So.set(-.5,-.5,0),nr,a,ir,s,r),wo(Yr.set(.5,-.5,0),nr,a,ir,s,r),wo(Eo.set(.5,.5,0),nr,a,ir,s,r),Nd.set(0,0),Th.set(1,0),Ud.set(1,1);let o=e.ray.intersectTriangle(So,Yr,Eo,!1,Xr);if(o===null&&(wo(Yr.set(-.5,.5,0),nr,a,ir,s,r),Th.set(0,1),o=e.ray.intersectTriangle(So,Eo,Yr,!1,Xr),o===null))return;let c=e.ray.origin.distanceTo(Xr);c<e.near||c>e.far||t.push({distance:c,point:Xr.clone(),uv:pn.getInterpolation(Xr,So,Yr,Eo,Nd,Th,Ud,new te),face:null,object:this})}copy(e,t){return super.copy(e,t),e.center!==void 0&&this.center.copy(e.center),this.material=e.material,this}};function wo(n,e,t,i,s,r){sr.subVectors(n,t).addScalar(.5).multiply(i),s!==void 0?(qr.x=r*sr.x-s*sr.y,qr.y=s*sr.x+r*sr.y):qr.copy(sr),n.copy(e),n.x+=qr.x,n.y+=qr.y,n.applyMatrix4(kf)}var Cn=new A,Ah=new A,To=new A,Jn=new A,Rh=new A,Ao=new A,Ch=new A,jn=class{constructor(e=new A,t=new A(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Cn)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);let i=t.dot(this.direction);return i<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){let t=Cn.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(Cn.copy(this.origin).addScaledVector(this.direction,t),Cn.distanceToSquared(e))}distanceSqToSegment(e,t,i,s){Ah.copy(e).add(t).multiplyScalar(.5),To.copy(t).sub(e).normalize(),Jn.copy(this.origin).sub(Ah);let r=e.distanceTo(t)*.5,a=-this.direction.dot(To),o=Jn.dot(this.direction),c=-Jn.dot(To),l=Jn.lengthSq(),h=Math.abs(1-a*a),d,u,f,g;if(h>0)if(d=a*c-o,u=a*o-c,g=r*h,d>=0)if(u>=-g)if(u<=g){let x=1/h;d*=x,u*=x,f=d*(d+a*u+2*o)+u*(a*d+u+2*c)+l}else u=r,d=Math.max(0,-(a*u+o)),f=-d*d+u*(u+2*c)+l;else u=-r,d=Math.max(0,-(a*u+o)),f=-d*d+u*(u+2*c)+l;else u<=-g?(d=Math.max(0,-(-a*r+o)),u=d>0?-r:Math.min(Math.max(-r,-c),r),f=-d*d+u*(u+2*c)+l):u<=g?(d=0,u=Math.min(Math.max(-r,-c),r),f=u*(u+2*c)+l):(d=Math.max(0,-(a*r+o)),u=d>0?r:Math.min(Math.max(-r,-c),r),f=-d*d+u*(u+2*c)+l);else u=a>0?-r:r,d=Math.max(0,-(a*u+o)),f=-d*d+u*(u+2*c)+l;return i&&i.copy(this.origin).addScaledVector(this.direction,d),s&&s.copy(Ah).addScaledVector(To,u),f}intersectSphere(e,t){Cn.subVectors(e.center,this.origin);let i=Cn.dot(this.direction),s=Cn.dot(Cn)-i*i,r=e.radius*e.radius;if(s>r)return null;let a=Math.sqrt(r-s),o=i-a,c=i+a;return c<0?null:o<0?this.at(c,t):this.at(o,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){let t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;let i=-(this.origin.dot(e.normal)+e.constant)/t;return i>=0?i:null}intersectPlane(e,t){let i=this.distanceToPlane(e);return i===null?null:this.at(i,t)}intersectsPlane(e){let t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let i,s,r,a,o,c,l=1/this.direction.x,h=1/this.direction.y,d=1/this.direction.z,u=this.origin;return l>=0?(i=(e.min.x-u.x)*l,s=(e.max.x-u.x)*l):(i=(e.max.x-u.x)*l,s=(e.min.x-u.x)*l),h>=0?(r=(e.min.y-u.y)*h,a=(e.max.y-u.y)*h):(r=(e.max.y-u.y)*h,a=(e.min.y-u.y)*h),i>a||r>s||((r>i||isNaN(i))&&(i=r),(a<s||isNaN(s))&&(s=a),d>=0?(o=(e.min.z-u.z)*d,c=(e.max.z-u.z)*d):(o=(e.max.z-u.z)*d,c=(e.min.z-u.z)*d),i>c||o>s)||((o>i||i!==i)&&(i=o),(c<s||s!==s)&&(s=c),s<0)?null:this.at(i>=0?i:s,t)}intersectsBox(e){return this.intersectBox(e,Cn)!==null}intersectTriangle(e,t,i,s,r){Rh.subVectors(t,e),Ao.subVectors(i,e),Ch.crossVectors(Rh,Ao);let a=this.direction.dot(Ch),o;if(a>0){if(s)return null;o=1}else if(a<0)o=-1,a=-a;else return null;Jn.subVectors(this.origin,e);let c=o*this.direction.dot(Ao.crossVectors(Jn,Ao));if(c<0)return null;let l=o*this.direction.dot(Rh.cross(Jn));if(l<0||c+l>a)return null;let h=-o*Jn.dot(Ch);return h<0?null:this.at(h/a,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},In=class extends ki{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Le(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ri,this.combine=Ol,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}},Fd=new rt,bs=new jn,Ro=new Ci,Od=new A,Co=new A,Po=new A,Io=new A,Ph=new A,Do=new A,Bd=new A,Lo=new A,tt=class extends pt{constructor(e=new mt,t=new In){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){let t=this.geometry.morphAttributes,i=Object.keys(t);if(i.length>0){let s=t[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(e,t){let i=this.geometry,s=i.attributes.position,r=i.morphAttributes.position,a=i.morphTargetsRelative;t.fromBufferAttribute(s,e);let o=this.morphTargetInfluences;if(r&&o){Do.set(0,0,0);for(let c=0,l=r.length;c<l;c++){let h=o[c],d=r[c];h!==0&&(Ph.fromBufferAttribute(d,e),a?Do.addScaledVector(Ph,h):Do.addScaledVector(Ph.sub(t),h))}t.add(Do)}return t}raycast(e,t){let i=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),Ro.copy(i.boundingSphere),Ro.applyMatrix4(r),bs.copy(e.ray).recast(e.near),!(Ro.containsPoint(bs.origin)===!1&&(bs.intersectSphere(Ro,Od)===null||bs.origin.distanceToSquared(Od)>(e.far-e.near)**2))&&(Fd.copy(r).invert(),bs.copy(e.ray).applyMatrix4(Fd),!(i.boundingBox!==null&&bs.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(e,t,bs)))}_computeIntersections(e,t,i){let s,r=this.geometry,a=this.material,o=r.index,c=r.attributes.position,l=r.attributes.uv,h=r.attributes.uv1,d=r.attributes.normal,u=r.groups,f=r.drawRange;if(o!==null)if(Array.isArray(a))for(let g=0,x=u.length;g<x;g++){let p=u[g],m=a[p.materialIndex],M=Math.max(p.start,f.start),b=Math.min(o.count,Math.min(p.start+p.count,f.start+f.count));for(let v=M,T=b;v<T;v+=3){let w=o.getX(v),C=o.getX(v+1),_=o.getX(v+2);s=No(this,m,e,i,l,h,d,w,C,_),s&&(s.faceIndex=Math.floor(v/3),s.face.materialIndex=p.materialIndex,t.push(s))}}else{let g=Math.max(0,f.start),x=Math.min(o.count,f.start+f.count);for(let p=g,m=x;p<m;p+=3){let M=o.getX(p),b=o.getX(p+1),v=o.getX(p+2);s=No(this,a,e,i,l,h,d,M,b,v),s&&(s.faceIndex=Math.floor(p/3),t.push(s))}}else if(c!==void 0)if(Array.isArray(a))for(let g=0,x=u.length;g<x;g++){let p=u[g],m=a[p.materialIndex],M=Math.max(p.start,f.start),b=Math.min(c.count,Math.min(p.start+p.count,f.start+f.count));for(let v=M,T=b;v<T;v+=3){let w=v,C=v+1,_=v+2;s=No(this,m,e,i,l,h,d,w,C,_),s&&(s.faceIndex=Math.floor(v/3),s.face.materialIndex=p.materialIndex,t.push(s))}}else{let g=Math.max(0,f.start),x=Math.min(c.count,f.start+f.count);for(let p=g,m=x;p<m;p+=3){let M=p,b=p+1,v=p+2;s=No(this,a,e,i,l,h,d,M,b,v),s&&(s.faceIndex=Math.floor(p/3),t.push(s))}}}};function rg(n,e,t,i,s,r,a,o){let c;if(e.side===Qt?c=i.intersectTriangle(a,r,s,!0,o):c=i.intersectTriangle(s,r,a,e.side===ji,o),c===null)return null;Lo.copy(o),Lo.applyMatrix4(n.matrixWorld);let l=t.ray.origin.distanceTo(Lo);return l<t.near||l>t.far?null:{distance:l,point:Lo.clone(),object:n}}function No(n,e,t,i,s,r,a,o,c,l){n.getVertexPosition(o,Co),n.getVertexPosition(c,Po),n.getVertexPosition(l,Io);let h=rg(n,e,t,i,Co,Po,Io,Bd);if(h){let d=new A;pn.getBarycoord(Bd,Co,Po,Io,d),s&&(h.uv=pn.getInterpolatedAttribute(s,o,c,l,d,new te)),r&&(h.uv1=pn.getInterpolatedAttribute(r,o,c,l,d,new te)),a&&(h.normal=pn.getInterpolatedAttribute(a,o,c,l,d,new A),h.normal.dot(i.direction)>0&&h.normal.multiplyScalar(-1));let u={a:o,b:c,c:l,normal:new A,materialIndex:0};pn.getNormal(Co,Po,Io,u.normal),h.face=u,h.barycoord=d}return h}var Dn=class extends ui{constructor(e=null,t=1,i=1,s,r,a,o,c,l=Ot,h=Ot,d,u){super(null,a,o,c,l,h,s,r,d,u),this.isDataTexture=!0,this.image={data:e,width:t,height:i},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var da=class extends Yt{constructor(e,t,i,s=1){super(e,t,i),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){let e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}},rr=new rt,zd=new rt,Uo=[],kd=new di,ag=new rt,$r=new tt,Zr=new Ci,$t=class extends tt{constructor(e,t,i){super(e,t),this.isInstancedMesh=!0,this.instanceMatrix=new da(new Float32Array(i*16),16),this.instanceColor=null,this.morphTexture=null,this.count=i,this.boundingBox=null,this.boundingSphere=null;for(let s=0;s<i;s++)this.setMatrixAt(s,ag)}computeBoundingBox(){let e=this.geometry,t=this.count;this.boundingBox===null&&(this.boundingBox=new di),e.boundingBox===null&&e.computeBoundingBox(),this.boundingBox.makeEmpty();for(let i=0;i<t;i++)this.getMatrixAt(i,rr),kd.copy(e.boundingBox).applyMatrix4(rr),this.boundingBox.union(kd)}computeBoundingSphere(){let e=this.geometry,t=this.count;this.boundingSphere===null&&(this.boundingSphere=new Ci),e.boundingSphere===null&&e.computeBoundingSphere(),this.boundingSphere.makeEmpty();for(let i=0;i<t;i++)this.getMatrixAt(i,rr),Zr.copy(e.boundingSphere).applyMatrix4(rr),this.boundingSphere.union(Zr)}copy(e,t){return super.copy(e,t),this.instanceMatrix.copy(e.instanceMatrix),e.morphTexture!==null&&(this.morphTexture=e.morphTexture.clone()),e.instanceColor!==null&&(this.instanceColor=e.instanceColor.clone()),this.count=e.count,e.boundingBox!==null&&(this.boundingBox=e.boundingBox.clone()),e.boundingSphere!==null&&(this.boundingSphere=e.boundingSphere.clone()),this}getColorAt(e,t){return this.instanceColor===null?t.setRGB(1,1,1):t.fromArray(this.instanceColor.array,e*3)}getMatrixAt(e,t){return t.fromArray(this.instanceMatrix.array,e*16)}getMorphAt(e,t){let i=t.morphTargetInfluences,s=this.morphTexture.source.data.data,r=i.length+1,a=e*r+1;for(let o=0;o<i.length;o++)i[o]=s[a+o]}raycast(e,t){let i=this.matrixWorld,s=this.count;if($r.geometry=this.geometry,$r.material=this.material,$r.material!==void 0&&(this.boundingSphere===null&&this.computeBoundingSphere(),Zr.copy(this.boundingSphere),Zr.applyMatrix4(i),e.ray.intersectsSphere(Zr)!==!1))for(let r=0;r<s;r++){this.getMatrixAt(r,rr),zd.multiplyMatrices(i,rr),$r.matrixWorld=zd,$r.raycast(e,Uo);for(let a=0,o=Uo.length;a<o;a++){let c=Uo[a];c.instanceId=r,c.object=this,t.push(c)}Uo.length=0}}setColorAt(e,t){return this.instanceColor===null&&(this.instanceColor=new da(new Float32Array(this.instanceMatrix.count*3).fill(1),3)),t.toArray(this.instanceColor.array,e*3),this}setMatrixAt(e,t){return t.toArray(this.instanceMatrix.array,e*16),this}setMorphAt(e,t){let i=t.morphTargetInfluences,s=i.length+1;this.morphTexture===null&&(this.morphTexture=new Dn(new Float32Array(s*this.count),s,this.count,Wl,Vi));let r=this.morphTexture.source.data.data,a=0;for(let l=0;l<i.length;l++)a+=i[l];let o=this.geometry.morphTargetsRelative?1:1-a,c=s*e;return r[c]=o,r.set(i,c+1),this}updateMorphTargets(){}dispose(){this.dispatchEvent({type:"dispose"}),this.morphTexture!==null&&(this.morphTexture.dispose(),this.morphTexture=null)}},Ih=new A,og=new A,lg=new je,Bi=class{constructor(e=new A(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,i,s){return this.normal.set(e,t,i),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,i){let s=Ih.subVectors(i,t).cross(og.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){let e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t,i=!0){let s=e.delta(Ih),r=this.normal.dot(s);if(r===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;let a=-(e.start.dot(this.normal)+this.constant)/r;return i===!0&&(a<0||a>1)?null:t.copy(e.start).addScaledVector(s,a)}intersectsLine(e){let t=this.distanceToPoint(e.start),i=this.distanceToPoint(e.end);return t<0&&i>0||i<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){let i=t||lg.getNormalMatrix(e),s=this.coplanarPoint(Ih).applyMatrix4(e),r=this.normal.applyMatrix3(i).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}},Ss=new Ci,cg=new te(.5,.5),Fo=new A,vr=class{constructor(e=new Bi,t=new Bi,i=new Bi,s=new Bi,r=new Bi,a=new Bi){this.planes=[e,t,i,s,r,a]}set(e,t,i,s,r,a){let o=this.planes;return o[0].copy(e),o[1].copy(t),o[2].copy(i),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(e){let t=this.planes;for(let i=0;i<6;i++)t[i].copy(e.planes[i]);return this}setFromProjectionMatrix(e,t=Ki,i=!1){let s=this.planes,r=e.elements,a=r[0],o=r[1],c=r[2],l=r[3],h=r[4],d=r[5],u=r[6],f=r[7],g=r[8],x=r[9],p=r[10],m=r[11],M=r[12],b=r[13],v=r[14],T=r[15];if(s[0].setComponents(l-a,f-h,m-g,T-M).normalize(),s[1].setComponents(l+a,f+h,m+g,T+M).normalize(),s[2].setComponents(l+o,f+d,m+x,T+b).normalize(),s[3].setComponents(l-o,f-d,m-x,T-b).normalize(),i)s[4].setComponents(c,u,p,v).normalize(),s[5].setComponents(l-c,f-u,m-p,T-v).normalize();else if(s[4].setComponents(l-c,f-u,m-p,T-v).normalize(),t===Ki)s[5].setComponents(l+c,f+u,m+p,T+v).normalize();else if(t===dr)s[5].setComponents(c,u,p,v).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Ss.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{let t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Ss.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Ss)}intersectsSprite(e){Ss.center.set(0,0,0);let t=cg.distanceTo(e.center);return Ss.radius=.7071067811865476+t,Ss.applyMatrix4(e.matrixWorld),this.intersectsSphere(Ss)}intersectsSphere(e){let t=this.planes,i=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(i)<s)return!1;return!0}intersectsBox(e){let t=this.planes;for(let i=0;i<6;i++){let s=t[i];if(Fo.x=s.normal.x>0?e.max.x:e.min.x,Fo.y=s.normal.y>0?e.max.y:e.min.y,Fo.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(Fo)<0)return!1}return!0}containsPoint(e){let t=this.planes;for(let i=0;i<6;i++)if(t[i].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}};var Ln=class extends ki{constructor(e){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new Le(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.linewidth=e.linewidth,this.linecap=e.linecap,this.linejoin=e.linejoin,this.fog=e.fog,this}},hl=new A,ul=new A,Hd=new rt,Jr=new jn,Oo=new Ci,Dh=new A,Vd=new A,dl=class extends pt{constructor(e=new mt,t=new Ln){super(),this.isLine=!0,this.type="Line",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}computeLineDistances(){let e=this.geometry;if(e.index===null){let t=e.attributes.position,i=[0];for(let s=1,r=t.count;s<r;s++)hl.fromBufferAttribute(t,s-1),ul.fromBufferAttribute(t,s),i[s]=i[s-1],i[s]+=hl.distanceTo(ul);e.setAttribute("lineDistance",new nt(i,1))}else Ye("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(e,t){let i=this.geometry,s=this.matrixWorld,r=e.params.Line.threshold,a=i.drawRange;if(i.boundingSphere===null&&i.computeBoundingSphere(),Oo.copy(i.boundingSphere),Oo.applyMatrix4(s),Oo.radius+=r,e.ray.intersectsSphere(Oo)===!1)return;Hd.copy(s).invert(),Jr.copy(e.ray).applyMatrix4(Hd);let o=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=o*o,l=this.isLineSegments?2:1,h=i.index,u=i.attributes.position;if(h!==null){let f=Math.max(0,a.start),g=Math.min(h.count,a.start+a.count);for(let x=f,p=g-1;x<p;x+=l){let m=h.getX(x),M=h.getX(x+1),b=Bo(this,e,Jr,c,m,M,x);b&&t.push(b)}if(this.isLineLoop){let x=h.getX(g-1),p=h.getX(f),m=Bo(this,e,Jr,c,x,p,g-1);m&&t.push(m)}}else{let f=Math.max(0,a.start),g=Math.min(u.count,a.start+a.count);for(let x=f,p=g-1;x<p;x+=l){let m=Bo(this,e,Jr,c,x,x+1,x);m&&t.push(m)}if(this.isLineLoop){let x=Bo(this,e,Jr,c,g-1,f,g-1);x&&t.push(x)}}}updateMorphTargets(){let t=this.geometry.morphAttributes,i=Object.keys(t);if(i.length>0){let s=t[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}};function Bo(n,e,t,i,s,r,a){let o=n.geometry.attributes.position;if(hl.fromBufferAttribute(o,s),ul.fromBufferAttribute(o,r),t.distanceSqToSegment(hl,ul,Dh,Vd)>i)return;Dh.applyMatrix4(n.matrixWorld);let l=e.ray.origin.distanceTo(Dh);if(!(l<e.near||l>e.far))return{distance:l,point:Vd.clone().applyMatrix4(n.matrixWorld),index:a,face:null,faceIndex:null,barycoord:null,object:n}}var Gd=new A,Wd=new A,Qn=class extends dl{constructor(e,t){super(e,t),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){let e=this.geometry;if(e.index===null){let t=e.attributes.position,i=[];for(let s=0,r=t.count;s<r;s+=2)Gd.fromBufferAttribute(t,s),Wd.fromBufferAttribute(t,s+1),i[s]=s===0?0:i[s-1],i[s+1]=i[s]+Gd.distanceTo(Wd);e.setAttribute("lineDistance",new nt(i,1))}else Ye("LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}};var fa=class extends ui{constructor(e=[],t=ls,i,s,r,a,o,c,l,h){super(e,t,i,s,r,a,o,c,l,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}},Nn=class extends ui{constructor(e,t,i,s,r,a,o,c,l){super(e,t,i,s,r,a,o,c,l),this.isCanvasTexture=!0,this.needsUpdate=!0}};var en=class extends ui{constructor(e,t,i=an,s,r,a,o=Ot,c=Ot,l,h=_n,d=1){if(h!==_n&&h!==xn)throw new Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");let u={width:e,height:t,depth:d};super(u,s,r,a,o,c,h,i,l),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new mr(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){let t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}},fl=class extends en{constructor(e,t=an,i=ls,s,r,a=Ot,o=Ot,c,l=_n){let h={width:e,height:e,depth:1},d=[h,h,h,h,h,h];super(e,e,t,i,s,r,a,o,c,l),this.image=d,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}},pa=class extends ui{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}},Bt=class n extends mt{constructor(e=1,t=1,i=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:i,widthSegments:s,heightSegments:r,depthSegments:a};let o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);let c=[],l=[],h=[],d=[],u=0,f=0;g("z","y","x",-1,-1,i,t,e,a,r,0),g("z","y","x",1,-1,i,t,-e,a,r,1),g("x","z","y",1,1,e,i,t,s,a,2),g("x","z","y",1,-1,e,i,-t,s,a,3),g("x","y","z",1,-1,e,t,i,s,r,4),g("x","y","z",-1,-1,e,t,-i,s,r,5),this.setIndex(c),this.setAttribute("position",new nt(l,3)),this.setAttribute("normal",new nt(h,3)),this.setAttribute("uv",new nt(d,2));function g(x,p,m,M,b,v,T,w,C,_,E){let P=v/C,I=T/_,L=v/2,X=T/2,W=w/2,U=C+1,z=_+1,H=0,Q=0,ie=new A;for(let q=0;q<z;q++){let Z=q*I-X;for(let j=0;j<U;j++){let de=j*P-L;ie[x]=de*M,ie[p]=Z*b,ie[m]=W,l.push(ie.x,ie.y,ie.z),ie[x]=0,ie[p]=0,ie[m]=w>0?1:-1,h.push(ie.x,ie.y,ie.z),d.push(j/C),d.push(1-q/_),H+=1}}for(let q=0;q<_;q++)for(let Z=0;Z<C;Z++){let j=u+Z+U*q,de=u+Z+U*(q+1),Ge=u+(Z+1)+U*(q+1),me=u+(Z+1)+U*q;c.push(j,de,me),c.push(de,Ge,me),Q+=6}o.addGroup(f,Q,E),f+=Q,u+=H}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}};var tn=class n extends mt{constructor(e=1,t=1,i=1,s=32,r=1,a=!1,o=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:i,radialSegments:s,heightSegments:r,openEnded:a,thetaStart:o,thetaLength:c};let l=this;s=Math.floor(s),r=Math.floor(r);let h=[],d=[],u=[],f=[],g=0,x=[],p=i/2,m=0;M(),a===!1&&(e>0&&b(!0),t>0&&b(!1)),this.setIndex(h),this.setAttribute("position",new nt(d,3)),this.setAttribute("normal",new nt(u,3)),this.setAttribute("uv",new nt(f,2));function M(){let v=new A,T=new A,w=0,C=(t-e)/i;for(let _=0;_<=r;_++){let E=[],P=_/r,I=P*(t-e)+e;for(let L=0;L<=s;L++){let X=L/s,W=X*c+o,U=Math.sin(W),z=Math.cos(W);T.x=I*U,T.y=-P*i+p,T.z=I*z,d.push(T.x,T.y,T.z),v.set(U,C,z).normalize(),u.push(v.x,v.y,v.z),f.push(X,1-P),E.push(g++)}x.push(E)}for(let _=0;_<s;_++)for(let E=0;E<r;E++){let P=x[E][_],I=x[E+1][_],L=x[E+1][_+1],X=x[E][_+1];(e>0||E!==0)&&(h.push(P,I,X),w+=3),(t>0||E!==r-1)&&(h.push(I,L,X),w+=3)}l.addGroup(m,w,0),m+=w}function b(v){let T=g,w=new te,C=new A,_=0,E=v===!0?e:t,P=v===!0?1:-1;for(let L=1;L<=s;L++)d.push(0,p*P,0),u.push(0,P,0),f.push(.5,.5),g++;let I=g;for(let L=0;L<=s;L++){let W=L/s*c+o,U=Math.cos(W),z=Math.sin(W);C.x=E*z,C.y=p*P,C.z=E*U,d.push(C.x,C.y,C.z),u.push(0,P,0),w.x=U*.5+.5,w.y=z*.5*P+.5,f.push(w.x,w.y),g++}for(let L=0;L<s;L++){let X=T+L,W=I+L;v===!0?h.push(W,W+1,X):h.push(W+1,W,X),_+=3}l.addGroup(m,_,v===!0?1:2),m+=_}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}},yr=class n extends tn{constructor(e=1,t=1,i=32,s=1,r=!1,a=0,o=Math.PI*2){super(0,e,t,i,s,r,a,o),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:i,heightSegments:s,openEnded:r,thetaStart:a,thetaLength:o}}static fromJSON(e){return new n(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}},ma=class n extends mt{constructor(e=[],t=[],i=1,s=0){super(),this.type="PolyhedronGeometry",this.parameters={vertices:e,indices:t,radius:i,detail:s};let r=[],a=[];o(s),l(i),h(),this.setAttribute("position",new nt(r,3)),this.setAttribute("normal",new nt(r.slice(),3)),this.setAttribute("uv",new nt(a,2)),s===0?this.computeVertexNormals():this.normalizeNormals();function o(M){let b=new A,v=new A,T=new A;for(let w=0;w<t.length;w+=3)f(t[w+0],b),f(t[w+1],v),f(t[w+2],T),c(b,v,T,M)}function c(M,b,v,T){let w=T+1,C=[];for(let _=0;_<=w;_++){C[_]=[];let E=M.clone().lerp(v,_/w),P=b.clone().lerp(v,_/w),I=w-_;for(let L=0;L<=I;L++)L===0&&_===w?C[_][L]=E:C[_][L]=E.clone().lerp(P,L/I)}for(let _=0;_<w;_++)for(let E=0;E<2*(w-_)-1;E++){let P=Math.floor(E/2);E%2===0?(u(C[_][P+1]),u(C[_+1][P]),u(C[_][P])):(u(C[_][P+1]),u(C[_+1][P+1]),u(C[_+1][P]))}}function l(M){let b=new A;for(let v=0;v<r.length;v+=3)b.x=r[v+0],b.y=r[v+1],b.z=r[v+2],b.normalize().multiplyScalar(M),r[v+0]=b.x,r[v+1]=b.y,r[v+2]=b.z}function h(){let M=new A;for(let b=0;b<r.length;b+=3){M.x=r[b+0],M.y=r[b+1],M.z=r[b+2];let v=p(M)/2/Math.PI+.5,T=m(M)/Math.PI+.5;a.push(v,1-T)}g(),d()}function d(){for(let M=0;M<a.length;M+=6){let b=a[M+0],v=a[M+2],T=a[M+4],w=Math.max(b,v,T),C=Math.min(b,v,T);w>.9&&C<.1&&(b<.2&&(a[M+0]+=1),v<.2&&(a[M+2]+=1),T<.2&&(a[M+4]+=1))}}function u(M){r.push(M.x,M.y,M.z)}function f(M,b){let v=M*3;b.x=e[v+0],b.y=e[v+1],b.z=e[v+2]}function g(){let M=new A,b=new A,v=new A,T=new A,w=new te,C=new te,_=new te;for(let E=0,P=0;E<r.length;E+=9,P+=6){M.set(r[E+0],r[E+1],r[E+2]),b.set(r[E+3],r[E+4],r[E+5]),v.set(r[E+6],r[E+7],r[E+8]),w.set(a[P+0],a[P+1]),C.set(a[P+2],a[P+3]),_.set(a[P+4],a[P+5]),T.copy(M).add(b).add(v).divideScalar(3);let I=p(T);x(w,P+0,M,I),x(C,P+2,b,I),x(_,P+4,v,I)}}function x(M,b,v,T){T<0&&M.x===1&&(a[b]=M.x-1),v.x===0&&v.z===0&&(a[b]=T/2/Math.PI+.5)}function p(M){return Math.atan2(M.z,-M.x)}function m(M){return Math.atan2(-M.y,Math.sqrt(M.x*M.x+M.z*M.z))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.vertices,e.indices,e.radius,e.detail)}},ga=class n extends ma{constructor(e=1,t=0){let i=(1+Math.sqrt(5))/2,s=1/i,r=[-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,0,-s,-i,0,-s,i,0,s,-i,0,s,i,-s,-i,0,-s,i,0,s,-i,0,s,i,0,-i,0,-s,i,0,-s,-i,0,s,i,0,s],a=[3,11,7,3,7,15,3,15,13,7,19,17,7,17,6,7,6,15,17,4,8,17,8,10,17,10,6,8,0,16,8,16,2,8,2,10,0,12,1,0,1,18,0,18,16,6,10,2,6,2,13,6,13,15,2,16,18,2,18,3,2,3,13,18,1,9,18,9,11,18,11,3,4,14,12,4,12,0,4,0,8,11,9,5,11,5,19,11,19,7,19,5,14,19,14,4,19,4,17,1,12,14,1,14,5,1,5,9];super(r,a,e,t),this.type="DodecahedronGeometry",this.parameters={radius:e,detail:t}}static fromJSON(e){return new n(e.radius,e.detail)}},zo=new A,ko=new A,Lh=new A,Ho=new pn,_a=class extends mt{constructor(e=null,t=1){if(super(),this.type="EdgesGeometry",this.parameters={geometry:e,thresholdAngle:t},e!==null){let s=Math.pow(10,4),r=Math.cos(hr*t),a=e.getIndex(),o=e.getAttribute("position"),c=a?a.count:o.count,l=[0,0,0],h=["a","b","c"],d=new Array(3),u={},f=[];for(let g=0;g<c;g+=3){a?(l[0]=a.getX(g),l[1]=a.getX(g+1),l[2]=a.getX(g+2)):(l[0]=g,l[1]=g+1,l[2]=g+2);let{a:x,b:p,c:m}=Ho;if(x.fromBufferAttribute(o,l[0]),p.fromBufferAttribute(o,l[1]),m.fromBufferAttribute(o,l[2]),Ho.getNormal(Lh),d[0]=`${Math.round(x.x*s)},${Math.round(x.y*s)},${Math.round(x.z*s)}`,d[1]=`${Math.round(p.x*s)},${Math.round(p.y*s)},${Math.round(p.z*s)}`,d[2]=`${Math.round(m.x*s)},${Math.round(m.y*s)},${Math.round(m.z*s)}`,!(d[0]===d[1]||d[1]===d[2]||d[2]===d[0]))for(let M=0;M<3;M++){let b=(M+1)%3,v=d[M],T=d[b],w=Ho[h[M]],C=Ho[h[b]],_=`${v}_${T}`,E=`${T}_${v}`;E in u&&u[E]?(Lh.dot(u[E].normal)<=r&&(f.push(w.x,w.y,w.z),f.push(C.x,C.y,C.z)),u[E]=null):_ in u||(u[_]={index0:l[M],index1:l[b],normal:Lh.clone()})}}for(let g in u)if(u[g]){let{index0:x,index1:p}=u[g];zo.fromBufferAttribute(o,x),ko.fromBufferAttribute(o,p),f.push(zo.x,zo.y,zo.z),f.push(ko.x,ko.y,ko.z)}this.setAttribute("position",new nt(f,3))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}},Ii=class{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){Ye("Curve: .getPoint() not implemented.")}getPointAt(e,t){let i=this.getUtoTmapping(e);return this.getPoint(i,t)}getPoints(e=5){let t=[];for(let i=0;i<=e;i++)t.push(this.getPoint(i/e));return t}getSpacedPoints(e=5){let t=[];for(let i=0;i<=e;i++)t.push(this.getPointAt(i/e));return t}getLength(){let e=this.getLengths();return e[e.length-1]}getLengths(e=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===e+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;let t=[],i,s=this.getPoint(0),r=0;t.push(0);for(let a=1;a<=e;a++)i=this.getPoint(a/e),r+=i.distanceTo(s),t.push(r),s=i;return this.cacheArcLengths=t,t}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(e,t=null){let i=this.getLengths(),s=0,r=i.length,a;t?a=t:a=e*i[r-1];let o=0,c=r-1,l;for(;o<=c;)if(s=Math.floor(o+(c-o)/2),l=i[s]-a,l<0)o=s+1;else if(l>0)c=s-1;else{c=s;break}if(s=c,i[s]===a)return s/(r-1);let h=i[s],u=i[s+1]-h,f=(a-h)/u;return(s+f)/(r-1)}getTangent(e,t){let s=e-1e-4,r=e+1e-4;s<0&&(s=0),r>1&&(r=1);let a=this.getPoint(s),o=this.getPoint(r),c=t||(a.isVector2?new te:new A);return c.copy(o).sub(a).normalize(),c}getTangentAt(e,t){let i=this.getUtoTmapping(e);return this.getTangent(i,t)}computeFrenetFrames(e,t=!1){let i=new A,s=[],r=[],a=[],o=new A,c=new rt;for(let f=0;f<=e;f++){let g=f/e;s[f]=this.getTangentAt(g,new A)}r[0]=new A,a[0]=new A;let l=Number.MAX_VALUE,h=Math.abs(s[0].x),d=Math.abs(s[0].y),u=Math.abs(s[0].z);h<=l&&(l=h,i.set(1,0,0)),d<=l&&(l=d,i.set(0,1,0)),u<=l&&i.set(0,0,1),o.crossVectors(s[0],i).normalize(),r[0].crossVectors(s[0],o),a[0].crossVectors(s[0],r[0]);for(let f=1;f<=e;f++){if(r[f]=r[f-1].clone(),a[f]=a[f-1].clone(),o.crossVectors(s[f-1],s[f]),o.length()>Number.EPSILON){o.normalize();let g=Math.acos(Ke(s[f-1].dot(s[f]),-1,1));r[f].applyMatrix4(c.makeRotationAxis(o,g))}a[f].crossVectors(s[f],r[f])}if(t===!0){let f=Math.acos(Ke(r[0].dot(r[e]),-1,1));f/=e,s[0].dot(o.crossVectors(r[0],r[e]))>0&&(f=-f);for(let g=1;g<=e;g++)r[g].applyMatrix4(c.makeRotationAxis(s[g],f*g)),a[g].crossVectors(s[g],r[g])}return{tangents:s,normals:r,binormals:a}}clone(){return new this.constructor().copy(this)}copy(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}toJSON(){let e={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return e.arcLengthDivisions=this.arcLengthDivisions,e.type=this.type,e}fromJSON(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}},Mr=class extends Ii{constructor(e=0,t=0,i=1,s=1,r=0,a=Math.PI*2,o=!1,c=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=e,this.aY=t,this.xRadius=i,this.yRadius=s,this.aStartAngle=r,this.aEndAngle=a,this.aClockwise=o,this.aRotation=c}getPoint(e,t=new te){let i=t,s=Math.PI*2,r=this.aEndAngle-this.aStartAngle,a=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=s;for(;r>s;)r-=s;r<Number.EPSILON&&(a?r=0:r=s),this.aClockwise===!0&&!a&&(r===s?r=-s:r=r-s);let o=this.aStartAngle+e*r,c=this.aX+this.xRadius*Math.cos(o),l=this.aY+this.yRadius*Math.sin(o);if(this.aRotation!==0){let h=Math.cos(this.aRotation),d=Math.sin(this.aRotation),u=c-this.aX,f=l-this.aY;c=u*h-f*d+this.aX,l=u*d+f*h+this.aY}return i.set(c,l)}copy(e){return super.copy(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}toJSON(){let e=super.toJSON();return e.aX=this.aX,e.aY=this.aY,e.xRadius=this.xRadius,e.yRadius=this.yRadius,e.aStartAngle=this.aStartAngle,e.aEndAngle=this.aEndAngle,e.aClockwise=this.aClockwise,e.aRotation=this.aRotation,e}fromJSON(e){return super.fromJSON(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}},pl=class extends Mr{constructor(e,t,i,s,r,a){super(e,t,i,i,s,r,a),this.isArcCurve=!0,this.type="ArcCurve"}};function du(){let n=0,e=0,t=0,i=0;function s(r,a,o,c){n=r,e=o,t=-3*r+3*a-2*o-c,i=2*r-2*a+o+c}return{initCatmullRom:function(r,a,o,c,l){s(a,o,l*(o-r),l*(c-a))},initNonuniformCatmullRom:function(r,a,o,c,l,h,d){let u=(a-r)/l-(o-r)/(l+h)+(o-a)/h,f=(o-a)/h-(c-a)/(h+d)+(c-o)/d;u*=h,f*=h,s(a,o,u,f)},calc:function(r){let a=r*r,o=a*r;return n+e*r+t*a+i*o}}}var Xd=new A,qd=new A,Nh=new du,Uh=new du,Fh=new du,ml=class extends Ii{constructor(e=[],t=!1,i="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=e,this.closed=t,this.curveType=i,this.tension=s}getPoint(e,t=new A){let i=t,s=this.points,r=s.length,a=(r-(this.closed?0:1))*e,o=Math.floor(a),c=a-o;this.closed?o+=o>0?0:(Math.floor(Math.abs(o)/r)+1)*r:c===0&&o===r-1&&(o=r-2,c=1);let l,h;this.closed||o>0?l=s[(o-1)%r]:(qd.subVectors(s[0],s[1]).add(s[0]),l=qd);let d=s[o%r],u=s[(o+1)%r];if(this.closed||o+2<r?h=s[(o+2)%r]:(Xd.subVectors(s[r-1],s[r-2]).add(s[r-1]),h=Xd),this.curveType==="centripetal"||this.curveType==="chordal"){let f=this.curveType==="chordal"?.5:.25,g=Math.pow(l.distanceToSquared(d),f),x=Math.pow(d.distanceToSquared(u),f),p=Math.pow(u.distanceToSquared(h),f);x<1e-4&&(x=1),g<1e-4&&(g=x),p<1e-4&&(p=x),Nh.initNonuniformCatmullRom(l.x,d.x,u.x,h.x,g,x,p),Uh.initNonuniformCatmullRom(l.y,d.y,u.y,h.y,g,x,p),Fh.initNonuniformCatmullRom(l.z,d.z,u.z,h.z,g,x,p)}else this.curveType==="catmullrom"&&(Nh.initCatmullRom(l.x,d.x,u.x,h.x,this.tension),Uh.initCatmullRom(l.y,d.y,u.y,h.y,this.tension),Fh.initCatmullRom(l.z,d.z,u.z,h.z,this.tension));return i.set(Nh.calc(c),Uh.calc(c),Fh.calc(c)),i}copy(e){super.copy(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(s.clone())}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}toJSON(){let e=super.toJSON();e.points=[];for(let t=0,i=this.points.length;t<i;t++){let s=this.points[t];e.points.push(s.toArray())}return e.closed=this.closed,e.curveType=this.curveType,e.tension=this.tension,e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(new A().fromArray(s))}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}};function Yd(n,e,t,i,s){let r=(i-e)*.5,a=(s-t)*.5,o=n*n,c=n*o;return(2*t-2*i+r+a)*c+(-3*t+3*i-2*r-a)*o+r*n+t}function hg(n,e){let t=1-n;return t*t*e}function ug(n,e){return 2*(1-n)*n*e}function dg(n,e){return n*n*e}function Qr(n,e,t,i){return hg(n,e)+ug(n,t)+dg(n,i)}function fg(n,e){let t=1-n;return t*t*t*e}function pg(n,e){let t=1-n;return 3*t*t*n*e}function mg(n,e){return 3*(1-n)*n*n*e}function gg(n,e){return n*n*n*e}function ea(n,e,t,i,s){return fg(n,e)+pg(n,t)+mg(n,i)+gg(n,s)}var xa=class extends Ii{constructor(e=new te,t=new te,i=new te,s=new te){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=e,this.v1=t,this.v2=i,this.v3=s}getPoint(e,t=new te){let i=t,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return i.set(ea(e,s.x,r.x,a.x,o.x),ea(e,s.y,r.y,a.y,o.y)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}},gl=class extends Ii{constructor(e=new A,t=new A,i=new A,s=new A){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=e,this.v1=t,this.v2=i,this.v3=s}getPoint(e,t=new A){let i=t,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return i.set(ea(e,s.x,r.x,a.x,o.x),ea(e,s.y,r.y,a.y,o.y),ea(e,s.z,r.z,a.z,o.z)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}},va=class extends Ii{constructor(e=new te,t=new te){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=e,this.v2=t}getPoint(e,t=new te){let i=t;return e===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(e).add(this.v1)),i}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new te){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},_l=class extends Ii{constructor(e=new A,t=new A){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=e,this.v2=t}getPoint(e,t=new A){let i=t;return e===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(e).add(this.v1)),i}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new A){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},ya=class extends Ii{constructor(e=new te,t=new te,i=new te){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=e,this.v1=t,this.v2=i}getPoint(e,t=new te){let i=t,s=this.v0,r=this.v1,a=this.v2;return i.set(Qr(e,s.x,r.x,a.x),Qr(e,s.y,r.y,a.y)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},xl=class extends Ii{constructor(e=new A,t=new A,i=new A){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=e,this.v1=t,this.v2=i}getPoint(e,t=new A){let i=t,s=this.v0,r=this.v1,a=this.v2;return i.set(Qr(e,s.x,r.x,a.x),Qr(e,s.y,r.y,a.y),Qr(e,s.z,r.z,a.z)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){let e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}},Ma=class extends Ii{constructor(e=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=e}getPoint(e,t=new te){let i=t,s=this.points,r=(s.length-1)*e,a=Math.floor(r),o=r-a,c=s[a===0?a:a-1],l=s[a],h=s[a>s.length-2?s.length-1:a+1],d=s[a>s.length-3?s.length-1:a+2];return i.set(Yd(o,c.x,l.x,h.x,d.x),Yd(o,c.y,l.y,h.y,d.y)),i}copy(e){super.copy(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(s.clone())}return this}toJSON(){let e=super.toJSON();e.points=[];for(let t=0,i=this.points.length;t<i;t++){let s=this.points[t];e.points.push(s.toArray())}return e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){let s=e.points[t];this.points.push(new te().fromArray(s))}return this}},Wh=Object.freeze({__proto__:null,ArcCurve:pl,CatmullRomCurve3:ml,CubicBezierCurve:xa,CubicBezierCurve3:gl,EllipseCurve:Mr,LineCurve:va,LineCurve3:_l,QuadraticBezierCurve:ya,QuadraticBezierCurve3:xl,SplineCurve:Ma}),vl=class extends Ii{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(e){this.curves.push(e)}closePath(){let e=this.curves[0].getPoint(0),t=this.curves[this.curves.length-1].getPoint(1);if(!e.equals(t)){let i=e.isVector2===!0?"LineCurve":"LineCurve3";this.curves.push(new Wh[i](t,e))}return this}getPoint(e,t){let i=e*this.getLength(),s=this.getCurveLengths(),r=0;for(;r<s.length;){if(s[r]>=i){let a=s[r]-i,o=this.curves[r],c=o.getLength(),l=c===0?0:1-a/c;return o.getPointAt(l,t)}r++}return null}getLength(){let e=this.getCurveLengths();return e[e.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;let e=[],t=0;for(let i=0,s=this.curves.length;i<s;i++)t+=this.curves[i].getLength(),e.push(t);return this.cacheLengths=e,e}getSpacedPoints(e=40){let t=[];for(let i=0;i<=e;i++)t.push(this.getPoint(i/e));return this.autoClose&&t.push(t[0]),t}getPoints(e=12){let t=[],i;for(let s=0,r=this.curves;s<r.length;s++){let a=r[s],o=a.isEllipseCurve?e*2:a.isLineCurve||a.isLineCurve3?1:a.isSplineCurve?e*a.points.length:e,c=a.getPoints(o);for(let l=0;l<c.length;l++){let h=c[l];i&&i.equals(h)||(t.push(h),i=h)}}return this.autoClose&&t.length>1&&!t[t.length-1].equals(t[0])&&t.push(t[0]),t}copy(e){super.copy(e),this.curves=[];for(let t=0,i=e.curves.length;t<i;t++){let s=e.curves[t];this.curves.push(s.clone())}return this.autoClose=e.autoClose,this}toJSON(){let e=super.toJSON();e.autoClose=this.autoClose,e.curves=[];for(let t=0,i=this.curves.length;t<i;t++){let s=this.curves[t];e.curves.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.autoClose=e.autoClose,this.curves=[];for(let t=0,i=e.curves.length;t<i;t++){let s=e.curves[t];this.curves.push(new Wh[s.type]().fromJSON(s))}return this}},Ps=class extends vl{constructor(e){super(),this.type="Path",this.currentPoint=new te,e&&this.setFromPoints(e)}setFromPoints(e){this.moveTo(e[0].x,e[0].y);for(let t=1,i=e.length;t<i;t++)this.lineTo(e[t].x,e[t].y);return this}moveTo(e,t){return this.currentPoint.set(e,t),this}lineTo(e,t){let i=new va(this.currentPoint.clone(),new te(e,t));return this.curves.push(i),this.currentPoint.set(e,t),this}quadraticCurveTo(e,t,i,s){let r=new ya(this.currentPoint.clone(),new te(e,t),new te(i,s));return this.curves.push(r),this.currentPoint.set(i,s),this}bezierCurveTo(e,t,i,s,r,a){let o=new xa(this.currentPoint.clone(),new te(e,t),new te(i,s),new te(r,a));return this.curves.push(o),this.currentPoint.set(r,a),this}splineThru(e){let t=[this.currentPoint.clone()].concat(e),i=new Ma(t);return this.curves.push(i),this.currentPoint.copy(e[e.length-1]),this}arc(e,t,i,s,r,a){let o=this.currentPoint.x,c=this.currentPoint.y;return this.absarc(e+o,t+c,i,s,r,a),this}absarc(e,t,i,s,r,a){return this.absellipse(e,t,i,i,s,r,a),this}ellipse(e,t,i,s,r,a,o,c){let l=this.currentPoint.x,h=this.currentPoint.y;return this.absellipse(e+l,t+h,i,s,r,a,o,c),this}absellipse(e,t,i,s,r,a,o,c){let l=new Mr(e,t,i,s,r,a,o,c);if(this.curves.length>0){let d=l.getPoint(0);d.equals(this.currentPoint)||this.lineTo(d.x,d.y)}this.curves.push(l);let h=l.getPoint(1);return this.currentPoint.copy(h),this}copy(e){return super.copy(e),this.currentPoint.copy(e.currentPoint),this}toJSON(){let e=super.toJSON();return e.currentPoint=this.currentPoint.toArray(),e}fromJSON(e){return super.fromJSON(e),this.currentPoint.fromArray(e.currentPoint),this}},Di=class extends Ps{constructor(e){super(e),this.uuid=gn(),this.type="Shape",this.holes=[]}getPointsHoles(e){let t=[];for(let i=0,s=this.holes.length;i<s;i++)t[i]=this.holes[i].getPoints(e);return t}extractPoints(e){return{shape:this.getPoints(e),holes:this.getPointsHoles(e)}}copy(e){super.copy(e),this.holes=[];for(let t=0,i=e.holes.length;t<i;t++){let s=e.holes[t];this.holes.push(s.clone())}return this}toJSON(){let e=super.toJSON();e.uuid=this.uuid,e.holes=[];for(let t=0,i=this.holes.length;t<i;t++){let s=this.holes[t];e.holes.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.uuid=e.uuid,this.holes=[];for(let t=0,i=e.holes.length;t<i;t++){let s=e.holes[t];this.holes.push(new Ps().fromJSON(s))}return this}};function _g(n,e,t=2){let i=e&&e.length,s=i?e[0]*t:n.length,r=Hf(n,0,s,t,!0),a=[];if(!r||r.next===r.prev)return a;let o,c,l;if(i&&(r=bg(n,e,r,t)),n.length>80*t){o=n[0],c=n[1];let h=o,d=c;for(let u=t;u<s;u+=t){let f=n[u],g=n[u+1];f<o&&(o=f),g<c&&(c=g),f>h&&(h=f),g>d&&(d=g)}l=Math.max(h-o,d-c),l=l!==0?32767/l:0}return ba(r,a,t,o,c,l,0),a}function Hf(n,e,t,i,s){let r;if(s===Lg(n,e,t,i)>0)for(let a=e;a<t;a+=i)r=$d(a/i|0,n[a],n[a+1],r);else for(let a=t-i;a>=e;a-=i)r=$d(a/i|0,n[a],n[a+1],r);return r&&br(r,r.next)&&(Ea(r),r=r.next),r}function Is(n,e){if(!n)return n;e||(e=n);let t=n,i;do if(i=!1,!t.steiner&&(br(t,t.next)||Pt(t.prev,t,t.next)===0)){if(Ea(t),t=e=t.prev,t===t.next)break;i=!0}else t=t.next;while(i||t!==e);return e}function ba(n,e,t,i,s,r,a){if(!n)return;!a&&r&&Ag(n,i,s,r);let o=n;for(;n.prev!==n.next;){let c=n.prev,l=n.next;if(r?vg(n,i,s,r):xg(n)){e.push(c.i,n.i,l.i),Ea(n),n=l.next,o=l.next;continue}if(n=l,n===o){a?a===1?(n=yg(Is(n),e),ba(n,e,t,i,s,r,2)):a===2&&Mg(n,e,t,i,s,r):ba(Is(n),e,t,i,s,r,1);break}}}function xg(n){let e=n.prev,t=n,i=n.next;if(Pt(e,t,i)>=0)return!1;let s=e.x,r=t.x,a=i.x,o=e.y,c=t.y,l=i.y,h=Math.min(s,r,a),d=Math.min(o,c,l),u=Math.max(s,r,a),f=Math.max(o,c,l),g=i.next;for(;g!==e;){if(g.x>=h&&g.x<=u&&g.y>=d&&g.y<=f&&Kr(s,o,r,c,a,l,g.x,g.y)&&Pt(g.prev,g,g.next)>=0)return!1;g=g.next}return!0}function vg(n,e,t,i){let s=n.prev,r=n,a=n.next;if(Pt(s,r,a)>=0)return!1;let o=s.x,c=r.x,l=a.x,h=s.y,d=r.y,u=a.y,f=Math.min(o,c,l),g=Math.min(h,d,u),x=Math.max(o,c,l),p=Math.max(h,d,u),m=Xh(f,g,e,t,i),M=Xh(x,p,e,t,i),b=n.prevZ,v=n.nextZ;for(;b&&b.z>=m&&v&&v.z<=M;){if(b.x>=f&&b.x<=x&&b.y>=g&&b.y<=p&&b!==s&&b!==a&&Kr(o,h,c,d,l,u,b.x,b.y)&&Pt(b.prev,b,b.next)>=0||(b=b.prevZ,v.x>=f&&v.x<=x&&v.y>=g&&v.y<=p&&v!==s&&v!==a&&Kr(o,h,c,d,l,u,v.x,v.y)&&Pt(v.prev,v,v.next)>=0))return!1;v=v.nextZ}for(;b&&b.z>=m;){if(b.x>=f&&b.x<=x&&b.y>=g&&b.y<=p&&b!==s&&b!==a&&Kr(o,h,c,d,l,u,b.x,b.y)&&Pt(b.prev,b,b.next)>=0)return!1;b=b.prevZ}for(;v&&v.z<=M;){if(v.x>=f&&v.x<=x&&v.y>=g&&v.y<=p&&v!==s&&v!==a&&Kr(o,h,c,d,l,u,v.x,v.y)&&Pt(v.prev,v,v.next)>=0)return!1;v=v.nextZ}return!0}function yg(n,e){let t=n;do{let i=t.prev,s=t.next.next;!br(i,s)&&Gf(i,t,t.next,s)&&Sa(i,s)&&Sa(s,i)&&(e.push(i.i,t.i,s.i),Ea(t),Ea(t.next),t=n=s),t=t.next}while(t!==n);return Is(t)}function Mg(n,e,t,i,s,r){let a=n;do{let o=a.next.next;for(;o!==a.prev;){if(a.i!==o.i&&Pg(a,o)){let c=Wf(a,o);a=Is(a,a.next),c=Is(c,c.next),ba(a,e,t,i,s,r,0),ba(c,e,t,i,s,r,0);return}o=o.next}a=a.next}while(a!==n)}function bg(n,e,t,i){let s=[];for(let r=0,a=e.length;r<a;r++){let o=e[r]*i,c=r<a-1?e[r+1]*i:n.length,l=Hf(n,o,c,i,!1);l===l.next&&(l.steiner=!0),s.push(Cg(l))}s.sort(Sg);for(let r=0;r<s.length;r++)t=Eg(s[r],t);return t}function Sg(n,e){let t=n.x-e.x;if(t===0&&(t=n.y-e.y,t===0)){let i=(n.next.y-n.y)/(n.next.x-n.x),s=(e.next.y-e.y)/(e.next.x-e.x);t=i-s}return t}function Eg(n,e){let t=wg(n,e);if(!t)return e;let i=Wf(t,n);return Is(i,i.next),Is(t,t.next)}function wg(n,e){let t=e,i=n.x,s=n.y,r=-1/0,a;if(br(n,t))return t;do{if(br(n,t.next))return t.next;if(s<=t.y&&s>=t.next.y&&t.next.y!==t.y){let d=t.x+(s-t.y)*(t.next.x-t.x)/(t.next.y-t.y);if(d<=i&&d>r&&(r=d,a=t.x<t.next.x?t:t.next,d===i))return a}t=t.next}while(t!==e);if(!a)return null;let o=a,c=a.x,l=a.y,h=1/0;t=a;do{if(i>=t.x&&t.x>=c&&i!==t.x&&Vf(s<l?i:r,s,c,l,s<l?r:i,s,t.x,t.y)){let d=Math.abs(s-t.y)/(i-t.x);Sa(t,n)&&(d<h||d===h&&(t.x>a.x||t.x===a.x&&Tg(a,t)))&&(a=t,h=d)}t=t.next}while(t!==o);return a}function Tg(n,e){return Pt(n.prev,n,e.prev)<0&&Pt(e.next,n,n.next)<0}function Ag(n,e,t,i){let s=n;do s.z===0&&(s.z=Xh(s.x,s.y,e,t,i)),s.prevZ=s.prev,s.nextZ=s.next,s=s.next;while(s!==n);s.prevZ.nextZ=null,s.prevZ=null,Rg(s)}function Rg(n){let e,t=1;do{let i=n,s;n=null;let r=null;for(e=0;i;){e++;let a=i,o=0;for(let l=0;l<t&&(o++,a=a.nextZ,!!a);l++);let c=t;for(;o>0||c>0&&a;)o!==0&&(c===0||!a||i.z<=a.z)?(s=i,i=i.nextZ,o--):(s=a,a=a.nextZ,c--),r?r.nextZ=s:n=s,s.prevZ=r,r=s;i=a}r.nextZ=null,t*=2}while(e>1);return n}function Xh(n,e,t,i,s){return n=(n-t)*s|0,e=(e-i)*s|0,n=(n|n<<8)&16711935,n=(n|n<<4)&252645135,n=(n|n<<2)&858993459,n=(n|n<<1)&1431655765,e=(e|e<<8)&16711935,e=(e|e<<4)&252645135,e=(e|e<<2)&858993459,e=(e|e<<1)&1431655765,n|e<<1}function Cg(n){let e=n,t=n;do(e.x<t.x||e.x===t.x&&e.y<t.y)&&(t=e),e=e.next;while(e!==n);return t}function Vf(n,e,t,i,s,r,a,o){return(s-a)*(e-o)>=(n-a)*(r-o)&&(n-a)*(i-o)>=(t-a)*(e-o)&&(t-a)*(r-o)>=(s-a)*(i-o)}function Kr(n,e,t,i,s,r,a,o){return!(n===a&&e===o)&&Vf(n,e,t,i,s,r,a,o)}function Pg(n,e){return n.next.i!==e.i&&n.prev.i!==e.i&&!Ig(n,e)&&(Sa(n,e)&&Sa(e,n)&&Dg(n,e)&&(Pt(n.prev,n,e.prev)||Pt(n,e.prev,e))||br(n,e)&&Pt(n.prev,n,n.next)>0&&Pt(e.prev,e,e.next)>0)}function Pt(n,e,t){return(e.y-n.y)*(t.x-e.x)-(e.x-n.x)*(t.y-e.y)}function br(n,e){return n.x===e.x&&n.y===e.y}function Gf(n,e,t,i){let s=Go(Pt(n,e,t)),r=Go(Pt(n,e,i)),a=Go(Pt(t,i,n)),o=Go(Pt(t,i,e));return!!(s!==r&&a!==o||s===0&&Vo(n,t,e)||r===0&&Vo(n,i,e)||a===0&&Vo(t,n,i)||o===0&&Vo(t,e,i))}function Vo(n,e,t){return e.x<=Math.max(n.x,t.x)&&e.x>=Math.min(n.x,t.x)&&e.y<=Math.max(n.y,t.y)&&e.y>=Math.min(n.y,t.y)}function Go(n){return n>0?1:n<0?-1:0}function Ig(n,e){let t=n;do{if(t.i!==n.i&&t.next.i!==n.i&&t.i!==e.i&&t.next.i!==e.i&&Gf(t,t.next,n,e))return!0;t=t.next}while(t!==n);return!1}function Sa(n,e){return Pt(n.prev,n,n.next)<0?Pt(n,e,n.next)>=0&&Pt(n,n.prev,e)>=0:Pt(n,e,n.prev)<0||Pt(n,n.next,e)<0}function Dg(n,e){let t=n,i=!1,s=(n.x+e.x)/2,r=(n.y+e.y)/2;do t.y>r!=t.next.y>r&&t.next.y!==t.y&&s<(t.next.x-t.x)*(r-t.y)/(t.next.y-t.y)+t.x&&(i=!i),t=t.next;while(t!==n);return i}function Wf(n,e){let t=qh(n.i,n.x,n.y),i=qh(e.i,e.x,e.y),s=n.next,r=e.prev;return n.next=e,e.prev=n,t.next=s,s.prev=t,i.next=t,t.prev=i,r.next=i,i.prev=r,i}function $d(n,e,t,i){let s=qh(n,e,t);return i?(s.next=i.next,s.prev=i,i.next.prev=s,i.next=s):(s.prev=s,s.next=s),s}function Ea(n){n.next.prev=n.prev,n.prev.next=n.next,n.prevZ&&(n.prevZ.nextZ=n.nextZ),n.nextZ&&(n.nextZ.prevZ=n.prevZ)}function qh(n,e,t){return{i:n,x:e,y:t,prev:null,next:null,z:0,prevZ:null,nextZ:null,steiner:!1}}function Lg(n,e,t,i){let s=0;for(let r=e,a=t-i;r<t;r+=i)s+=(n[a]-n[r])*(n[r+1]+n[a+1]),a=r;return s}var Yh=class{static triangulate(e,t,i=2){return _g(e,t,i)}},ws=class n{static area(e){let t=e.length,i=0;for(let s=t-1,r=0;r<t;s=r++)i+=e[s].x*e[r].y-e[r].x*e[s].y;return i*.5}static isClockWise(e){return n.area(e)<0}static triangulateShape(e,t){let i=[],s=[],r=[];Zd(e),Jd(i,e);let a=e.length;t.forEach(Zd);for(let c=0;c<t.length;c++)s.push(a),a+=t[c].length,Jd(i,t[c]);let o=Yh.triangulate(i,s);for(let c=0;c<o.length;c+=3)r.push(o.slice(c,c+3));return r}};function Zd(n){let e=n.length;e>2&&n[e-1].equals(n[0])&&n.pop()}function Jd(n,e){for(let t=0;t<e.length;t++)n.push(e[t].x),n.push(e[t].y)}var Hi=class n extends mt{constructor(e=new Di([new te(.5,.5),new te(-.5,.5),new te(-.5,-.5),new te(.5,-.5)]),t={}){super(),this.type="ExtrudeGeometry",this.parameters={shapes:e,options:t},e=Array.isArray(e)?e:[e];let i=this,s=[],r=[];for(let o=0,c=e.length;o<c;o++){let l=e[o];a(l)}this.setAttribute("position",new nt(s,3)),this.setAttribute("uv",new nt(r,2)),this.computeVertexNormals();function a(o){let c=[],l=t.curveSegments!==void 0?t.curveSegments:12,h=t.steps!==void 0?t.steps:1,d=t.depth!==void 0?t.depth:1,u=t.bevelEnabled!==void 0?t.bevelEnabled:!0,f=t.bevelThickness!==void 0?t.bevelThickness:.2,g=t.bevelSize!==void 0?t.bevelSize:f-.1,x=t.bevelOffset!==void 0?t.bevelOffset:0,p=t.bevelSegments!==void 0?t.bevelSegments:3,m=t.extrudePath,M=t.UVGenerator!==void 0?t.UVGenerator:Ng,b,v=!1,T,w,C,_;if(m){b=m.getSpacedPoints(h),v=!0,u=!1;let oe=m.isCatmullRomCurve3?m.closed:!1;T=m.computeFrenetFrames(h,oe),w=new A,C=new A,_=new A}u||(p=0,f=0,g=0,x=0);let E=o.extractPoints(l),P=E.shape,I=E.holes;if(!ws.isClockWise(P)){P=P.reverse();for(let oe=0,ee=I.length;oe<ee;oe++){let le=I[oe];ws.isClockWise(le)&&(I[oe]=le.reverse())}}function X(oe){let le=10000000000000001e-36,J=oe[0];for(let se=1;se<=oe.length;se++){let fe=se%oe.length,ge=oe[fe],we=ge.x-J.x,Se=ge.y-J.y,D=we*we+Se*Se,Pe=Math.max(Math.abs(ge.x),Math.abs(ge.y),Math.abs(J.x),Math.abs(J.y)),Ze=le*Pe*Pe;if(D<=Ze){oe.splice(fe,1),se--;continue}J=ge}}X(P),I.forEach(X);let W=I.length,U=P;for(let oe=0;oe<W;oe++){let ee=I[oe];P=P.concat(ee)}function z(oe,ee,le){return ee||$e("ExtrudeGeometry: vec does not exist"),oe.clone().addScaledVector(ee,le)}let H=P.length;function Q(oe,ee,le){let J,se,fe,ge=oe.x-ee.x,we=oe.y-ee.y,Se=le.x-oe.x,D=le.y-oe.y,Pe=ge*ge+we*we,Ze=ge*D-we*Se;if(Math.abs(Ze)>Number.EPSILON){let R=Math.sqrt(Pe),y=Math.sqrt(Se*Se+D*D),F=ee.x-we/R,B=ee.y+ge/R,Y=le.x-D/y,pe=le.y+Se/y,_e=((Y-F)*D-(pe-B)*Se)/(ge*D-we*Se);J=F+ge*_e-oe.x,se=B+we*_e-oe.y;let K=J*J+se*se;if(K<=2)return new te(J,se);fe=Math.sqrt(K/2)}else{let R=!1;ge>Number.EPSILON?Se>Number.EPSILON&&(R=!0):ge<-Number.EPSILON?Se<-Number.EPSILON&&(R=!0):Math.sign(we)===Math.sign(D)&&(R=!0),R?(J=-we,se=ge,fe=Math.sqrt(Pe)):(J=ge,se=we,fe=Math.sqrt(Pe/2))}return new te(J/fe,se/fe)}let ie=[];for(let oe=0,ee=U.length,le=ee-1,J=oe+1;oe<ee;oe++,le++,J++)le===ee&&(le=0),J===ee&&(J=0),ie[oe]=Q(U[oe],U[le],U[J]);let q=[],Z,j=ie.concat();for(let oe=0,ee=W;oe<ee;oe++){let le=I[oe];Z=[];for(let J=0,se=le.length,fe=se-1,ge=J+1;J<se;J++,fe++,ge++)fe===se&&(fe=0),ge===se&&(ge=0),Z[J]=Q(le[J],le[fe],le[ge]);q.push(Z),j=j.concat(Z)}let de;if(p===0)de=ws.triangulateShape(U,I);else{let oe=[],ee=[];for(let le=0;le<p;le++){let J=le/p,se=f*Math.cos(J*Math.PI/2),fe=g*Math.sin(J*Math.PI/2)+x;for(let ge=0,we=U.length;ge<we;ge++){let Se=z(U[ge],ie[ge],fe);Te(Se.x,Se.y,-se),J===0&&oe.push(Se)}for(let ge=0,we=W;ge<we;ge++){let Se=I[ge];Z=q[ge];let D=[];for(let Pe=0,Ze=Se.length;Pe<Ze;Pe++){let R=z(Se[Pe],Z[Pe],fe);Te(R.x,R.y,-se),J===0&&D.push(R)}J===0&&ee.push(D)}}de=ws.triangulateShape(oe,ee)}let Ge=de.length,me=g+x;for(let oe=0;oe<H;oe++){let ee=u?z(P[oe],j[oe],me):P[oe];v?(C.copy(T.normals[0]).multiplyScalar(ee.x),w.copy(T.binormals[0]).multiplyScalar(ee.y),_.copy(b[0]).add(C).add(w),Te(_.x,_.y,_.z)):Te(ee.x,ee.y,0)}for(let oe=1;oe<=h;oe++)for(let ee=0;ee<H;ee++){let le=u?z(P[ee],j[ee],me):P[ee];v?(C.copy(T.normals[oe]).multiplyScalar(le.x),w.copy(T.binormals[oe]).multiplyScalar(le.y),_.copy(b[oe]).add(C).add(w),Te(_.x,_.y,_.z)):Te(le.x,le.y,d/h*oe)}for(let oe=p-1;oe>=0;oe--){let ee=oe/p,le=f*Math.cos(ee*Math.PI/2),J=g*Math.sin(ee*Math.PI/2)+x;for(let se=0,fe=U.length;se<fe;se++){let ge=z(U[se],ie[se],J);Te(ge.x,ge.y,d+le)}for(let se=0,fe=I.length;se<fe;se++){let ge=I[se];Z=q[se];for(let we=0,Se=ge.length;we<Se;we++){let D=z(ge[we],Z[we],J);v?Te(D.x,D.y+b[h-1].y,b[h-1].x+le):Te(D.x,D.y,d+le)}}}k(),ce();function k(){let oe=s.length/3;if(u){let ee=0,le=H*ee;for(let J=0;J<Ge;J++){let se=de[J];Ue(se[2]+le,se[1]+le,se[0]+le)}ee=h+p*2,le=H*ee;for(let J=0;J<Ge;J++){let se=de[J];Ue(se[0]+le,se[1]+le,se[2]+le)}}else{for(let ee=0;ee<Ge;ee++){let le=de[ee];Ue(le[2],le[1],le[0])}for(let ee=0;ee<Ge;ee++){let le=de[ee];Ue(le[0]+H*h,le[1]+H*h,le[2]+H*h)}}i.addGroup(oe,s.length/3-oe,0)}function ce(){let oe=s.length/3,ee=0;ae(U,ee),ee+=U.length;for(let le=0,J=I.length;le<J;le++){let se=I[le];ae(se,ee),ee+=se.length}i.addGroup(oe,s.length/3-oe,1)}function ae(oe,ee){let le=oe.length;for(;--le>=0;){let J=le,se=le-1;se<0&&(se=oe.length-1);for(let fe=0,ge=h+p*2;fe<ge;fe++){let we=H*fe,Se=H*(fe+1),D=ee+J+we,Pe=ee+se+we,Ze=ee+se+Se,R=ee+J+Se;Oe(D,Pe,Ze,R)}}}function Te(oe,ee,le){c.push(oe),c.push(ee),c.push(le)}function Ue(oe,ee,le){st(oe),st(ee),st(le);let J=s.length/3,se=M.generateTopUV(i,s,J-3,J-2,J-1);He(se[0]),He(se[1]),He(se[2])}function Oe(oe,ee,le,J){st(oe),st(ee),st(J),st(ee),st(le),st(J);let se=s.length/3,fe=M.generateSideWallUV(i,s,se-6,se-3,se-2,se-1);He(fe[0]),He(fe[1]),He(fe[3]),He(fe[1]),He(fe[2]),He(fe[3])}function st(oe){s.push(c[oe*3+0]),s.push(c[oe*3+1]),s.push(c[oe*3+2])}function He(oe){r.push(oe.x),r.push(oe.y)}}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}toJSON(){let e=super.toJSON(),t=this.parameters.shapes,i=this.parameters.options;return Ug(t,i,e)}static fromJSON(e,t){let i=[];for(let r=0,a=e.shapes.length;r<a;r++){let o=t[e.shapes[r]];i.push(o)}let s=e.options.extrudePath;return s!==void 0&&(e.options.extrudePath=new Wh[s.type]().fromJSON(s)),new n(i,e.options)}},Ng={generateTopUV:function(n,e,t,i,s){let r=e[t*3],a=e[t*3+1],o=e[i*3],c=e[i*3+1],l=e[s*3],h=e[s*3+1];return[new te(r,a),new te(o,c),new te(l,h)]},generateSideWallUV:function(n,e,t,i,s,r){let a=e[t*3],o=e[t*3+1],c=e[t*3+2],l=e[i*3],h=e[i*3+1],d=e[i*3+2],u=e[s*3],f=e[s*3+1],g=e[s*3+2],x=e[r*3],p=e[r*3+1],m=e[r*3+2];return Math.abs(o-h)<Math.abs(a-l)?[new te(a,1-c),new te(l,1-d),new te(u,1-g),new te(x,1-m)]:[new te(o,1-c),new te(h,1-d),new te(f,1-g),new te(p,1-m)]}};function Ug(n,e,t){if(t.shapes=[],Array.isArray(n))for(let i=0,s=n.length;i<s;i++){let r=n[i];t.shapes.push(r.uuid)}else t.shapes.push(n.uuid);return t.options=Object.assign({},e),e.extrudePath!==void 0&&(t.options.extrudePath=e.extrudePath.toJSON()),t}var nn=class n extends ma{constructor(e=1,t=0){let i=(1+Math.sqrt(5))/2,s=[-1,i,0,1,i,0,-1,-i,0,1,-i,0,0,-1,i,0,1,i,0,-1,-i,0,1,-i,i,0,-1,i,0,1,-i,0,-1,-i,0,1],r=[0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1];super(s,r,e,t),this.type="IcosahedronGeometry",this.parameters={radius:e,detail:t}}static fromJSON(e){return new n(e.radius,e.detail)}};var sn=class n extends mt{constructor(e=1,t=1,i=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:i,heightSegments:s};let r=e/2,a=t/2,o=Math.floor(i),c=Math.floor(s),l=o+1,h=c+1,d=e/o,u=t/c,f=[],g=[],x=[],p=[];for(let m=0;m<h;m++){let M=m*u-a;for(let b=0;b<l;b++){let v=b*d-r;g.push(v,-M,0),x.push(0,0,1),p.push(b/o),p.push(1-m/c)}}for(let m=0;m<c;m++)for(let M=0;M<o;M++){let b=M+l*m,v=M+l*(m+1),T=M+1+l*(m+1),w=M+1+l*m;f.push(b,v,w),f.push(v,T,w)}this.setIndex(f),this.setAttribute("position",new nt(g,3)),this.setAttribute("normal",new nt(x,3)),this.setAttribute("uv",new nt(p,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.width,e.height,e.widthSegments,e.heightSegments)}},wa=class n extends mt{constructor(e=.5,t=1,i=32,s=1,r=0,a=Math.PI*2){super(),this.type="RingGeometry",this.parameters={innerRadius:e,outerRadius:t,thetaSegments:i,phiSegments:s,thetaStart:r,thetaLength:a},i=Math.max(3,i),s=Math.max(1,s);let o=[],c=[],l=[],h=[],d=e,u=(t-e)/s,f=new A,g=new te;for(let x=0;x<=s;x++){for(let p=0;p<=i;p++){let m=r+p/i*a;f.x=d*Math.cos(m),f.y=d*Math.sin(m),c.push(f.x,f.y,f.z),l.push(0,0,1),g.x=(f.x/t+1)/2,g.y=(f.y/t+1)/2,h.push(g.x,g.y)}d+=u}for(let x=0;x<s;x++){let p=x*(i+1);for(let m=0;m<i;m++){let M=m+p,b=M,v=M+i+1,T=M+i+2,w=M+1;o.push(b,v,w),o.push(v,T,w)}}this.setIndex(o),this.setAttribute("position",new nt(c,3)),this.setAttribute("normal",new nt(l,3)),this.setAttribute("uv",new nt(h,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new n(e.innerRadius,e.outerRadius,e.thetaSegments,e.phiSegments,e.thetaStart,e.thetaLength)}};var Ta=class extends mt{constructor(e=null){if(super(),this.type="WireframeGeometry",this.parameters={geometry:e},e!==null){let t=[],i=new Set,s=new A,r=new A;if(e.index!==null){let a=e.attributes.position,o=e.index,c=e.groups;c.length===0&&(c=[{start:0,count:o.count,materialIndex:0}]);for(let l=0,h=c.length;l<h;++l){let d=c[l],u=d.start,f=d.count;for(let g=u,x=u+f;g<x;g+=3)for(let p=0;p<3;p++){let m=o.getX(g+p),M=o.getX(g+(p+1)%3);s.fromBufferAttribute(a,m),r.fromBufferAttribute(a,M),Kd(s,r,i)===!0&&(t.push(s.x,s.y,s.z),t.push(r.x,r.y,r.z))}}}else{let a=e.attributes.position;for(let o=0,c=a.count/3;o<c;o++)for(let l=0;l<3;l++){let h=3*o+l,d=3*o+(l+1)%3;s.fromBufferAttribute(a,h),r.fromBufferAttribute(a,d),Kd(s,r,i)===!0&&(t.push(s.x,s.y,s.z),t.push(r.x,r.y,r.z))}}this.setAttribute("position",new nt(t,3))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}};function Kd(n,e,t){let i=`${n.x},${n.y},${n.z}-${e.x},${e.y},${e.z}`,s=`${e.x},${e.y},${e.z}-${n.x},${n.y},${n.z}`;return t.has(i)===!0||t.has(s)===!0?!1:(t.add(i),t.add(s),!0)}function Fs(n){let e={};for(let t in n){e[t]={};for(let i in n[t]){let s=n[t][i];if(jd(s))s.isRenderTargetTexture?(Ye("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][i]=null):e[t][i]=s.clone();else if(Array.isArray(s))if(jd(s[0])){let r=[];for(let a=0,o=s.length;a<o;a++)r[a]=s[a].clone();e[t][i]=r}else e[t][i]=s.slice();else e[t][i]=s}}return e}function ci(n){let e={};for(let t=0;t<n.length;t++){let i=Fs(n[t]);for(let s in i)e[s]=i[s]}return e}function jd(n){return n&&(n.isColor||n.isMatrix3||n.isMatrix4||n.isVector2||n.isVector3||n.isVector4||n.isTexture||n.isQuaternion)}function Fg(n){let e=[];for(let t=0;t<n.length;t++)e.push(n[t].clone());return e}function fu(n){let e=n.getRenderTarget();return e===null?n.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:ht.workingColorSpace}var fi={clone:Fs,merge:ci},Og=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,Bg=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,Rt=class extends ki{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=Og,this.fragmentShader=Bg,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Fs(e.uniforms),this.uniformsGroups=Fg(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){let t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(let s in this.uniforms){let a=this.uniforms[s].value;a&&a.isTexture?t.uniforms[s]={type:"t",value:a.toJSON(e).uuid}:a&&a.isColor?t.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?t.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?t.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?t.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?t.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?t.uniforms[s]={type:"m4",value:a.toArray()}:t.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;let i={};for(let s in this.extensions)this.extensions[s]===!0&&(i[s]=!0);return Object.keys(i).length>0&&(t.extensions=i),t}fromJSON(e,t){if(super.fromJSON(e,t),e.uniforms!==void 0)for(let i in e.uniforms){let s=e.uniforms[i];switch(this.uniforms[i]={},s.type){case"t":this.uniforms[i].value=t[s.value]||null;break;case"c":this.uniforms[i].value=new Le().setHex(s.value);break;case"v2":this.uniforms[i].value=new te().fromArray(s.value);break;case"v3":this.uniforms[i].value=new A().fromArray(s.value);break;case"v4":this.uniforms[i].value=new gt().fromArray(s.value);break;case"m3":this.uniforms[i].value=new je().fromArray(s.value);break;case"m4":this.uniforms[i].value=new rt().fromArray(s.value);break;default:this.uniforms[i].value=s.value}}if(e.defines!==void 0&&(this.defines=e.defines),e.vertexShader!==void 0&&(this.vertexShader=e.vertexShader),e.fragmentShader!==void 0&&(this.fragmentShader=e.fragmentShader),e.glslVersion!==void 0&&(this.glslVersion=e.glslVersion),e.extensions!==void 0)for(let i in e.extensions)this.extensions[i]=e.extensions[i];return e.lights!==void 0&&(this.lights=e.lights),e.clipping!==void 0&&(this.clipping=e.clipping),this}},Sr=class extends Rt{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}},Qe=class extends ki{constructor(e){super(),this.isMeshStandardMaterial=!0,this.type="MeshStandardMaterial",this.defines={STANDARD:""},this.color=new Le(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Le(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Cr,this.normalScale=new te(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ri,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.defines={STANDARD:""},this.color.copy(e.color),this.roughness=e.roughness,this.metalness=e.metalness,this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.roughnessMap=e.roughnessMap,this.metalnessMap=e.metalnessMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.envMapIntensity=e.envMapIntensity,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}};var Aa=class extends ki{constructor(e){super(),this.isMeshNormalMaterial=!0,this.type="MeshNormalMaterial",this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Cr,this.normalScale=new te(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.flatShading=!1,this.setValues(e)}copy(e){return super.copy(e),this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.flatShading=e.flatShading,this}},Ra=class extends ki{constructor(e){super(),this.isMeshLambertMaterial=!0,this.type="MeshLambertMaterial",this.color=new Le(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Le(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Cr,this.normalScale=new te(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ri,this.combine=Ol,this.reflectivity=1,this.envMapIntensity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.envMapIntensity=e.envMapIntensity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}},yl=class extends ki{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Af,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}},Ml=class extends ki{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}};function Wo(n,e){return!n||n.constructor===e?n:typeof e.BYTES_PER_ELEMENT=="number"?new e(n):Array.prototype.slice.call(n)}var es=class{constructor(e,t,i,s){this.parameterPositions=e,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new t.constructor(i),this.sampleValues=t,this.valueSize=i,this.settings=null,this.DefaultSettings_={}}evaluate(e){let t=this.parameterPositions,i=this._cachedIndex,s=t[i],r=t[i-1];i:{e:{let a;t:{n:if(!(e<s)){for(let o=i+2;;){if(s===void 0){if(e<r)break n;return i=t.length,this._cachedIndex=i,this.copySampleValue_(i-1)}if(i===o)break;if(r=s,s=t[++i],e<s)break e}a=t.length;break t}if(!(e>=r)){let o=t[1];e<o&&(i=2,r=o);for(let c=i-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(i===c)break;if(s=r,r=t[--i-1],e>=r)break e}a=i,i=0;break t}break i}for(;i<a;){let o=i+a>>>1;e<t[o]?a=o:i=o+1}if(s=t[i],r=t[i-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return i=t.length,this._cachedIndex=i,this.copySampleValue_(i-1)}this._cachedIndex=i,this.intervalChanged_(i,r,s)}return this.interpolate_(i,r,e,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(e){let t=this.resultBuffer,i=this.sampleValues,s=this.valueSize,r=e*s;for(let a=0;a!==s;++a)t[a]=i[r+a];return t}interpolate_(){throw new Error("THREE.Interpolant: Call to abstract method.")}intervalChanged_(){}},bl=class extends es{constructor(e,t,i,s){super(e,t,i,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:kh,endingEnd:kh}}intervalChanged_(e,t,i){let s=this.parameterPositions,r=e-2,a=e+1,o=s[r],c=s[a];if(o===void 0)switch(this.getSettings_().endingStart){case Hh:r=e,o=2*t-i;break;case Vh:r=s.length-2,o=t+s[r]-s[r+1];break;default:r=e,o=i}if(c===void 0)switch(this.getSettings_().endingEnd){case Hh:a=e,c=2*i-t;break;case Vh:a=1,c=i+s[1]-s[0];break;default:a=e-1,c=t}let l=(i-t)*.5,h=this.valueSize;this._weightPrev=l/(t-o),this._weightNext=l/(c-i),this._offsetPrev=r*h,this._offsetNext=a*h}interpolate_(e,t,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=e*o,l=c-o,h=this._offsetPrev,d=this._offsetNext,u=this._weightPrev,f=this._weightNext,g=(i-t)/(s-t),x=g*g,p=x*g,m=-u*p+2*u*x-u*g,M=(1+u)*p+(-1.5-2*u)*x+(-.5+u)*g+1,b=(-1-f)*p+(1.5+f)*x+.5*g,v=f*p-f*x;for(let T=0;T!==o;++T)r[T]=m*a[h+T]+M*a[l+T]+b*a[c+T]+v*a[d+T];return r}},Sl=class extends es{constructor(e,t,i,s){super(e,t,i,s)}interpolate_(e,t,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=e*o,l=c-o,h=(i-t)/(s-t),d=1-h;for(let u=0;u!==o;++u)r[u]=a[l+u]*d+a[c+u]*h;return r}},El=class extends es{constructor(e,t,i,s){super(e,t,i,s)}interpolate_(e){return this.copySampleValue_(e-1)}},wl=class extends es{interpolate_(e,t,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=e*o,l=c-o,h=this.inTangents,d=this.outTangents;if(!h||!d){let g=(i-t)/(s-t),x=1-g;for(let p=0;p!==o;++p)r[p]=a[l+p]*x+a[c+p]*g;return r}let u=o*2,f=e-1;for(let g=0;g!==o;++g){let x=a[l+g],p=a[c+g],m=f*u+g*2,M=d[m],b=d[m+1],v=e*u+g*2,T=h[v],w=h[v+1],C=(i-t)/(s-t),_,E,P,I,L;for(let X=0;X<8;X++){_=C*C,E=_*C,P=1-C,I=P*P,L=I*P;let U=L*t+3*I*C*M+3*P*_*T+E*s-i;if(Math.abs(U)<1e-10)break;let z=3*I*(M-t)+6*P*C*(T-M)+3*_*(s-T);if(Math.abs(z)<1e-10)break;C=C-U/z,C=Math.max(0,Math.min(1,C))}r[g]=L*x+3*I*C*b+3*P*_*w+E*p}return r}},Li=class{constructor(e,t,i,s){if(e===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(t===void 0||t.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+e);this.name=e,this.times=Wo(t,this.TimeBufferType),this.values=Wo(i,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(e){let t=e.constructor,i;if(t.toJSON!==this.toJSON)i=t.toJSON(e);else{i={name:e.name,times:Wo(e.times,Array),values:Wo(e.values,Array)};let s=e.getInterpolation();s!==e.DefaultInterpolation&&(i.interpolation=s)}return i.type=e.ValueTypeName,i}InterpolantFactoryMethodDiscrete(e){return new El(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodLinear(e){return new Sl(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodSmooth(e){return new bl(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodBezier(e){let t=new wl(this.times,this.values,this.getValueSize(),e);return this.settings&&(t.inTangents=this.settings.inTangents,t.outTangents=this.settings.outTangents),t}setInterpolation(e){let t;switch(e){case ta:t=this.InterpolantFactoryMethodDiscrete;break;case rl:t=this.InterpolantFactoryMethodLinear;break;case $o:t=this.InterpolantFactoryMethodSmooth;break;case zh:t=this.InterpolantFactoryMethodBezier;break}if(t===void 0){let i="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(e!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(i);return Ye("KeyframeTrack:",i),this}return this.createInterpolant=t,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return ta;case this.InterpolantFactoryMethodLinear:return rl;case this.InterpolantFactoryMethodSmooth:return $o;case this.InterpolantFactoryMethodBezier:return zh}}getValueSize(){return this.values.length/this.times.length}shift(e){if(e!==0){let t=this.times;for(let i=0,s=t.length;i!==s;++i)t[i]+=e}return this}scale(e){if(e!==1){let t=this.times;for(let i=0,s=t.length;i!==s;++i)t[i]*=e}return this}trim(e,t){let i=this.times,s=i.length,r=0,a=s-1;for(;r!==s&&i[r]<e;)++r;for(;a!==-1&&i[a]>t;)--a;if(++a,r!==0||a!==s){r>=a&&(a=Math.max(a,1),r=a-1);let o=this.getValueSize();this.times=i.slice(r,a),this.values=this.values.slice(r*o,a*o)}return this}validate(){let e=!0,t=this.getValueSize();t-Math.floor(t)!==0&&($e("KeyframeTrack: Invalid value size in track.",this),e=!1);let i=this.times,s=this.values,r=i.length;r===0&&($e("KeyframeTrack: Track is empty.",this),e=!1);let a=null;for(let o=0;o!==r;o++){let c=i[o];if(typeof c=="number"&&isNaN(c)){$e("KeyframeTrack: Time is not a valid number.",this,o,c),e=!1;break}if(a!==null&&a>c){$e("KeyframeTrack: Out of order keys.",this,o,c,a),e=!1;break}a=c}if(s!==void 0&&Rm(s))for(let o=0,c=s.length;o!==c;++o){let l=s[o];if(isNaN(l)){$e("KeyframeTrack: Value is not a valid number.",this,o,l),e=!1;break}}return e}optimize(){let e=this.times.slice(),t=this.values.slice(),i=this.getValueSize(),s=this.getInterpolation()===$o,r=e.length-1,a=1;for(let o=1;o<r;++o){let c=!1,l=e[o],h=e[o+1];if(l!==h&&(o!==1||l!==e[0]))if(s)c=!0;else{let d=o*i,u=d-i,f=d+i;for(let g=0;g!==i;++g){let x=t[d+g];if(x!==t[u+g]||x!==t[f+g]){c=!0;break}}}if(c){if(o!==a){e[a]=e[o];let d=o*i,u=a*i;for(let f=0;f!==i;++f)t[u+f]=t[d+f]}++a}}if(r>0){e[a]=e[r];for(let o=r*i,c=a*i,l=0;l!==i;++l)t[c+l]=t[o+l];++a}return a!==e.length?(this.times=e.slice(0,a),this.values=t.slice(0,a*i)):(this.times=e,this.values=t),this}clone(){let e=this.times.slice(),t=this.values.slice(),i=this.constructor,s=new i(this.name,e,t);return s.createInterpolant=this.createInterpolant,s}};Li.prototype.ValueTypeName="";Li.prototype.TimeBufferType=Float32Array;Li.prototype.ValueBufferType=Float32Array;Li.prototype.DefaultInterpolation=rl;var ts=class extends Li{constructor(e,t,i){super(e,t,i)}};ts.prototype.ValueTypeName="bool";ts.prototype.ValueBufferType=Array;ts.prototype.DefaultInterpolation=ta;ts.prototype.InterpolantFactoryMethodLinear=void 0;ts.prototype.InterpolantFactoryMethodSmooth=void 0;var Tl=class extends Li{constructor(e,t,i,s){super(e,t,i,s)}};Tl.prototype.ValueTypeName="color";var Al=class extends Li{constructor(e,t,i,s){super(e,t,i,s)}};Al.prototype.ValueTypeName="number";var Rl=class extends es{constructor(e,t,i,s){super(e,t,i,s)}interpolate_(e,t,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=(i-t)/(s-t),l=e*o;for(let h=l+o;l!==h;l+=4)Ai.slerpFlat(r,0,a,l-o,a,l,c);return r}},Ca=class extends Li{constructor(e,t,i,s){super(e,t,i,s)}InterpolantFactoryMethodLinear(e){return new Rl(this.times,this.values,this.getValueSize(),e)}};Ca.prototype.ValueTypeName="quaternion";Ca.prototype.InterpolantFactoryMethodSmooth=void 0;var is=class extends Li{constructor(e,t,i){super(e,t,i)}};is.prototype.ValueTypeName="string";is.prototype.ValueBufferType=Array;is.prototype.DefaultInterpolation=ta;is.prototype.InterpolantFactoryMethodLinear=void 0;is.prototype.InterpolantFactoryMethodSmooth=void 0;var Cl=class extends Li{constructor(e,t,i,s){super(e,t,i,s)}};Cl.prototype.ValueTypeName="vector";var Pl=class{constructor(e,t,i){let s=this,r=!1,a=0,o=0,c,l=[];this.onStart=void 0,this.onLoad=e,this.onProgress=t,this.onError=i,this._abortController=null,this.itemStart=function(h){o++,r===!1&&s.onStart!==void 0&&s.onStart(h,a,o),r=!0},this.itemEnd=function(h){a++,s.onProgress!==void 0&&s.onProgress(h,a,o),a===o&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return h=h.normalize("NFC"),c?c(h):h},this.setURLModifier=function(h){return c=h,this},this.addHandler=function(h,d){return l.push(h,d),this},this.removeHandler=function(h){let d=l.indexOf(h);return d!==-1&&l.splice(d,2),this},this.getHandler=function(h){for(let d=0,u=l.length;d<u;d+=2){let f=l[d],g=l[d+1];if(f.global&&(f.lastIndex=0),f.test(h))return g}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}},Xf=new Pl,Il=class{constructor(e){this.manager=e!==void 0?e:Xf,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(e,t){let i=this;return new Promise(function(s,r){i.load(e,s,t,r)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}};Il.DEFAULT_MATERIAL_NAME="__DEFAULT";var Er=class extends pt{constructor(e,t=1){super(),this.isLight=!0,this.type="Light",this.color=new Le(e),this.intensity=t}dispose(){this.dispatchEvent({type:"dispose"})}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){let t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,t}},Pa=class extends Er{constructor(e,t,i){super(e,i),this.isHemisphereLight=!0,this.type="HemisphereLight",this.position.copy(pt.DEFAULT_UP),this.updateMatrix(),this.groundColor=new Le(t)}copy(e,t){return super.copy(e,t),this.groundColor.copy(e.groundColor),this}toJSON(e){let t=super.toJSON(e);return t.object.groundColor=this.groundColor.getHex(),t}},Oh=new rt,Qd=new A,ef=new A,Dl=class{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.biasNode=null,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new te(512,512),this.mapType=li,this.map=null,this.mapPass=null,this.matrix=new rt,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new vr,this._frameExtents=new te(1,1),this._viewportCount=1,this._viewports=[new gt(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){let t=this.camera,i=this.matrix;Qd.setFromMatrixPosition(e.matrixWorld),t.position.copy(Qd),ef.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(ef),t.updateMatrixWorld(),Oh.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix(Oh,t.coordinateSystem,t.reversedDepth),t.coordinateSystem===dr||t.reversedDepth?i.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):i.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),i.multiply(Oh)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this.biasNode=e.biasNode,this}clone(){return new this.constructor().copy(this)}toJSON(){let e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}},Xo=new A,qo=new Ai,fn=new A,Ia=class extends pt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new rt,this.projectionMatrix=new rt,this.projectionMatrixInverse=new rt,this.coordinateSystem=Ki,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(Xo,qo,fn),fn.x===1&&fn.y===1&&fn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Xo,qo,fn.set(1,1,1)).invert()}updateWorldMatrix(e,t,i=!1){super.updateWorldMatrix(e,t,i),this.matrixWorld.decompose(Xo,qo,fn),fn.x===1&&fn.y===1&&fn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Xo,qo,fn.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},Kn=new A,tf=new te,nf=new te,Kt=class extends Ia{constructor(e=50,t=1,i=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=i,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){let t=.5*this.getFilmHeight()/e;this.fov=pr*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){let e=Math.tan(hr*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return pr*2*Math.atan(Math.tan(hr*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,i){Kn.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(Kn.x,Kn.y).multiplyScalar(-e/Kn.z),Kn.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(Kn.x,Kn.y).multiplyScalar(-e/Kn.z)}getViewSize(e,t){return this.getViewBounds(e,tf,nf),t.subVectors(nf,tf)}setViewOffset(e,t,i,s,r,a){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=this.near,t=e*Math.tan(hr*.5*this.fov)/this.zoom,i=2*t,s=this.aspect*i,r=-.5*s,a=this.view;if(this.view!==null&&this.view.enabled){let c=a.fullWidth,l=a.fullHeight;r+=a.offsetX*s/c,t-=a.offsetY*i/l,s*=a.width/c,i*=a.height/l}let o=this.filmOffset;o!==0&&(r+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-i,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}};var $h=class extends Dl{constructor(){super(new Kt(90,1,.5,500)),this.isPointLightShadow=!0}},Da=class extends Er{constructor(e,t,i=0,s=2){super(e,t),this.isPointLight=!0,this.type="PointLight",this.distance=i,this.decay=s,this.shadow=new $h}get power(){return this.intensity*4*Math.PI}set power(e){this.intensity=e/(4*Math.PI)}dispose(){super.dispose(),this.shadow.dispose()}copy(e,t){return super.copy(e,t),this.distance=e.distance,this.decay=e.decay,this.shadow=e.shadow.clone(),this}toJSON(e){let t=super.toJSON(e);return t.object.distance=this.distance,t.object.decay=this.decay,t.object.shadow=this.shadow.toJSON(),t}},ns=class extends Ia{constructor(e=-1,t=1,i=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=i,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,i,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=i-e,a=i+e,o=s+t,c=s-t;if(this.view!==null&&this.view.enabled){let l=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=l*this.view.offsetX,a=r+l*this.view.width,o-=h*this.view.offsetY,c=o-h*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,c,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}},Zh=class extends Dl{constructor(){super(new ns(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}},wr=class extends Er{constructor(e,t){super(e,t),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(pt.DEFAULT_UP),this.updateMatrix(),this.target=new pt,this.shadow=new Zh}dispose(){super.dispose(),this.shadow.dispose()}copy(e){return super.copy(e),this.target=e.target.clone(),this.shadow=e.shadow.clone(),this}toJSON(e){let t=super.toJSON(e);return t.object.shadow=this.shadow.toJSON(),t.object.target=this.target.uuid,t}};var La=class extends mt{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){let e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}};var ar=-90,or=1,Ll=class extends pt{constructor(e,t,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new Kt(ar,or,e,t);s.layers=this.layers,this.add(s);let r=new Kt(ar,or,e,t);r.layers=this.layers,this.add(r);let a=new Kt(ar,or,e,t);a.layers=this.layers,this.add(a);let o=new Kt(ar,or,e,t);o.layers=this.layers,this.add(o);let c=new Kt(ar,or,e,t);c.layers=this.layers,this.add(c);let l=new Kt(ar,or,e,t);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){let e=this.coordinateSystem,t=this.children.concat(),[i,s,r,a,o,c]=t;for(let l of t)this.remove(l);if(e===Ki)i.up.set(0,1,0),i.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(e===dr)i.up.set(0,-1,0),i.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(let l of t)this.add(l),l.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();let{renderTarget:i,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());let[r,a,o,c,l,h]=this.children,d=e.getRenderTarget(),u=e.getActiveCubeFace(),f=e.getActiveMipmapLevel(),g=e.xr.enabled;e.xr.enabled=!1;let x=i.texture.generateMipmaps;i.texture.generateMipmaps=!1;let p=!1;e.isWebGLRenderer===!0?p=e.state.buffers.depth.getReversed():p=e.reversedDepthBuffer,e.setRenderTarget(i,0,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,r),e.setRenderTarget(i,1,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,a),e.setRenderTarget(i,2,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,o),e.setRenderTarget(i,3,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,c),e.setRenderTarget(i,4,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,l),i.texture.generateMipmaps=x,e.setRenderTarget(i,5,s),p&&e.autoClear===!1&&e.clearDepth(),e.render(t,h),e.setRenderTarget(d,u,f),e.xr.enabled=g,i.texture.needsPMREMUpdate=!0}},Nl=class extends Kt{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}},Na=class{constructor(){this._previousTime=0,this._currentTime=0,this._startTime=performance.now(),this._delta=0,this._elapsed=0,this._timescale=1,this._document=null,this._pageVisibilityHandler=null}connect(e){this._document=e,e.hidden!==void 0&&(this._pageVisibilityHandler=zg.bind(this),e.addEventListener("visibilitychange",this._pageVisibilityHandler,!1))}disconnect(){this._pageVisibilityHandler!==null&&(this._document.removeEventListener("visibilitychange",this._pageVisibilityHandler),this._pageVisibilityHandler=null),this._document=null}getDelta(){return this._delta/1e3}getElapsed(){return this._elapsed/1e3}getTimescale(){return this._timescale}setTimescale(e){return this._timescale=e,this}reset(){return this._currentTime=performance.now()-this._startTime,this}dispose(){this.disconnect()}update(e){return this._pageVisibilityHandler!==null&&this._document.hidden===!0?this._delta=0:(this._previousTime=this._currentTime,this._currentTime=(e!==void 0?e:performance.now())-this._startTime,this._delta=(this._currentTime-this._previousTime)*this._timescale,this._elapsed+=this._delta),this}};function zg(){this._document.hidden===!1&&this.reset()}var pu="\\[\\]\\.:\\/",kg=new RegExp("["+pu+"]","g"),mu="[^"+pu+"]",Hg="[^"+pu.replace("\\.","")+"]",Vg=/((?:WC+[\/:])*)/.source.replace("WC",mu),Gg=/(WCOD+)?/.source.replace("WCOD",Hg),Wg=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",mu),Xg=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",mu),qg=new RegExp("^"+Vg+Gg+Wg+Xg+"$"),Yg=["material","materials","bones","map"],Jh=class{constructor(e,t,i){let s=i||Tt.parseTrackName(t);this._targetGroup=e,this._bindings=e.subscribe_(t,s)}getValue(e,t){this.bind();let i=this._targetGroup.nCachedObjects_,s=this._bindings[i];s!==void 0&&s.getValue(e,t)}setValue(e,t){let i=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=i.length;s!==r;++s)i[s].setValue(e,t)}bind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,i=e.length;t!==i;++t)e[t].bind()}unbind(){let e=this._bindings;for(let t=this._targetGroup.nCachedObjects_,i=e.length;t!==i;++t)e[t].unbind()}},Tt=class n{constructor(e,t,i){this.path=t,this.parsedPath=i||n.parseTrackName(t),this.node=n.findNode(e,this.parsedPath.nodeName),this.rootNode=e,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(e,t,i){return e&&e.isAnimationObjectGroup?new n.Composite(e,t,i):new n(e,t,i)}static sanitizeNodeName(e){return e.replace(/\s/g,"_").replace(kg,"")}static parseTrackName(e){let t=qg.exec(e);if(t===null)throw new Error("THREE.PropertyBinding: Cannot parse trackName: "+e);let i={nodeName:t[2],objectName:t[3],objectIndex:t[4],propertyName:t[5],propertyIndex:t[6]},s=i.nodeName&&i.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=i.nodeName.substring(s+1);Yg.indexOf(r)!==-1&&(i.nodeName=i.nodeName.substring(0,s),i.objectName=r)}if(i.propertyName===null||i.propertyName.length===0)throw new Error("THREE.PropertyBinding: can not parse propertyName from trackName: "+e);return i}static findNode(e,t){if(t===void 0||t===""||t==="."||t===-1||t===e.name||t===e.uuid)return e;if(e.skeleton){let i=e.skeleton.getBoneByName(t);if(i!==void 0)return i}if(e.children){let i=function(r){for(let a=0;a<r.length;a++){let o=r[a];if(o.name===t||o.uuid===t)return o;let c=i(o.children);if(c)return c}return null},s=i(e.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,t){e[t]=this.targetObject[this.propertyName]}_getValue_array(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)e[t++]=i[s]}_getValue_arrayElement(e,t){e[t]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,t){this.resolvedProperty.toArray(e,t)}_setValue_direct(e,t){this.targetObject[this.propertyName]=e[t]}_setValue_direct_setNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,t){this.targetObject[this.propertyName]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[t++]}_setValue_array_setNeedsUpdate(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[t++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,t){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[t++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,t){this.resolvedProperty[this.propertyIndex]=e[t]}_setValue_arrayElement_setNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty[this.propertyIndex]=e[t],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,t){this.resolvedProperty.fromArray(e,t)}_setValue_fromArray_setNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,t){this.resolvedProperty.fromArray(e,t),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,t){this.bind(),this.getValue(e,t)}_setValue_unbound(e,t){this.bind(),this.setValue(e,t)}bind(){let e=this.node,t=this.parsedPath,i=t.objectName,s=t.propertyName,r=t.propertyIndex;if(e||(e=n.findNode(this.rootNode,t.nodeName),this.node=e),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!e){Ye("PropertyBinding: No target node found for track: "+this.path+".");return}if(i){let l=t.objectIndex;switch(i){case"materials":if(!e.material){$e("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.materials){$e("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}e=e.material.materials;break;case"bones":if(!e.skeleton){$e("PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}e=e.skeleton.bones;for(let h=0;h<e.length;h++)if(e[h].name===l){l=h;break}break;case"map":if("map"in e){e=e.map;break}if(!e.material){$e("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.map){$e("PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}e=e.material.map;break;default:if(e[i]===void 0){$e("PropertyBinding: Can not bind to objectName of node undefined.",this);return}e=e[i]}if(l!==void 0){if(e[l]===void 0){$e("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,e);return}e=e[l]}}let a=e[s];if(a===void 0){let l=t.nodeName;$e("PropertyBinding: Trying to update property for track: "+l+"."+s+" but it wasn't found.",e);return}let o=this.Versioning.None;this.targetObject=e,e.isMaterial===!0?o=this.Versioning.NeedsUpdate:e.isObject3D===!0&&(o=this.Versioning.MatrixWorldNeedsUpdate);let c=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!e.geometry){$e("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!e.geometry.morphAttributes){$e("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}e.morphTargetDictionary[r]!==void 0&&(r=e.morphTargetDictionary[r])}c=this.BindingType.ArrayElement,this.resolvedProperty=a,this.propertyIndex=r}else a.fromArray!==void 0&&a.toArray!==void 0?(c=this.BindingType.HasFromToArray,this.resolvedProperty=a):Array.isArray(a)?(c=this.BindingType.EntireArray,this.resolvedProperty=a):this.propertyName=s;this.getValue=this.GetterByBindingType[c],this.setValue=this.SetterByBindingTypeAndVersioning[c][o]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};Tt.Composite=Jh;Tt.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};Tt.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};Tt.prototype.GetterByBindingType=[Tt.prototype._getValue_direct,Tt.prototype._getValue_array,Tt.prototype._getValue_arrayElement,Tt.prototype._getValue_toArray];Tt.prototype.SetterByBindingTypeAndVersioning=[[Tt.prototype._setValue_direct,Tt.prototype._setValue_direct_setNeedsUpdate,Tt.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[Tt.prototype._setValue_array,Tt.prototype._setValue_array_setNeedsUpdate,Tt.prototype._setValue_array_setMatrixWorldNeedsUpdate],[Tt.prototype._setValue_arrayElement,Tt.prototype._setValue_arrayElement_setNeedsUpdate,Tt.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[Tt.prototype._setValue_fromArray,Tt.prototype._setValue_fromArray_setNeedsUpdate,Tt.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var ZM=new Float32Array(1);var ss=class extends ha{constructor(e,t,i=1){super(e,t),this.isInstancedInterleavedBuffer=!0,this.meshPerAttribute=i}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}clone(e){let t=super.clone(e);return t.meshPerAttribute=this.meshPerAttribute,t}toJSON(e){let t=super.toJSON(e);return t.isInstancedInterleavedBuffer=!0,t.meshPerAttribute=this.meshPerAttribute,t}};var sf=new rt,Ua=class{constructor(e,t,i=0,s=1/0){this.ray=new jn(e,t),this.near=i,this.far=s,this.camera=null,this.layers=new gr,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(e,t){this.ray.set(e,t)}setFromCamera(e,t){t.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t):t.isOrthographicCamera?(this.ray.origin.set(e.x,e.y,t.projectionMatrix.elements[14]).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t):$e("Raycaster: Unsupported camera type: "+t.type)}setFromXRController(e){return sf.identity().extractRotation(e.matrixWorld),this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(sf),this}intersectObject(e,t=!0,i=[]){return Kh(e,this,i,t),i.sort(rf),i}intersectObjects(e,t=!0,i=[]){for(let s=0,r=e.length;s<r;s++)Kh(e[s],this,i,t);return i.sort(rf),i}};function rf(n,e){return n.distance-e.distance}function Kh(n,e,t,i){let s=!0;if(n.layers.test(e.layers)&&n.raycast(e,t)===!1&&(s=!1),s===!0&&i===!0){let r=n.children;for(let a=0,o=r.length;a<o;a++)Kh(r[a],e,t,!0)}}var Tr=class{constructor(e=1,t=0,i=0){this.radius=e,this.phi=t,this.theta=i}set(e,t,i){return this.radius=e,this.phi=t,this.theta=i,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=Ke(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,i){return this.radius=Math.sqrt(e*e+t*t+i*i),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,i),this.phi=Math.acos(Ke(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}};var Mu=class Mu{constructor(e,t,i,s){this.elements=[1,0,0,1],e!==void 0&&this.set(e,t,i,s)}identity(){return this.set(1,0,0,1),this}fromArray(e,t=0){for(let i=0;i<4;i++)this.elements[i]=e[i+t];return this}set(e,t,i,s){let r=this.elements;return r[0]=e,r[2]=t,r[1]=i,r[3]=s,this}};Mu.prototype.isMatrix2=!0;var jh=Mu;var af=new A,Yo=new A,lr=new A,cr=new A,Bh=new A,$g=new A,Zg=new A,Fa=class{constructor(e=new A,t=new A){this.start=e,this.end=t}set(e,t){return this.start.copy(e),this.end.copy(t),this}copy(e){return this.start.copy(e.start),this.end.copy(e.end),this}getCenter(e){return e.addVectors(this.start,this.end).multiplyScalar(.5)}delta(e){return e.subVectors(this.end,this.start)}distanceSq(){return this.start.distanceToSquared(this.end)}distance(){return this.start.distanceTo(this.end)}at(e,t){return this.delta(t).multiplyScalar(e).add(this.start)}closestPointToPointParameter(e,t){af.subVectors(e,this.start),Yo.subVectors(this.end,this.start);let i=Yo.dot(Yo);if(i===0)return 0;let r=Yo.dot(af)/i;return t&&(r=Ke(r,0,1)),r}closestPointToPoint(e,t,i){let s=this.closestPointToPointParameter(e,t);return this.delta(i).multiplyScalar(s).add(this.start)}distanceSqToLine3(e,t=$g,i=Zg){let s=10000000000000001e-32,r,a,o=this.start,c=e.start,l=this.end,h=e.end;lr.subVectors(l,o),cr.subVectors(h,c),Bh.subVectors(o,c);let d=lr.dot(lr),u=cr.dot(cr),f=cr.dot(Bh);if(d<=s&&u<=s)return t.copy(o),i.copy(c),t.sub(i),t.dot(t);if(d<=s)r=0,a=f/u,a=Ke(a,0,1);else{let g=lr.dot(Bh);if(u<=s)a=0,r=Ke(-g/d,0,1);else{let x=lr.dot(cr),p=d*u-x*x;p!==0?r=Ke((x*f-g*u)/p,0,1):r=0,a=(x*r+f)/u,a<0?(a=0,r=Ke(-g/d,0,1)):a>1&&(a=1,r=Ke((x-g)/d,0,1))}}return t.copy(o).addScaledVector(lr,r),i.copy(c).addScaledVector(cr,a),t.distanceToSquared(i)}applyMatrix4(e){return this.start.applyMatrix4(e),this.end.applyMatrix4(e),this}equals(e){return e.start.equals(this.start)&&e.end.equals(this.end)}clone(){return new this.constructor().copy(this)}};var Oa=class extends Qi{constructor(e,t=null){super(),this.object=e,this.domElement=t,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(e){if(e===void 0){Ye("Controls: connect() now requires an element.");return}this.domElement!==null&&this.disconnect(),this.domElement=e}disconnect(){}dispose(){}update(){}};function gu(n,e,t,i){let s=Jg(i);switch(t){case lu:return n*e;case Wl:return n*e/s.components*s.byteLength;case Xl:return n*e/s.components*s.byteLength;case us:return n*e*2/s.components*s.byteLength;case ql:return n*e*2/s.components*s.byteLength;case cu:return n*e*3/s.components*s.byteLength;case vi:return n*e*4/s.components*s.byteLength;case Yl:return n*e*4/s.components*s.byteLength;case Ya:case $a:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*8;case Za:case Ja:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case Zl:case Kl:return Math.max(n,16)*Math.max(e,8)/4;case $l:case Jl:return Math.max(n,8)*Math.max(e,8)/2;case jl:case Ql:case tc:case ic:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*8;case ec:case Ka:case nc:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case sc:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case rc:return Math.floor((n+4)/5)*Math.floor((e+3)/4)*16;case ac:return Math.floor((n+4)/5)*Math.floor((e+4)/5)*16;case oc:return Math.floor((n+5)/6)*Math.floor((e+4)/5)*16;case lc:return Math.floor((n+5)/6)*Math.floor((e+5)/6)*16;case cc:return Math.floor((n+7)/8)*Math.floor((e+4)/5)*16;case hc:return Math.floor((n+7)/8)*Math.floor((e+5)/6)*16;case uc:return Math.floor((n+7)/8)*Math.floor((e+7)/8)*16;case dc:return Math.floor((n+9)/10)*Math.floor((e+4)/5)*16;case fc:return Math.floor((n+9)/10)*Math.floor((e+5)/6)*16;case pc:return Math.floor((n+9)/10)*Math.floor((e+7)/8)*16;case mc:return Math.floor((n+9)/10)*Math.floor((e+9)/10)*16;case gc:return Math.floor((n+11)/12)*Math.floor((e+9)/10)*16;case _c:return Math.floor((n+11)/12)*Math.floor((e+11)/12)*16;case xc:case vc:case yc:return Math.ceil(n/4)*Math.ceil(e/4)*16;case Mc:case bc:return Math.ceil(n/4)*Math.ceil(e/4)*8;case ja:case Sc:return Math.ceil(n/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function Jg(n){switch(n){case li:case su:return{byteLength:1,components:1};case Rr:case ru:case ei:return{byteLength:2,components:1};case Vl:case Gl:return{byteLength:2,components:4};case an:case Hl:case Vi:return{byteLength:4,components:1};case au:case ou:return{byteLength:4,components:3}}throw new Error(`THREE.TextureUtils: Unknown texture type ${n}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"185"}}));typeof window<"u"&&(window.__THREE__?Ye("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="185");/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function pp(){let n=null,e=!1,t=null,i=null;function s(r,a){t(r,a),i=n.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&n!==null&&(i=n.requestAnimationFrame(s),e=!0)},stop:function(){n!==null&&n.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){n=r}}}function jg(n){let e=new WeakMap;function t(o,c){let l=o.array,h=o.usage,d=l.byteLength,u=n.createBuffer();n.bindBuffer(c,u),n.bufferData(c,l,h),o.onUploadCallback();let f;if(l instanceof Float32Array)f=n.FLOAT;else if(typeof Float16Array<"u"&&l instanceof Float16Array)f=n.HALF_FLOAT;else if(l instanceof Uint16Array)o.isFloat16BufferAttribute?f=n.HALF_FLOAT:f=n.UNSIGNED_SHORT;else if(l instanceof Int16Array)f=n.SHORT;else if(l instanceof Uint32Array)f=n.UNSIGNED_INT;else if(l instanceof Int32Array)f=n.INT;else if(l instanceof Int8Array)f=n.BYTE;else if(l instanceof Uint8Array)f=n.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)f=n.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:u,type:f,bytesPerElement:l.BYTES_PER_ELEMENT,version:o.version,size:d}}function i(o,c,l){let h=c.array,d=c.updateRanges;if(n.bindBuffer(l,o),d.length===0)n.bufferSubData(l,0,h);else{d.sort((f,g)=>f.start-g.start);let u=0;for(let f=1;f<d.length;f++){let g=d[u],x=d[f];x.start<=g.start+g.count+1?g.count=Math.max(g.count,x.start+x.count-g.start):(++u,d[u]=x)}d.length=u+1;for(let f=0,g=d.length;f<g;f++){let x=d[f];n.bufferSubData(l,x.start*h.BYTES_PER_ELEMENT,h,x.start,x.count)}c.clearUpdateRanges()}c.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),e.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);let c=e.get(o);c&&(n.deleteBuffer(c.buffer),e.delete(o))}function a(o,c){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){let h=e.get(o);(!h||h.version<o.version)&&e.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}let l=e.get(o);if(l===void 0)e.set(o,t(o,c));else if(l.version<o.version){if(l.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(l.buffer,o,c),l.version=o.version}}return{get:s,remove:r,update:a}}var Qg=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,e0=`#ifdef USE_ALPHAHASH
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
#endif`,t0=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,i0=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,n0=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,s0=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,r0=`#ifdef USE_AOMAP
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
#endif`,a0=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,o0=`#ifdef USE_BATCHING
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
#endif`,l0=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,c0=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,h0=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,u0=`float G_BlinnPhong_Implicit( ) {
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
} // validated`,d0=`#ifdef USE_IRIDESCENCE
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
#endif`,f0=`#ifdef USE_BUMPMAP
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
#endif`,p0=`#if NUM_CLIPPING_PLANES > 0
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
#endif`,m0=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,g0=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,_0=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,x0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,v0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,y0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,M0=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
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
#endif`,b0=`#define PI 3.141592653589793
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
} // validated`,S0=`#ifdef ENVMAP_TYPE_CUBE_UV
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
#endif`,E0=`vec3 transformedNormal = objectNormal;
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
#endif`,w0=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,T0=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,A0=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,R0=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,C0="gl_FragColor = linearToOutputTexel( gl_FragColor );",P0=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,I0=`#ifdef USE_ENVMAP
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
#endif`,D0=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,L0=`#ifdef USE_ENVMAP
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
#endif`,N0=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,U0=`#ifdef USE_ENVMAP
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
#endif`,F0=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,O0=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,B0=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,z0=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,k0=`#ifdef USE_GRADIENTMAP
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
}`,H0=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,V0=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,G0=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,W0=`uniform bool receiveShadow;
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
#include <lightprobes_pars_fragment>`,X0=`#ifdef USE_ENVMAP
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
#endif`,q0=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Y0=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,$0=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Z0=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,J0=`PhysicalMaterial material;
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
#endif`,K0=`uniform sampler2D dfgLUT;
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
}`,j0=`
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
#endif`,Q0=`#if defined( RE_IndirectDiffuse )
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
#endif`,e_=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,t_=`#ifdef USE_LIGHT_PROBES_GRID
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
#endif`,i_=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,n_=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,s_=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,r_=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,a_=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,o_=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,l_=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
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
#endif`,c_=`#if defined( USE_POINTS_UV )
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
#endif`,h_=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,u_=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,d_=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,f_=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,p_=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,m_=`#ifdef USE_MORPHTARGETS
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
#endif`,g_=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,__=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
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
vec3 nonPerturbedNormal = normal;`,x_=`#ifdef USE_NORMALMAP_OBJECTSPACE
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
#endif`,v_=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,y_=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,M_=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,b_=`#ifdef USE_NORMALMAP
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
#endif`,S_=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,E_=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,w_=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,T_=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,A_=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,R_=`vec3 packNormalToRGB( const in vec3 normal ) {
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
}`,C_=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,P_=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,I_=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,D_=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,L_=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,N_=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,U_=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,F_=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,O_=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
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
#endif`,B_=`float getShadowMask() {
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
}`,z_=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,k_=`#ifdef USE_SKINNING
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
#endif`,H_=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,V_=`#ifdef USE_SKINNING
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
#endif`,G_=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,W_=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,X_=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,q_=`#ifndef saturate
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
vec3 CustomToneMapping( vec3 color ) { return color; }`,Y_=`#ifdef USE_TRANSMISSION
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
#endif`,$_=`#ifdef USE_TRANSMISSION
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
#endif`,Z_=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,J_=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,K_=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,j_=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,Q_=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,ex=`uniform sampler2D t2D;
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
}`,tx=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,ix=`#ifdef ENVMAP_TYPE_CUBE
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
}`,nx=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,sx=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,rx=`#include <common>
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
}`,ax=`#if DEPTH_PACKING == 3200
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
}`,ox=`#define DISTANCE
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
}`,lx=`#define DISTANCE
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
}`,cx=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,hx=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,ux=`uniform float scale;
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
}`,dx=`uniform vec3 diffuse;
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
}`,fx=`#include <common>
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
}`,px=`uniform vec3 diffuse;
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
}`,mx=`#define LAMBERT
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
}`,gx=`#define LAMBERT
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
}`,_x=`#define MATCAP
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
}`,xx=`#define MATCAP
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
}`,vx=`#define NORMAL
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
}`,yx=`#define NORMAL
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
}`,Mx=`#define PHONG
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
}`,bx=`#define PHONG
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
}`,Sx=`#define STANDARD
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
}`,Ex=`#define STANDARD
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
}`,wx=`#define TOON
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
}`,Tx=`#define TOON
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
}`,Ax=`uniform float size;
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
}`,Rx=`uniform vec3 diffuse;
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
}`,Cx=`#include <common>
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
}`,Px=`uniform vec3 color;
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
}`,Ix=`uniform float rotation;
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
}`,Dx=`uniform vec3 diffuse;
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
}`,at={alphahash_fragment:Qg,alphahash_pars_fragment:e0,alphamap_fragment:t0,alphamap_pars_fragment:i0,alphatest_fragment:n0,alphatest_pars_fragment:s0,aomap_fragment:r0,aomap_pars_fragment:a0,batching_pars_vertex:o0,batching_vertex:l0,begin_vertex:c0,beginnormal_vertex:h0,bsdfs:u0,iridescence_fragment:d0,bumpmap_pars_fragment:f0,clipping_planes_fragment:p0,clipping_planes_pars_fragment:m0,clipping_planes_pars_vertex:g0,clipping_planes_vertex:_0,color_fragment:x0,color_pars_fragment:v0,color_pars_vertex:y0,color_vertex:M0,common:b0,cube_uv_reflection_fragment:S0,defaultnormal_vertex:E0,displacementmap_pars_vertex:w0,displacementmap_vertex:T0,emissivemap_fragment:A0,emissivemap_pars_fragment:R0,colorspace_fragment:C0,colorspace_pars_fragment:P0,envmap_fragment:I0,envmap_common_pars_fragment:D0,envmap_pars_fragment:L0,envmap_pars_vertex:N0,envmap_physical_pars_fragment:X0,envmap_vertex:U0,fog_vertex:F0,fog_pars_vertex:O0,fog_fragment:B0,fog_pars_fragment:z0,gradientmap_pars_fragment:k0,lightmap_pars_fragment:H0,lights_lambert_fragment:V0,lights_lambert_pars_fragment:G0,lights_pars_begin:W0,lights_toon_fragment:q0,lights_toon_pars_fragment:Y0,lights_phong_fragment:$0,lights_phong_pars_fragment:Z0,lights_physical_fragment:J0,lights_physical_pars_fragment:K0,lights_fragment_begin:j0,lights_fragment_maps:Q0,lights_fragment_end:e_,lightprobes_pars_fragment:t_,logdepthbuf_fragment:i_,logdepthbuf_pars_fragment:n_,logdepthbuf_pars_vertex:s_,logdepthbuf_vertex:r_,map_fragment:a_,map_pars_fragment:o_,map_particle_fragment:l_,map_particle_pars_fragment:c_,metalnessmap_fragment:h_,metalnessmap_pars_fragment:u_,morphinstance_vertex:d_,morphcolor_vertex:f_,morphnormal_vertex:p_,morphtarget_pars_vertex:m_,morphtarget_vertex:g_,normal_fragment_begin:__,normal_fragment_maps:x_,normal_pars_fragment:v_,normal_pars_vertex:y_,normal_vertex:M_,normalmap_pars_fragment:b_,clearcoat_normal_fragment_begin:S_,clearcoat_normal_fragment_maps:E_,clearcoat_pars_fragment:w_,iridescence_pars_fragment:T_,opaque_fragment:A_,packing:R_,premultiplied_alpha_fragment:C_,project_vertex:P_,dithering_fragment:I_,dithering_pars_fragment:D_,roughnessmap_fragment:L_,roughnessmap_pars_fragment:N_,shadowmap_pars_fragment:U_,shadowmap_pars_vertex:F_,shadowmap_vertex:O_,shadowmask_pars_fragment:B_,skinbase_vertex:z_,skinning_pars_vertex:k_,skinning_vertex:H_,skinnormal_vertex:V_,specularmap_fragment:G_,specularmap_pars_fragment:W_,tonemapping_fragment:X_,tonemapping_pars_fragment:q_,transmission_fragment:Y_,transmission_pars_fragment:$_,uv_pars_fragment:Z_,uv_pars_vertex:J_,uv_vertex:K_,worldpos_vertex:j_,background_vert:Q_,background_frag:ex,backgroundCube_vert:tx,backgroundCube_frag:ix,cube_vert:nx,cube_frag:sx,depth_vert:rx,depth_frag:ax,distance_vert:ox,distance_frag:lx,equirect_vert:cx,equirect_frag:hx,linedashed_vert:ux,linedashed_frag:dx,meshbasic_vert:fx,meshbasic_frag:px,meshlambert_vert:mx,meshlambert_frag:gx,meshmatcap_vert:_x,meshmatcap_frag:xx,meshnormal_vert:vx,meshnormal_frag:yx,meshphong_vert:Mx,meshphong_frag:bx,meshphysical_vert:Sx,meshphysical_frag:Ex,meshtoon_vert:wx,meshtoon_frag:Tx,points_vert:Ax,points_frag:Rx,shadow_vert:Cx,shadow_frag:Px,sprite_vert:Ix,sprite_frag:Dx},ye={common:{diffuse:{value:new Le(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new je},alphaMap:{value:null},alphaMapTransform:{value:new je},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new je}},envmap:{envMap:{value:null},envMapRotation:{value:new je},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new je}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new je}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new je},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new je},normalScale:{value:new te(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new je},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new je}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new je}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new je}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Le(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:null},probesMin:{value:new A},probesMax:{value:new A},probesResolution:{value:new A}},points:{diffuse:{value:new Le(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new je},alphaTest:{value:0},uvTransform:{value:new je}},sprite:{diffuse:{value:new Le(16777215)},opacity:{value:1},center:{value:new te(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new je},alphaMap:{value:null},alphaMapTransform:{value:new je},alphaTest:{value:0}}},pi={basic:{uniforms:ci([ye.common,ye.specularmap,ye.envmap,ye.aomap,ye.lightmap,ye.fog]),vertexShader:at.meshbasic_vert,fragmentShader:at.meshbasic_frag},lambert:{uniforms:ci([ye.common,ye.specularmap,ye.envmap,ye.aomap,ye.lightmap,ye.emissivemap,ye.bumpmap,ye.normalmap,ye.displacementmap,ye.fog,ye.lights,{emissive:{value:new Le(0)},envMapIntensity:{value:1}}]),vertexShader:at.meshlambert_vert,fragmentShader:at.meshlambert_frag},phong:{uniforms:ci([ye.common,ye.specularmap,ye.envmap,ye.aomap,ye.lightmap,ye.emissivemap,ye.bumpmap,ye.normalmap,ye.displacementmap,ye.fog,ye.lights,{emissive:{value:new Le(0)},specular:{value:new Le(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:at.meshphong_vert,fragmentShader:at.meshphong_frag},standard:{uniforms:ci([ye.common,ye.envmap,ye.aomap,ye.lightmap,ye.emissivemap,ye.bumpmap,ye.normalmap,ye.displacementmap,ye.roughnessmap,ye.metalnessmap,ye.fog,ye.lights,{emissive:{value:new Le(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:at.meshphysical_vert,fragmentShader:at.meshphysical_frag},toon:{uniforms:ci([ye.common,ye.aomap,ye.lightmap,ye.emissivemap,ye.bumpmap,ye.normalmap,ye.displacementmap,ye.gradientmap,ye.fog,ye.lights,{emissive:{value:new Le(0)}}]),vertexShader:at.meshtoon_vert,fragmentShader:at.meshtoon_frag},matcap:{uniforms:ci([ye.common,ye.bumpmap,ye.normalmap,ye.displacementmap,ye.fog,{matcap:{value:null}}]),vertexShader:at.meshmatcap_vert,fragmentShader:at.meshmatcap_frag},points:{uniforms:ci([ye.points,ye.fog]),vertexShader:at.points_vert,fragmentShader:at.points_frag},dashed:{uniforms:ci([ye.common,ye.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:at.linedashed_vert,fragmentShader:at.linedashed_frag},depth:{uniforms:ci([ye.common,ye.displacementmap]),vertexShader:at.depth_vert,fragmentShader:at.depth_frag},normal:{uniforms:ci([ye.common,ye.bumpmap,ye.normalmap,ye.displacementmap,{opacity:{value:1}}]),vertexShader:at.meshnormal_vert,fragmentShader:at.meshnormal_frag},sprite:{uniforms:ci([ye.sprite,ye.fog]),vertexShader:at.sprite_vert,fragmentShader:at.sprite_frag},background:{uniforms:{uvTransform:{value:new je},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:at.background_vert,fragmentShader:at.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new je}},vertexShader:at.backgroundCube_vert,fragmentShader:at.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:at.cube_vert,fragmentShader:at.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:at.equirect_vert,fragmentShader:at.equirect_frag},distance:{uniforms:ci([ye.common,ye.displacementmap,{referencePosition:{value:new A},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:at.distance_vert,fragmentShader:at.distance_frag},shadow:{uniforms:ci([ye.lights,ye.fog,{color:{value:new Le(0)},opacity:{value:1}}]),vertexShader:at.shadow_vert,fragmentShader:at.shadow_frag}};pi.physical={uniforms:ci([pi.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new je},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new je},clearcoatNormalScale:{value:new te(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new je},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new je},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new je},sheen:{value:0},sheenColor:{value:new Le(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new je},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new je},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new je},transmissionSamplerSize:{value:new te},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new je},attenuationDistance:{value:0},attenuationColor:{value:new Le(0)},specularColor:{value:new Le(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new je},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new je},anisotropyVector:{value:new te},anisotropyMap:{value:null},anisotropyMapTransform:{value:new je}}]),vertexShader:at.meshphysical_vert,fragmentShader:at.meshphysical_frag};var Tc={r:0,b:0,g:0},Lx=new rt,mp=new je;mp.set(-1,0,0,0,1,0,0,0,1);function Nx(n,e,t,i,s,r){let a=new Le(0),o=s===!0?0:1,c,l,h=null,d=0,u=null;function f(M){let b=M.isScene===!0?M.background:null;if(b&&b.isTexture){let v=M.backgroundBlurriness>0;b=e.get(b,v)}return b}function g(M){let b=!1,v=f(M);v===null?p(a,o):v&&v.isColor&&(p(v,1),b=!0);let T=n.xr.getEnvironmentBlendMode();T==="additive"?t.buffers.color.setClear(0,0,0,1,r):T==="alpha-blend"&&t.buffers.color.setClear(0,0,0,0,r),(n.autoClear||b)&&(t.buffers.depth.setTest(!0),t.buffers.depth.setMask(!0),t.buffers.color.setMask(!0),n.clear(n.autoClearColor,n.autoClearDepth,n.autoClearStencil))}function x(M,b){let v=f(b);v&&(v.isCubeTexture||v.mapping===Xa)?(l===void 0&&(l=new tt(new Bt(1,1,1),new Rt({name:"BackgroundCubeMaterial",uniforms:Fs(pi.backgroundCube.uniforms),vertexShader:pi.backgroundCube.vertexShader,fragmentShader:pi.backgroundCube.fragmentShader,side:Qt,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),l.geometry.deleteAttribute("uv"),l.onBeforeRender=function(T,w,C){this.matrixWorld.copyPosition(C.matrixWorld)},Object.defineProperty(l.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(l)),l.material.uniforms.envMap.value=v,l.material.uniforms.backgroundBlurriness.value=b.backgroundBlurriness,l.material.uniforms.backgroundIntensity.value=b.backgroundIntensity,l.material.uniforms.backgroundRotation.value.setFromMatrix4(Lx.makeRotationFromEuler(b.backgroundRotation)).transpose(),v.isCubeTexture&&v.isRenderTargetTexture===!1&&l.material.uniforms.backgroundRotation.value.premultiply(mp),l.material.toneMapped=ht.getTransfer(v.colorSpace)!==ft,(h!==v||d!==v.version||u!==n.toneMapping)&&(l.material.needsUpdate=!0,h=v,d=v.version,u=n.toneMapping),l.layers.enableAll(),M.unshift(l,l.geometry,l.material,0,0,null)):v&&v.isTexture&&(c===void 0&&(c=new tt(new sn(2,2),new Rt({name:"BackgroundMaterial",uniforms:Fs(pi.background.uniforms),vertexShader:pi.background.vertexShader,fragmentShader:pi.background.fragmentShader,side:ji,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(c)),c.material.uniforms.t2D.value=v,c.material.uniforms.backgroundIntensity.value=b.backgroundIntensity,c.material.toneMapped=ht.getTransfer(v.colorSpace)!==ft,v.matrixAutoUpdate===!0&&v.updateMatrix(),c.material.uniforms.uvTransform.value.copy(v.matrix),(h!==v||d!==v.version||u!==n.toneMapping)&&(c.material.needsUpdate=!0,h=v,d=v.version,u=n.toneMapping),c.layers.enableAll(),M.unshift(c,c.geometry,c.material,0,0,null))}function p(M,b){M.getRGB(Tc,fu(n)),t.buffers.color.setClear(Tc.r,Tc.g,Tc.b,b,r)}function m(){l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(M,b=1){a.set(M),o=b,p(a,o)},getClearAlpha:function(){return o},setClearAlpha:function(M){o=M,p(a,o)},render:g,addToRenderList:x,dispose:m}}function Ux(n,e){let t=n.getParameter(n.MAX_VERTEX_ATTRIBS),i={},s=u(null),r=s,a=!1;function o(I,L,X,W,U){let z=!1,H=d(I,W,X,L);r!==H&&(r=H,l(r.object)),z=f(I,W,X,U),z&&g(I,W,X,U),U!==null&&e.update(U,n.ELEMENT_ARRAY_BUFFER),(z||a)&&(a=!1,v(I,L,X,W),U!==null&&n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,e.get(U).buffer))}function c(){return n.createVertexArray()}function l(I){return n.bindVertexArray(I)}function h(I){return n.deleteVertexArray(I)}function d(I,L,X,W){let U=W.wireframe===!0,z=i[L.id];z===void 0&&(z={},i[L.id]=z);let H=I.isInstancedMesh===!0?I.id:0,Q=z[H];Q===void 0&&(Q={},z[H]=Q);let ie=Q[X.id];ie===void 0&&(ie={},Q[X.id]=ie);let q=ie[U];return q===void 0&&(q=u(c()),ie[U]=q),q}function u(I){let L=[],X=[],W=[];for(let U=0;U<t;U++)L[U]=0,X[U]=0,W[U]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:L,enabledAttributes:X,attributeDivisors:W,object:I,attributes:{},index:null}}function f(I,L,X,W){let U=r.attributes,z=L.attributes,H=0,Q=X.getAttributes();for(let ie in Q)if(Q[ie].location>=0){let Z=U[ie],j=z[ie];if(j===void 0&&(ie==="instanceMatrix"&&I.instanceMatrix&&(j=I.instanceMatrix),ie==="instanceColor"&&I.instanceColor&&(j=I.instanceColor)),Z===void 0||Z.attribute!==j||j&&Z.data!==j.data)return!0;H++}return r.attributesNum!==H||r.index!==W}function g(I,L,X,W){let U={},z=L.attributes,H=0,Q=X.getAttributes();for(let ie in Q)if(Q[ie].location>=0){let Z=z[ie];Z===void 0&&(ie==="instanceMatrix"&&I.instanceMatrix&&(Z=I.instanceMatrix),ie==="instanceColor"&&I.instanceColor&&(Z=I.instanceColor));let j={};j.attribute=Z,Z&&Z.data&&(j.data=Z.data),U[ie]=j,H++}r.attributes=U,r.attributesNum=H,r.index=W}function x(){let I=r.newAttributes;for(let L=0,X=I.length;L<X;L++)I[L]=0}function p(I){m(I,0)}function m(I,L){let X=r.newAttributes,W=r.enabledAttributes,U=r.attributeDivisors;X[I]=1,W[I]===0&&(n.enableVertexAttribArray(I),W[I]=1),U[I]!==L&&(n.vertexAttribDivisor(I,L),U[I]=L)}function M(){let I=r.newAttributes,L=r.enabledAttributes;for(let X=0,W=L.length;X<W;X++)L[X]!==I[X]&&(n.disableVertexAttribArray(X),L[X]=0)}function b(I,L,X,W,U,z,H){H===!0?n.vertexAttribIPointer(I,L,X,U,z):n.vertexAttribPointer(I,L,X,W,U,z)}function v(I,L,X,W){x();let U=W.attributes,z=X.getAttributes(),H=L.defaultAttributeValues;for(let Q in z){let ie=z[Q];if(ie.location>=0){let q=U[Q];if(q===void 0&&(Q==="instanceMatrix"&&I.instanceMatrix&&(q=I.instanceMatrix),Q==="instanceColor"&&I.instanceColor&&(q=I.instanceColor)),q!==void 0){let Z=q.normalized,j=q.itemSize,de=e.get(q);if(de===void 0)continue;let Ge=de.buffer,me=de.type,k=de.bytesPerElement,ce=me===n.INT||me===n.UNSIGNED_INT||q.gpuType===Hl;if(q.isInterleavedBufferAttribute){let ae=q.data,Te=ae.stride,Ue=q.offset;if(ae.isInstancedInterleavedBuffer){for(let Oe=0;Oe<ie.locationSize;Oe++)m(ie.location+Oe,ae.meshPerAttribute);I.isInstancedMesh!==!0&&W._maxInstanceCount===void 0&&(W._maxInstanceCount=ae.meshPerAttribute*ae.count)}else for(let Oe=0;Oe<ie.locationSize;Oe++)p(ie.location+Oe);n.bindBuffer(n.ARRAY_BUFFER,Ge);for(let Oe=0;Oe<ie.locationSize;Oe++)b(ie.location+Oe,j/ie.locationSize,me,Z,Te*k,(Ue+j/ie.locationSize*Oe)*k,ce)}else{if(q.isInstancedBufferAttribute){for(let ae=0;ae<ie.locationSize;ae++)m(ie.location+ae,q.meshPerAttribute);I.isInstancedMesh!==!0&&W._maxInstanceCount===void 0&&(W._maxInstanceCount=q.meshPerAttribute*q.count)}else for(let ae=0;ae<ie.locationSize;ae++)p(ie.location+ae);n.bindBuffer(n.ARRAY_BUFFER,Ge);for(let ae=0;ae<ie.locationSize;ae++)b(ie.location+ae,j/ie.locationSize,me,Z,j*k,j/ie.locationSize*ae*k,ce)}}else if(H!==void 0){let Z=H[Q];if(Z!==void 0)switch(Z.length){case 2:n.vertexAttrib2fv(ie.location,Z);break;case 3:n.vertexAttrib3fv(ie.location,Z);break;case 4:n.vertexAttrib4fv(ie.location,Z);break;default:n.vertexAttrib1fv(ie.location,Z)}}}}M()}function T(){E();for(let I in i){let L=i[I];for(let X in L){let W=L[X];for(let U in W){let z=W[U];for(let H in z)h(z[H].object),delete z[H];delete W[U]}}delete i[I]}}function w(I){if(i[I.id]===void 0)return;let L=i[I.id];for(let X in L){let W=L[X];for(let U in W){let z=W[U];for(let H in z)h(z[H].object),delete z[H];delete W[U]}}delete i[I.id]}function C(I){for(let L in i){let X=i[L];for(let W in X){let U=X[W];if(U[I.id]===void 0)continue;let z=U[I.id];for(let H in z)h(z[H].object),delete z[H];delete U[I.id]}}}function _(I){for(let L in i){let X=i[L],W=I.isInstancedMesh===!0?I.id:0,U=X[W];if(U!==void 0){for(let z in U){let H=U[z];for(let Q in H)h(H[Q].object),delete H[Q];delete U[z]}delete X[W],Object.keys(X).length===0&&delete i[L]}}}function E(){P(),a=!0,r!==s&&(r=s,l(r.object))}function P(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:E,resetDefaultState:P,dispose:T,releaseStatesOfGeometry:w,releaseStatesOfObject:_,releaseStatesOfProgram:C,initAttributes:x,enableAttribute:p,disableUnusedAttributes:M}}function Fx(n,e,t){let i;function s(c){i=c}function r(c,l){n.drawArrays(i,c,l),t.update(l,i,1)}function a(c,l,h){h!==0&&(n.drawArraysInstanced(i,c,l,h),t.update(l,i,h))}function o(c,l,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,c,0,l,0,h);let u=0;for(let f=0;f<h;f++)u+=l[f];t.update(u,i,1)}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o}function Ox(n,e,t,i){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){let C=e.get("EXT_texture_filter_anisotropic");s=n.getParameter(C.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(C){return!(C!==vi&&i.convert(C)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(C){let _=C===ei&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(C!==li&&i.convert(C)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_TYPE)&&C!==Vi&&!_)}function c(C){if(C==="highp"){if(n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.HIGH_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.HIGH_FLOAT).precision>0)return"highp";C="mediump"}return C==="mediump"&&n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.MEDIUM_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=t.precision!==void 0?t.precision:"highp",h=c(l);h!==l&&(Ye("WebGLRenderer:",l,"not supported, using",h,"instead."),l=h);let d=t.logarithmicDepthBuffer===!0,u=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control");t.reversedDepthBuffer===!0&&u===!1&&Ye("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.");let f=n.getParameter(n.MAX_TEXTURE_IMAGE_UNITS),g=n.getParameter(n.MAX_VERTEX_TEXTURE_IMAGE_UNITS),x=n.getParameter(n.MAX_TEXTURE_SIZE),p=n.getParameter(n.MAX_CUBE_MAP_TEXTURE_SIZE),m=n.getParameter(n.MAX_VERTEX_ATTRIBS),M=n.getParameter(n.MAX_VERTEX_UNIFORM_VECTORS),b=n.getParameter(n.MAX_VARYING_VECTORS),v=n.getParameter(n.MAX_FRAGMENT_UNIFORM_VECTORS),T=n.getParameter(n.MAX_SAMPLES),w=n.getParameter(n.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:c,textureFormatReadable:a,textureTypeReadable:o,precision:l,logarithmicDepthBuffer:d,reversedDepthBuffer:u,maxTextures:f,maxVertexTextures:g,maxTextureSize:x,maxCubemapSize:p,maxAttributes:m,maxVertexUniforms:M,maxVaryings:b,maxFragmentUniforms:v,maxSamples:T,samples:w}}function Bx(n){let e=this,t=null,i=0,s=!1,r=!1,a=new Bi,o=new je,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(d,u){let f=d.length!==0||u||i!==0||s;return s=u,i=d.length,f},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(d,u){t=h(d,u,0)},this.setState=function(d,u,f){let g=d.clippingPlanes,x=d.clipIntersection,p=d.clipShadows,m=n.get(d);if(!s||g===null||g.length===0||r&&!p)r?h(null):l();else{let M=r?0:i,b=M*4,v=m.clippingState||null;c.value=v,v=h(g,u,b,f);for(let T=0;T!==b;++T)v[T]=t[T];m.clippingState=v,this.numIntersection=x?this.numPlanes:0,this.numPlanes+=M}};function l(){c.value!==t&&(c.value=t,c.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function h(d,u,f,g){let x=d!==null?d.length:0,p=null;if(x!==0){if(p=c.value,g!==!0||p===null){let m=f+x*4,M=u.matrixWorldInverse;o.getNormalMatrix(M),(p===null||p.length<m)&&(p=new Float32Array(m));for(let b=0,v=f;b!==x;++b,v+=4)a.copy(d[b]).applyMatrix4(M,o),a.normal.toArray(p,v),p[v+3]=a.constant}c.value=p,c.needsUpdate=!0}return e.numPlanes=x,e.numIntersection=0,p}}var ds=4,qf=[.125,.215,.35,.446,.526,.582],Os=20,zx=256,Qa=new ns,Yf=new Le,bu=null,Su=0,Eu=0,wu=!1,kx=new A,Lr=class{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,i=.1,s=100,r={}){let{size:a=256,position:o=kx}=r;bu=this._renderer.getRenderTarget(),Su=this._renderer.getActiveCubeFace(),Eu=this._renderer.getActiveMipmapLevel(),wu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(a);let c=this._allocateTargets();return c.depthBuffer=!0,this._sceneToCubeUV(e,i,s,c,o),t>0&&this._blur(c,0,0,t),this._applyPMREM(c),this._cleanup(c),c}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Jf(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Zf(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(bu,Su,Eu),this._renderer.xr.enabled=wu,e.scissorTest=!1,Ir(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===ls||e.mapping===Us?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),bu=this._renderer.getRenderTarget(),Su=this._renderer.getActiveCubeFace(),Eu=this._renderer.getActiveMipmapLevel(),wu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;let i=t||this._allocateTargets();return this._textureToCubeUV(e,i),this._applyPMREM(i),this._cleanup(i),i}_allocateTargets(){let e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,i={magFilter:jt,minFilter:jt,generateMipmaps:!1,type:ei,format:vi,colorSpace:ia,depthBuffer:!1},s=$f(e,t,i);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=$f(e,t,i);let{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=Hx(r)),this._blurMaterial=Gx(r,e,t),this._ggxMaterial=Vx(r,e,t)}return s}_compileMaterial(e){let t=new tt(new mt,e);this._renderer.compile(t,Qa)}_sceneToCubeUV(e,t,i,s,r){let c=new Kt(90,1,t,i),l=[1,-1,1,1,1,1],h=[1,1,1,-1,-1,-1],d=this._renderer,u=d.autoClear,f=d.toneMapping;d.getClearColor(Yf),d.toneMapping=rn,d.autoClear=!1,d.state.buffers.depth.getReversed()&&(d.setRenderTarget(s),d.clearDepth(),d.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new tt(new Bt,new In({name:"PMREM.Background",side:Qt,depthWrite:!1,depthTest:!1})));let x=this._backgroundBox,p=x.material,m=!1,M=e.background;M?M.isColor&&(p.color.copy(M),e.background=null,m=!0):(p.color.copy(Yf),m=!0);for(let b=0;b<6;b++){let v=b%3;v===0?(c.up.set(0,l[b],0),c.position.set(r.x,r.y,r.z),c.lookAt(r.x+h[b],r.y,r.z)):v===1?(c.up.set(0,0,l[b]),c.position.set(r.x,r.y,r.z),c.lookAt(r.x,r.y+h[b],r.z)):(c.up.set(0,l[b],0),c.position.set(r.x,r.y,r.z),c.lookAt(r.x,r.y,r.z+h[b]));let T=this._cubeSize;Ir(s,v*T,b>2?T:0,T,T),d.setRenderTarget(s),m&&d.render(x,c),d.render(e,c)}d.toneMapping=f,d.autoClear=u,e.background=M}_textureToCubeUV(e,t){let i=this._renderer,s=e.mapping===ls||e.mapping===Us;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=Jf()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Zf());let r=s?this._cubemapMaterial:this._equirectMaterial,a=this._lodMeshes[0];a.material=r;let o=r.uniforms;o.envMap.value=e;let c=this._cubeSize;Ir(t,0,0,3*c,2*c),i.setRenderTarget(t),i.render(a,Qa)}_applyPMREM(e){let t=this._renderer,i=t.autoClear;t.autoClear=!1;let s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=i}_applyGGXFilter(e,t,i){let s=this._renderer,r=this._pingPongRenderTarget,a=this._ggxMaterial,o=this._lodMeshes[i];o.material=a;let c=a.uniforms,l=i/(this._lodMeshes.length-1),h=t/(this._lodMeshes.length-1),d=Math.sqrt(l*l-h*h),u=0+l*1.25,f=d*u,{_lodMax:g}=this,x=this._sizeLods[i],p=3*x*(i>g-ds?i-g+ds:0),m=4*(this._cubeSize-x);c.envMap.value=e.texture,c.roughness.value=f,c.mipInt.value=g-t,Ir(r,p,m,3*x,2*x),s.setRenderTarget(r),s.render(o,Qa),c.envMap.value=r.texture,c.roughness.value=0,c.mipInt.value=g-i,Ir(e,p,m,3*x,2*x),s.setRenderTarget(e),s.render(o,Qa)}_blur(e,t,i,s,r){let a=this._pingPongRenderTarget;this._halfBlur(e,a,t,i,s,"latitudinal",r),this._halfBlur(a,e,i,i,s,"longitudinal",r)}_halfBlur(e,t,i,s,r,a,o){let c=this._renderer,l=this._blurMaterial;a!=="latitudinal"&&a!=="longitudinal"&&$e("blur direction must be either latitudinal or longitudinal!");let h=3,d=this._lodMeshes[s];d.material=l;let u=l.uniforms,f=this._sizeLods[i]-1,g=isFinite(r)?Math.PI/(2*f):2*Math.PI/(2*Os-1),x=r/g,p=isFinite(r)?1+Math.floor(h*x):Os;p>Os&&Ye(`sigmaRadians, ${r}, is too large and will clip, as it requested ${p} samples when the maximum is set to ${Os}`);let m=[],M=0;for(let C=0;C<Os;++C){let _=C/x,E=Math.exp(-_*_/2);m.push(E),C===0?M+=E:C<p&&(M+=2*E)}for(let C=0;C<m.length;C++)m[C]=m[C]/M;u.envMap.value=e.texture,u.samples.value=p,u.weights.value=m,u.latitudinal.value=a==="latitudinal",o&&(u.poleAxis.value=o);let{_lodMax:b}=this;u.dTheta.value=g,u.mipInt.value=b-i;let v=this._sizeLods[s],T=3*v*(s>b-ds?s-b+ds:0),w=4*(this._cubeSize-v);Ir(t,T,w,3*v,2*v),c.setRenderTarget(t),c.render(d,Qa)}};function Hx(n){let e=[],t=[],i=[],s=n,r=n-ds+1+qf.length;for(let a=0;a<r;a++){let o=Math.pow(2,s);e.push(o);let c=1/o;a>n-ds?c=qf[a-n+ds-1]:a===0&&(c=0),t.push(c);let l=1/(o-2),h=-l,d=1+l,u=[h,h,d,h,d,d,h,h,d,d,h,d],f=6,g=6,x=3,p=2,m=1,M=new Float32Array(x*g*f),b=new Float32Array(p*g*f),v=new Float32Array(m*g*f);for(let w=0;w<f;w++){let C=w%3*2/3-1,_=w>2?0:-1,E=[C,_,0,C+2/3,_,0,C+2/3,_+1,0,C,_,0,C+2/3,_+1,0,C,_+1,0];M.set(E,x*g*w),b.set(u,p*g*w);let P=[w,w,w,w,w,w];v.set(P,m*g*w)}let T=new mt;T.setAttribute("position",new Yt(M,x)),T.setAttribute("uv",new Yt(b,p)),T.setAttribute("faceIndex",new Yt(v,m)),i.push(new tt(T,null)),s>ds&&s--}return{lodMeshes:i,sizeLods:e,sigmas:t}}function $f(n,e,t){let i=new Ht(n,e,t);return i.texture.mapping=Xa,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function Ir(n,e,t,i,s){n.viewport.set(e,t,i,s),n.scissor.set(e,t,i,s)}function Vx(n,e,t){return new Rt({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:zx,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${n}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Pc(),fragmentShader:`

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
		`,blending:zt,depthTest:!1,depthWrite:!1})}function Gx(n,e,t){let i=new Float32Array(Os),s=new A(0,1,0);return new Rt({name:"SphericalGaussianBlur",defines:{n:Os,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${n}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:i},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Pc(),fragmentShader:`

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
		`,blending:zt,depthTest:!1,depthWrite:!1})}function Zf(){return new Rt({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Pc(),fragmentShader:`

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
		`,blending:zt,depthTest:!1,depthWrite:!1})}function Jf(){return new Rt({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Pc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:zt,depthTest:!1,depthWrite:!1})}function Pc(){return`

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
	`}var Rc=class extends Ht{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;let i={width:e,height:e,depth:1},s=[i,i,i,i,i,i];this.texture=new fa(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;let i={uniforms:{tEquirect:{value:null}},vertexShader:`

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
			`},s=new Bt(5,5,5),r=new Rt({name:"CubemapFromEquirect",uniforms:Fs(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:Qt,blending:zt});r.uniforms.tEquirect.value=t;let a=new tt(s,r),o=t.minFilter;return t.minFilter===cs&&(t.minFilter=jt),new Ll(1,10,this).update(e,a),t.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,t=!0,i=!0,s=!0){let r=e.getRenderTarget();for(let a=0;a<6;a++)e.setRenderTarget(this,a),e.clear(t,i,s);e.setRenderTarget(r)}};function Wx(n){let e=new WeakMap,t=new WeakMap,i=null;function s(u,f=!1){return u==null?null:f?a(u):r(u)}function r(u){if(u&&u.isTexture){let f=u.mapping;if(f===Bl||f===zl)if(e.has(u)){let g=e.get(u).texture;return o(g,u.mapping)}else{let g=u.image;if(g&&g.height>0){let x=new Rc(g.height);return x.fromEquirectangularTexture(n,u),e.set(u,x),u.addEventListener("dispose",l),o(x.texture,u.mapping)}else return null}}return u}function a(u){if(u&&u.isTexture){let f=u.mapping,g=f===Bl||f===zl,x=f===ls||f===Us;if(g||x){let p=t.get(u),m=p!==void 0?p.texture.pmremVersion:0;if(u.isRenderTargetTexture&&u.pmremVersion!==m)return i===null&&(i=new Lr(n)),p=g?i.fromEquirectangular(u,p):i.fromCubemap(u,p),p.texture.pmremVersion=u.pmremVersion,t.set(u,p),p.texture;if(p!==void 0)return p.texture;{let M=u.image;return g&&M&&M.height>0||x&&M&&c(M)?(i===null&&(i=new Lr(n)),p=g?i.fromEquirectangular(u):i.fromCubemap(u),p.texture.pmremVersion=u.pmremVersion,t.set(u,p),u.addEventListener("dispose",h),p.texture):null}}}return u}function o(u,f){return f===Bl?u.mapping=ls:f===zl&&(u.mapping=Us),u}function c(u){let f=0,g=6;for(let x=0;x<g;x++)u[x]!==void 0&&f++;return f===g}function l(u){let f=u.target;f.removeEventListener("dispose",l);let g=e.get(f);g!==void 0&&(e.delete(f),g.dispose())}function h(u){let f=u.target;f.removeEventListener("dispose",h);let g=t.get(f);g!==void 0&&(t.delete(f),g.dispose())}function d(){e=new WeakMap,t=new WeakMap,i!==null&&(i.dispose(),i=null)}return{get:s,dispose:d}}function Xx(n){let e={};function t(i){if(e[i]!==void 0)return e[i];let s=n.getExtension(i);return e[i]=s,s}return{has:function(i){return t(i)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(i){let s=t(i);return s===null&&Ts("WebGLRenderer: "+i+" extension not supported."),s}}}function qx(n,e,t,i){let s={},r=new WeakMap;function a(d){let u=d.target;u.index!==null&&e.remove(u.index);for(let g in u.attributes)e.remove(u.attributes[g]);u.removeEventListener("dispose",a),delete s[u.id];let f=r.get(u);f&&(e.remove(f),r.delete(u)),i.releaseStatesOfGeometry(u),u.isInstancedBufferGeometry===!0&&delete u._maxInstanceCount,t.memory.geometries--}function o(d,u){return s[u.id]===!0||(u.addEventListener("dispose",a),s[u.id]=!0,t.memory.geometries++),u}function c(d){let u=d.attributes;for(let f in u)e.update(u[f],n.ARRAY_BUFFER)}function l(d){let u=[],f=d.index,g=d.attributes.position,x=0;if(g===void 0)return;if(f!==null){let M=f.array;x=f.version;for(let b=0,v=M.length;b<v;b+=3){let T=M[b+0],w=M[b+1],C=M[b+2];u.push(T,w,w,C,C,T)}}else{let M=g.array;x=g.version;for(let b=0,v=M.length/3-1;b<v;b+=3){let T=b+0,w=b+1,C=b+2;u.push(T,w,w,C,C,T)}}let p=new(g.count>=65535?ca:la)(u,1);p.version=x;let m=r.get(d);m&&e.remove(m),r.set(d,p)}function h(d){let u=r.get(d);if(u){let f=d.index;f!==null&&u.version<f.version&&l(d)}else l(d);return r.get(d)}return{get:o,update:c,getWireframeAttribute:h}}function Yx(n,e,t){let i;function s(d){i=d}let r,a;function o(d){r=d.type,a=d.bytesPerElement}function c(d,u){n.drawElements(i,u,r,d*a),t.update(u,i,1)}function l(d,u,f){f!==0&&(n.drawElementsInstanced(i,u,r,d*a,f),t.update(u,i,f))}function h(d,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,u,0,r,d,0,f);let x=0;for(let p=0;p<f;p++)x+=u[p];t.update(x,i,1)}this.setMode=s,this.setIndex=o,this.render=c,this.renderInstances=l,this.renderMultiDraw=h}function $x(n){let e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function i(r,a,o){switch(t.calls++,a){case n.TRIANGLES:t.triangles+=o*(r/3);break;case n.LINES:t.lines+=o*(r/2);break;case n.LINE_STRIP:t.lines+=o*(r-1);break;case n.LINE_LOOP:t.lines+=o*r;break;case n.POINTS:t.points+=o*r;break;default:$e("WebGLInfo: Unknown draw mode:",a);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:i}}function Zx(n,e,t){let i=new WeakMap,s=new gt;function r(a,o,c){let l=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,d=h!==void 0?h.length:0,u=i.get(o);if(u===void 0||u.count!==d){let E=function(){C.dispose(),i.delete(o),o.removeEventListener("dispose",E)};u!==void 0&&u.texture.dispose();let f=o.morphAttributes.position!==void 0,g=o.morphAttributes.normal!==void 0,x=o.morphAttributes.color!==void 0,p=o.morphAttributes.position||[],m=o.morphAttributes.normal||[],M=o.morphAttributes.color||[],b=0;f===!0&&(b=1),g===!0&&(b=2),x===!0&&(b=3);let v=o.attributes.position.count*b,T=1;v>e.maxTextureSize&&(T=Math.ceil(v/e.maxTextureSize),v=e.maxTextureSize);let w=new Float32Array(v*T*4*d),C=new aa(w,v,T,d);C.type=Vi,C.needsUpdate=!0;let _=b*4;for(let P=0;P<d;P++){let I=p[P],L=m[P],X=M[P],W=v*T*4*P;for(let U=0;U<I.count;U++){let z=U*_;f===!0&&(s.fromBufferAttribute(I,U),w[W+z+0]=s.x,w[W+z+1]=s.y,w[W+z+2]=s.z,w[W+z+3]=0),g===!0&&(s.fromBufferAttribute(L,U),w[W+z+4]=s.x,w[W+z+5]=s.y,w[W+z+6]=s.z,w[W+z+7]=0),x===!0&&(s.fromBufferAttribute(X,U),w[W+z+8]=s.x,w[W+z+9]=s.y,w[W+z+10]=s.z,w[W+z+11]=X.itemSize===4?s.w:1)}}u={count:d,texture:C,size:new te(v,T)},i.set(o,u),o.addEventListener("dispose",E)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)c.getUniforms().setValue(n,"morphTexture",a.morphTexture,t);else{let f=0;for(let x=0;x<l.length;x++)f+=l[x];let g=o.morphTargetsRelative?1:1-f;c.getUniforms().setValue(n,"morphTargetBaseInfluence",g),c.getUniforms().setValue(n,"morphTargetInfluences",l)}c.getUniforms().setValue(n,"morphTargetsTexture",u.texture,t),c.getUniforms().setValue(n,"morphTargetsTextureSize",u.size)}return{update:r}}function Jx(n,e,t,i,s){let r=new WeakMap;function a(l){let h=s.render.frame,d=l.geometry,u=e.get(l,d);if(r.get(u)!==h&&(e.update(u),r.set(u,h)),l.isInstancedMesh&&(l.hasEventListener("dispose",c)===!1&&l.addEventListener("dispose",c),r.get(l)!==h&&(t.update(l.instanceMatrix,n.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,n.ARRAY_BUFFER),r.set(l,h))),l.isSkinnedMesh){let f=l.skeleton;r.get(f)!==h&&(f.update(),r.set(f,h))}return u}function o(){r=new WeakMap}function c(l){let h=l.target;h.removeEventListener("dispose",c),i.releaseStatesOfObject(h),t.remove(h.instanceMatrix),h.instanceColor!==null&&t.remove(h.instanceColor)}return{update:a,dispose:o}}var Kx={[ka]:"LINEAR_TONE_MAPPING",[Ha]:"REINHARD_TONE_MAPPING",[Va]:"CINEON_TONE_MAPPING",[os]:"ACES_FILMIC_TONE_MAPPING",[Wa]:"AGX_TONE_MAPPING",[Ns]:"NEUTRAL_TONE_MAPPING",[Ga]:"CUSTOM_TONE_MAPPING"};function jx(n,e,t,i,s,r){let a=new Ht(e,t,{type:n,depthBuffer:s,stencilBuffer:r,samples:i?4:0,depthTexture:s?new en(e,t):void 0}),o=new Ht(e,t,{type:ei,depthBuffer:!1,stencilBuffer:!1}),c=new mt;c.setAttribute("position",new nt([-1,3,0,-1,-1,0,3,-1,0],3)),c.setAttribute("uv",new nt([0,2,0,0,2,0],2));let l=new Sr({uniforms:{tDiffuse:{value:null}},vertexShader:`
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
			}`,depthTest:!1,depthWrite:!1}),h=new tt(c,l),d=new ns(-1,1,1,-1,0,1),u=null,f=null,g=!1,x,p=null,m=[],M=!1;this.setSize=function(b,v){a.setSize(b,v),o.setSize(b,v);for(let T=0;T<m.length;T++){let w=m[T];w.setSize&&w.setSize(b,v)}},this.setEffects=function(b){m=b,M=m.length>0&&m[0].isRenderPass===!0;let v=a.width,T=a.height;for(let w=0;w<m.length;w++){let C=m[w];C.setSize&&C.setSize(v,T)}},this.begin=function(b,v){if(g||b.toneMapping===rn&&m.length===0)return!1;if(p=v,v!==null){let T=v.width,w=v.height;(a.width!==T||a.height!==w)&&this.setSize(T,w)}return M===!1&&b.setRenderTarget(a),x=b.toneMapping,b.toneMapping=rn,!0},this.hasRenderPass=function(){return M},this.end=function(b,v){b.toneMapping=x,g=!0;let T=a,w=o;for(let C=0;C<m.length;C++){let _=m[C];if(_.enabled!==!1&&(_.render(b,w,T,v),_.needsSwap!==!1)){let E=T;T=w,w=E}}if(u!==b.outputColorSpace||f!==b.toneMapping){u=b.outputColorSpace,f=b.toneMapping,l.defines={},ht.getTransfer(u)===ft&&(l.defines.SRGB_TRANSFER="");let C=Kx[f];C&&(l.defines[C]=""),l.needsUpdate=!0}l.uniforms.tDiffuse.value=T.texture,b.setRenderTarget(p),b.render(h,d),p=null,g=!1},this.isCompositing=function(){return g},this.dispose=function(){a.depthTexture&&a.depthTexture.dispose(),a.dispose(),o.dispose(),c.dispose(),l.dispose()}}var gp=new ui,Ru=new en(1,1),_p=new aa,xp=new cl,vp=new fa,Kf=[],jf=[],Qf=new Float32Array(16),ep=new Float32Array(9),tp=new Float32Array(4);function Nr(n,e,t){let i=n[0];if(i<=0||i>0)return n;let s=e*t,r=Kf[s];if(r===void 0&&(r=new Float32Array(s),Kf[s]=r),e!==0){i.toArray(r,0);for(let a=1,o=0;a!==e;++a)o+=t,n[a].toArray(r,o)}return r}function Gt(n,e){if(n.length!==e.length)return!1;for(let t=0,i=n.length;t<i;t++)if(n[t]!==e[t])return!1;return!0}function Wt(n,e){for(let t=0,i=e.length;t<i;t++)n[t]=e[t]}function Ic(n,e){let t=jf[e];t===void 0&&(t=new Int32Array(e),jf[e]=t);for(let i=0;i!==e;++i)t[i]=n.allocateTextureUnit();return t}function Qx(n,e){let t=this.cache;t[0]!==e&&(n.uniform1f(this.addr,e),t[0]=e)}function ev(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Gt(t,e))return;n.uniform2fv(this.addr,e),Wt(t,e)}}function tv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(n.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Gt(t,e))return;n.uniform3fv(this.addr,e),Wt(t,e)}}function iv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Gt(t,e))return;n.uniform4fv(this.addr,e),Wt(t,e)}}function nv(n,e){let t=this.cache,i=e.elements;if(i===void 0){if(Gt(t,e))return;n.uniformMatrix2fv(this.addr,!1,e),Wt(t,e)}else{if(Gt(t,i))return;tp.set(i),n.uniformMatrix2fv(this.addr,!1,tp),Wt(t,i)}}function sv(n,e){let t=this.cache,i=e.elements;if(i===void 0){if(Gt(t,e))return;n.uniformMatrix3fv(this.addr,!1,e),Wt(t,e)}else{if(Gt(t,i))return;ep.set(i),n.uniformMatrix3fv(this.addr,!1,ep),Wt(t,i)}}function rv(n,e){let t=this.cache,i=e.elements;if(i===void 0){if(Gt(t,e))return;n.uniformMatrix4fv(this.addr,!1,e),Wt(t,e)}else{if(Gt(t,i))return;Qf.set(i),n.uniformMatrix4fv(this.addr,!1,Qf),Wt(t,i)}}function av(n,e){let t=this.cache;t[0]!==e&&(n.uniform1i(this.addr,e),t[0]=e)}function ov(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Gt(t,e))return;n.uniform2iv(this.addr,e),Wt(t,e)}}function lv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Gt(t,e))return;n.uniform3iv(this.addr,e),Wt(t,e)}}function cv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Gt(t,e))return;n.uniform4iv(this.addr,e),Wt(t,e)}}function hv(n,e){let t=this.cache;t[0]!==e&&(n.uniform1ui(this.addr,e),t[0]=e)}function uv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Gt(t,e))return;n.uniform2uiv(this.addr,e),Wt(t,e)}}function dv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Gt(t,e))return;n.uniform3uiv(this.addr,e),Wt(t,e)}}function fv(n,e){let t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Gt(t,e))return;n.uniform4uiv(this.addr,e),Wt(t,e)}}function pv(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s);let r;this.type===n.SAMPLER_2D_SHADOW?(Ru.compareFunction=t.isReversedDepthBuffer()?wc:Ec,r=Ru):r=gp,t.setTexture2D(e||r,s)}function mv(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTexture3D(e||xp,s)}function gv(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTextureCube(e||vp,s)}function _v(n,e,t){let i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTexture2DArray(e||_p,s)}function xv(n){switch(n){case 5126:return Qx;case 35664:return ev;case 35665:return tv;case 35666:return iv;case 35674:return nv;case 35675:return sv;case 35676:return rv;case 5124:case 35670:return av;case 35667:case 35671:return ov;case 35668:case 35672:return lv;case 35669:case 35673:return cv;case 5125:return hv;case 36294:return uv;case 36295:return dv;case 36296:return fv;case 35678:case 36198:case 36298:case 36306:case 35682:return pv;case 35679:case 36299:case 36307:return mv;case 35680:case 36300:case 36308:case 36293:return gv;case 36289:case 36303:case 36311:case 36292:return _v}}function vv(n,e){n.uniform1fv(this.addr,e)}function yv(n,e){let t=Nr(e,this.size,2);n.uniform2fv(this.addr,t)}function Mv(n,e){let t=Nr(e,this.size,3);n.uniform3fv(this.addr,t)}function bv(n,e){let t=Nr(e,this.size,4);n.uniform4fv(this.addr,t)}function Sv(n,e){let t=Nr(e,this.size,4);n.uniformMatrix2fv(this.addr,!1,t)}function Ev(n,e){let t=Nr(e,this.size,9);n.uniformMatrix3fv(this.addr,!1,t)}function wv(n,e){let t=Nr(e,this.size,16);n.uniformMatrix4fv(this.addr,!1,t)}function Tv(n,e){n.uniform1iv(this.addr,e)}function Av(n,e){n.uniform2iv(this.addr,e)}function Rv(n,e){n.uniform3iv(this.addr,e)}function Cv(n,e){n.uniform4iv(this.addr,e)}function Pv(n,e){n.uniform1uiv(this.addr,e)}function Iv(n,e){n.uniform2uiv(this.addr,e)}function Dv(n,e){n.uniform3uiv(this.addr,e)}function Lv(n,e){n.uniform4uiv(this.addr,e)}function Nv(n,e,t){let i=this.cache,s=e.length,r=Ic(t,s);Gt(i,r)||(n.uniform1iv(this.addr,r),Wt(i,r));let a;this.type===n.SAMPLER_2D_SHADOW?a=Ru:a=gp;for(let o=0;o!==s;++o)t.setTexture2D(e[o]||a,r[o])}function Uv(n,e,t){let i=this.cache,s=e.length,r=Ic(t,s);Gt(i,r)||(n.uniform1iv(this.addr,r),Wt(i,r));for(let a=0;a!==s;++a)t.setTexture3D(e[a]||xp,r[a])}function Fv(n,e,t){let i=this.cache,s=e.length,r=Ic(t,s);Gt(i,r)||(n.uniform1iv(this.addr,r),Wt(i,r));for(let a=0;a!==s;++a)t.setTextureCube(e[a]||vp,r[a])}function Ov(n,e,t){let i=this.cache,s=e.length,r=Ic(t,s);Gt(i,r)||(n.uniform1iv(this.addr,r),Wt(i,r));for(let a=0;a!==s;++a)t.setTexture2DArray(e[a]||_p,r[a])}function Bv(n){switch(n){case 5126:return vv;case 35664:return yv;case 35665:return Mv;case 35666:return bv;case 35674:return Sv;case 35675:return Ev;case 35676:return wv;case 5124:case 35670:return Tv;case 35667:case 35671:return Av;case 35668:case 35672:return Rv;case 35669:case 35673:return Cv;case 5125:return Pv;case 36294:return Iv;case 36295:return Dv;case 36296:return Lv;case 35678:case 36198:case 36298:case 36306:case 35682:return Nv;case 35679:case 36299:case 36307:return Uv;case 35680:case 36300:case 36308:case 36293:return Fv;case 36289:case 36303:case 36311:case 36292:return Ov}}var Cu=class{constructor(e,t,i){this.id=e,this.addr=i,this.cache=[],this.type=t.type,this.setValue=xv(t.type)}},Pu=class{constructor(e,t,i){this.id=e,this.addr=i,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=Bv(t.type)}},Iu=class{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,i){let s=this.seq;for(let r=0,a=s.length;r!==a;++r){let o=s[r];o.setValue(e,t[o.id],i)}}},Tu=/(\w+)(\])?(\[|\.)?/g;function ip(n,e){n.seq.push(e),n.map[e.id]=e}function zv(n,e,t){let i=n.name,s=i.length;for(Tu.lastIndex=0;;){let r=Tu.exec(i),a=Tu.lastIndex,o=r[1],c=r[2]==="]",l=r[3];if(c&&(o=o|0),l===void 0||l==="["&&a+2===s){ip(t,l===void 0?new Cu(o,n,e):new Pu(o,n,e));break}else{let d=t.map[o];d===void 0&&(d=new Iu(o),ip(t,d)),t=d}}}var Dr=class{constructor(e,t){this.seq=[],this.map={};let i=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let a=0;a<i;++a){let o=e.getActiveUniform(t,a),c=e.getUniformLocation(t,o.name);zv(o,c,this)}let s=[],r=[];for(let a of this.seq)a.type===e.SAMPLER_2D_SHADOW||a.type===e.SAMPLER_CUBE_SHADOW||a.type===e.SAMPLER_2D_ARRAY_SHADOW?s.push(a):r.push(a);s.length>0&&(this.seq=s.concat(r))}setValue(e,t,i,s){let r=this.map[t];r!==void 0&&r.setValue(e,i,s)}setOptional(e,t,i){let s=t[i];s!==void 0&&this.setValue(e,i,s)}static upload(e,t,i,s){for(let r=0,a=t.length;r!==a;++r){let o=t[r],c=i[o.id];c.needsUpdate!==!1&&o.setValue(e,c.value,s)}}static seqWithValue(e,t){let i=[];for(let s=0,r=e.length;s!==r;++s){let a=e[s];a.id in t&&i.push(a)}return i}};function np(n,e,t){let i=n.createShader(e);return n.shaderSource(i,t),n.compileShader(i),i}var kv=37297,Hv=0;function Vv(n,e){let t=n.split(`
`),i=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let a=s;a<r;a++){let o=a+1;i.push(`${o===e?">":" "} ${o}: ${t[a]}`)}return i.join(`
`)}var sp=new je;function Gv(n){ht._getMatrix(sp,ht.workingColorSpace,n);let e=`mat3( ${sp.elements.map(t=>t.toFixed(4))} )`;switch(ht.getTransfer(n)){case na:return[e,"LinearTransferOETF"];case ft:return[e,"sRGBTransferOETF"];default:return Ye("WebGLProgram: Unsupported color space: ",n),[e,"LinearTransferOETF"]}}function rp(n,e,t){let i=n.getShaderParameter(e,n.COMPILE_STATUS),r=(n.getShaderInfoLog(e)||"").trim();if(i&&r==="")return"";let a=/ERROR: 0:(\d+)/.exec(r);if(a){let o=parseInt(a[1]);return t.toUpperCase()+`

`+r+`

`+Vv(n.getShaderSource(e),o)}else return r}function Wv(n,e){let t=Gv(e);return[`vec4 ${n}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}var Xv={[ka]:"Linear",[Ha]:"Reinhard",[Va]:"Cineon",[os]:"ACESFilmic",[Wa]:"AgX",[Ns]:"Neutral",[Ga]:"Custom"};function qv(n,e){let t=Xv[e];return t===void 0?(Ye("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+n+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+n+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}var Ac=new A;function Yv(){ht.getLuminanceCoefficients(Ac);let n=Ac.x.toFixed(4),e=Ac.y.toFixed(4),t=Ac.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${n}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function $v(n){return[n.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",n.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(to).join(`
`)}function Zv(n){let e=[];for(let t in n){let i=n[t];i!==!1&&e.push("#define "+t+" "+i)}return e.join(`
`)}function Jv(n,e){let t={},i=n.getProgramParameter(e,n.ACTIVE_ATTRIBUTES);for(let s=0;s<i;s++){let r=n.getActiveAttrib(e,s),a=r.name,o=1;r.type===n.FLOAT_MAT2&&(o=2),r.type===n.FLOAT_MAT3&&(o=3),r.type===n.FLOAT_MAT4&&(o=4),t[a]={type:r.type,location:n.getAttribLocation(e,a),locationSize:o}}return t}function to(n){return n!==""}function ap(n,e){let t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return n.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function op(n,e){return n.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}var Kv=/^[ \t]*#include +<([\w\d./]+)>/gm;function Du(n){return n.replace(Kv,Qv)}var jv=new Map;function Qv(n,e){let t=at[e];if(t===void 0){let i=jv.get(e);if(i!==void 0)t=at[i],Ye('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,i);else throw new Error("THREE.WebGLProgram: Can not resolve #include <"+e+">")}return Du(t)}var ey=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function lp(n){return n.replace(ey,ty)}function ty(n,e,t,i){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=i.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function cp(n){let e=`precision ${n.precision} float;
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
#define LOW_PRECISION`),e}var iy={[Ds]:"SHADOWMAP_TYPE_PCF",[Ar]:"SHADOWMAP_TYPE_VSM"};function ny(n){return iy[n.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}var sy={[ls]:"ENVMAP_TYPE_CUBE",[Us]:"ENVMAP_TYPE_CUBE",[Xa]:"ENVMAP_TYPE_CUBE_UV"};function ry(n){return n.envMap===!1?"ENVMAP_TYPE_CUBE":sy[n.envMapMode]||"ENVMAP_TYPE_CUBE"}var ay={[Us]:"ENVMAP_MODE_REFRACTION"};function oy(n){return n.envMap===!1?"ENVMAP_MODE_REFLECTION":ay[n.envMapMode]||"ENVMAP_MODE_REFLECTION"}var ly={[Ol]:"ENVMAP_BLENDING_MULTIPLY",[Ef]:"ENVMAP_BLENDING_MIX",[wf]:"ENVMAP_BLENDING_ADD"};function cy(n){return n.envMap===!1?"ENVMAP_BLENDING_NONE":ly[n.combine]||"ENVMAP_BLENDING_NONE"}function hy(n){let e=n.envMapCubeUVHeight;if(e===null)return null;let t=Math.log2(e)-2,i=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:i,maxMip:t}}function uy(n,e,t,i){let s=n.getContext(),r=t.defines,a=t.vertexShader,o=t.fragmentShader,c=ny(t),l=ry(t),h=oy(t),d=cy(t),u=hy(t),f=$v(t),g=Zv(r),x=s.createProgram(),p,m,M=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(p=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(to).join(`
`),p.length>0&&(p+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(to).join(`
`),m.length>0&&(m+=`
`)):(p=[cp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+h:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexNormals?"#define HAS_NORMAL":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(to).join(`
`),m=[cp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+l:"",t.envMap?"#define "+h:"",t.envMap?"#define "+d:"",u?"#define CUBEUV_TEXEL_WIDTH "+u.texelWidth:"",u?"#define CUBEUV_TEXEL_HEIGHT "+u.texelHeight:"",u?"#define CUBEUV_MAX_MIP "+u.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor?"#define USE_COLOR":"",t.vertexAlphas||t.batchingColor?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.numLightProbeGrids>0?"#define USE_LIGHT_PROBES_GRID":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==rn?"#define TONE_MAPPING":"",t.toneMapping!==rn?at.tonemapping_pars_fragment:"",t.toneMapping!==rn?qv("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",at.colorspace_pars_fragment,Wv("linearToOutputTexel",t.outputColorSpace),Yv(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(to).join(`
`)),a=Du(a),a=ap(a,t),a=op(a,t),o=Du(o),o=ap(o,t),o=op(o,t),a=lp(a),o=lp(o),t.isRawShaderMaterial!==!0&&(M=`#version 300 es
`,p=[f,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+p,m=["#define varying in",t.glslVersion===hu?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===hu?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);let b=M+p+a,v=M+m+o,T=np(s,s.VERTEX_SHADER,b),w=np(s,s.FRAGMENT_SHADER,v);s.attachShader(x,T),s.attachShader(x,w),t.index0AttributeName!==void 0?s.bindAttribLocation(x,0,t.index0AttributeName):t.hasPositionAttribute===!0&&s.bindAttribLocation(x,0,"position"),s.linkProgram(x);function C(I){if(n.debug.checkShaderErrors){let L=s.getProgramInfoLog(x)||"",X=s.getShaderInfoLog(T)||"",W=s.getShaderInfoLog(w)||"",U=L.trim(),z=X.trim(),H=W.trim(),Q=!0,ie=!0;if(s.getProgramParameter(x,s.LINK_STATUS)===!1)if(Q=!1,typeof n.debug.onShaderError=="function")n.debug.onShaderError(s,x,T,w);else{let q=rp(s,T,"vertex"),Z=rp(s,w,"fragment");$e("WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(x,s.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+U+`
`+q+`
`+Z)}else U!==""?Ye("WebGLProgram: Program Info Log:",U):(z===""||H==="")&&(ie=!1);ie&&(I.diagnostics={runnable:Q,programLog:U,vertexShader:{log:z,prefix:p},fragmentShader:{log:H,prefix:m}})}s.deleteShader(T),s.deleteShader(w),_=new Dr(s,x),E=Jv(s,x)}let _;this.getUniforms=function(){return _===void 0&&C(this),_};let E;this.getAttributes=function(){return E===void 0&&C(this),E};let P=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return P===!1&&(P=s.getProgramParameter(x,kv)),P},this.destroy=function(){i.releaseStatesOfProgram(this),s.deleteProgram(x),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=Hv++,this.cacheKey=e,this.usedTimes=1,this.program=x,this.vertexShader=T,this.fragmentShader=w,this}var dy=0,Lu=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e,t,i){let s=this._getShaderCacheForMaterial(e);return s.has(t)===!1&&(s.add(t),t.usedTimes++),s.has(i)===!1&&(s.add(i),i.usedTimes++),this}remove(e){let t=this.materialCache.get(e);for(let i of t)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(e),this}getVertexShaderStage(e){return this._getShaderStage(e.vertexShader)}getFragmentShaderStage(e){return this._getShaderStage(e.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){let t=this.materialCache,i=t.get(e);return i===void 0&&(i=new Set,t.set(e,i)),i}_getShaderStage(e){let t=this.shaderCache,i=t.get(e);return i===void 0&&(i=new Nu(e),t.set(e,i)),i}},Nu=class{constructor(e){this.id=dy++,this.code=e,this.usedTimes=0}};function fy(n){return n===us||n===Ka||n===ja}function py(n,e,t,i,s,r){let a=new gr,o=new Lu,c=new Set,l=[],h=new Map,d=i.logarithmicDepthBuffer,u=i.precision,f={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function g(_){return c.add(_),_===0?"uv":`uv${_}`}function x(_,E,P,I,L,X){let W=I.fog,U=L.geometry,z=_.isMeshStandardMaterial||_.isMeshLambertMaterial||_.isMeshPhongMaterial?I.environment:null,H=_.isMeshStandardMaterial||_.isMeshLambertMaterial&&!_.envMap||_.isMeshPhongMaterial&&!_.envMap,Q=e.get(_.envMap||z,H),ie=Q&&Q.mapping===Xa?Q.image.height:null,q=f[_.type];_.precision!==null&&(u=i.getMaxPrecision(_.precision),u!==_.precision&&Ye("WebGLProgram.getParameters:",_.precision,"not supported, using",u,"instead."));let Z=U.morphAttributes.position||U.morphAttributes.normal||U.morphAttributes.color,j=Z!==void 0?Z.length:0,de=0;U.morphAttributes.position!==void 0&&(de=1),U.morphAttributes.normal!==void 0&&(de=2),U.morphAttributes.color!==void 0&&(de=3);let Ge,me,k,ce;if(q){let Fe=pi[q];Ge=Fe.vertexShader,me=Fe.fragmentShader}else{Ge=_.vertexShader,me=_.fragmentShader;let Fe=o.getVertexShaderStage(_),Dt=o.getFragmentShaderStage(_);o.update(_,Fe,Dt),k=Fe.id,ce=Dt.id}let ae=n.getRenderTarget(),Te=n.state.buffers.depth.getReversed(),Ue=L.isInstancedMesh===!0,Oe=L.isBatchedMesh===!0,st=!!_.map,He=!!_.matcap,oe=!!Q,ee=!!_.aoMap,le=!!_.lightMap,J=!!_.bumpMap&&_.wireframe===!1,se=!!_.normalMap,fe=!!_.displacementMap,ge=!!_.emissiveMap,we=!!_.metalnessMap,Se=!!_.roughnessMap,D=_.anisotropy>0,Pe=_.clearcoat>0,Ze=_.dispersion>0,R=_.iridescence>0,y=_.sheen>0,F=_.transmission>0,B=D&&!!_.anisotropyMap,Y=Pe&&!!_.clearcoatMap,pe=Pe&&!!_.clearcoatNormalMap,_e=Pe&&!!_.clearcoatRoughnessMap,K=R&&!!_.iridescenceMap,ne=R&&!!_.iridescenceThicknessMap,Me=y&&!!_.sheenColorMap,ke=y&&!!_.sheenRoughnessMap,ve=!!_.specularMap,xe=!!_.specularColorMap,Be=!!_.specularIntensityMap,Xe=F&&!!_.transmissionMap,Je=F&&!!_.thicknessMap,N=!!_.gradientMap,be=!!_.alphaMap,re=_.alphaTest>0,Ee=!!_.alphaHash,Ce=!!_.extensions,he=rn;_.toneMapped&&(ae===null||ae.isXRRenderTarget===!0)&&(he=n.toneMapping);let Ve={shaderID:q,shaderType:_.type,shaderName:_.name,vertexShader:Ge,fragmentShader:me,defines:_.defines,customVertexShaderID:k,customFragmentShaderID:ce,isRawShaderMaterial:_.isRawShaderMaterial===!0,glslVersion:_.glslVersion,precision:u,batching:Oe,batchingColor:Oe&&L._colorsTexture!==null,instancing:Ue,instancingColor:Ue&&L.instanceColor!==null,instancingMorph:Ue&&L.morphTexture!==null,outputColorSpace:ae===null?n.outputColorSpace:ae.isXRRenderTarget===!0?ae.texture.colorSpace:ht.workingColorSpace,alphaToCoverage:!!_.alphaToCoverage,map:st,matcap:He,envMap:oe,envMapMode:oe&&Q.mapping,envMapCubeUVHeight:ie,aoMap:ee,lightMap:le,bumpMap:J,normalMap:se,displacementMap:fe,emissiveMap:ge,normalMapObjectSpace:se&&_.normalMapType===Rf,normalMapTangentSpace:se&&_.normalMapType===Cr,packedNormalMap:se&&_.normalMapType===Cr&&fy(_.normalMap.format),metalnessMap:we,roughnessMap:Se,anisotropy:D,anisotropyMap:B,clearcoat:Pe,clearcoatMap:Y,clearcoatNormalMap:pe,clearcoatRoughnessMap:_e,dispersion:Ze,iridescence:R,iridescenceMap:K,iridescenceThicknessMap:ne,sheen:y,sheenColorMap:Me,sheenRoughnessMap:ke,specularMap:ve,specularColorMap:xe,specularIntensityMap:Be,transmission:F,transmissionMap:Xe,thicknessMap:Je,gradientMap:N,opaque:_.transparent===!1&&_.blending===As&&_.alphaToCoverage===!1,alphaMap:be,alphaTest:re,alphaHash:Ee,combine:_.combine,mapUv:st&&g(_.map.channel),aoMapUv:ee&&g(_.aoMap.channel),lightMapUv:le&&g(_.lightMap.channel),bumpMapUv:J&&g(_.bumpMap.channel),normalMapUv:se&&g(_.normalMap.channel),displacementMapUv:fe&&g(_.displacementMap.channel),emissiveMapUv:ge&&g(_.emissiveMap.channel),metalnessMapUv:we&&g(_.metalnessMap.channel),roughnessMapUv:Se&&g(_.roughnessMap.channel),anisotropyMapUv:B&&g(_.anisotropyMap.channel),clearcoatMapUv:Y&&g(_.clearcoatMap.channel),clearcoatNormalMapUv:pe&&g(_.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:_e&&g(_.clearcoatRoughnessMap.channel),iridescenceMapUv:K&&g(_.iridescenceMap.channel),iridescenceThicknessMapUv:ne&&g(_.iridescenceThicknessMap.channel),sheenColorMapUv:Me&&g(_.sheenColorMap.channel),sheenRoughnessMapUv:ke&&g(_.sheenRoughnessMap.channel),specularMapUv:ve&&g(_.specularMap.channel),specularColorMapUv:xe&&g(_.specularColorMap.channel),specularIntensityMapUv:Be&&g(_.specularIntensityMap.channel),transmissionMapUv:Xe&&g(_.transmissionMap.channel),thicknessMapUv:Je&&g(_.thicknessMap.channel),alphaMapUv:be&&g(_.alphaMap.channel),vertexTangents:!!U.attributes.tangent&&(se||D),vertexNormals:!!U.attributes.normal,vertexColors:_.vertexColors,vertexAlphas:_.vertexColors===!0&&!!U.attributes.color&&U.attributes.color.itemSize===4,pointsUvs:L.isPoints===!0&&!!U.attributes.uv&&(st||be),fog:!!W,useFog:_.fog===!0,fogExp2:!!W&&W.isFogExp2,flatShading:_.wireframe===!1&&(_.flatShading===!0||U.attributes.normal===void 0&&se===!1&&(_.isMeshLambertMaterial||_.isMeshPhongMaterial||_.isMeshStandardMaterial||_.isMeshPhysicalMaterial)),sizeAttenuation:_.sizeAttenuation===!0,logarithmicDepthBuffer:d,reversedDepthBuffer:Te,skinning:L.isSkinnedMesh===!0,hasPositionAttribute:U.attributes.position!==void 0,morphTargets:U.morphAttributes.position!==void 0,morphNormals:U.morphAttributes.normal!==void 0,morphColors:U.morphAttributes.color!==void 0,morphTargetsCount:j,morphTextureStride:de,numDirLights:E.directional.length,numPointLights:E.point.length,numSpotLights:E.spot.length,numSpotLightMaps:E.spotLightMap.length,numRectAreaLights:E.rectArea.length,numHemiLights:E.hemi.length,numDirLightShadows:E.directionalShadowMap.length,numPointLightShadows:E.pointShadowMap.length,numSpotLightShadows:E.spotShadowMap.length,numSpotLightShadowsWithMaps:E.numSpotLightShadowsWithMaps,numLightProbes:E.numLightProbes,numLightProbeGrids:X.length,numClippingPlanes:r.numPlanes,numClipIntersection:r.numIntersection,dithering:_.dithering,shadowMapEnabled:n.shadowMap.enabled&&P.length>0,shadowMapType:n.shadowMap.type,toneMapping:he,decodeVideoTexture:st&&_.map.isVideoTexture===!0&&ht.getTransfer(_.map.colorSpace)===ft,decodeVideoTextureEmissive:ge&&_.emissiveMap.isVideoTexture===!0&&ht.getTransfer(_.emissiveMap.colorSpace)===ft,premultipliedAlpha:_.premultipliedAlpha,doubleSided:_.side===xi,flipSided:_.side===Qt,useDepthPacking:_.depthPacking>=0,depthPacking:_.depthPacking||0,index0AttributeName:_.index0AttributeName,extensionClipCullDistance:Ce&&_.extensions.clipCullDistance===!0&&t.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Ce&&_.extensions.multiDraw===!0||Oe)&&t.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:t.has("KHR_parallel_shader_compile"),customProgramCacheKey:_.customProgramCacheKey()};return Ve.vertexUv1s=c.has(1),Ve.vertexUv2s=c.has(2),Ve.vertexUv3s=c.has(3),c.clear(),Ve}function p(_){let E=[];if(_.shaderID?E.push(_.shaderID):(E.push(_.customVertexShaderID),E.push(_.customFragmentShaderID)),_.defines!==void 0)for(let P in _.defines)E.push(P),E.push(_.defines[P]);return _.isRawShaderMaterial===!1&&(m(E,_),M(E,_),E.push(n.outputColorSpace)),E.push(_.customProgramCacheKey),E.join()}function m(_,E){_.push(E.precision),_.push(E.outputColorSpace),_.push(E.envMapMode),_.push(E.envMapCubeUVHeight),_.push(E.mapUv),_.push(E.alphaMapUv),_.push(E.lightMapUv),_.push(E.aoMapUv),_.push(E.bumpMapUv),_.push(E.normalMapUv),_.push(E.displacementMapUv),_.push(E.emissiveMapUv),_.push(E.metalnessMapUv),_.push(E.roughnessMapUv),_.push(E.anisotropyMapUv),_.push(E.clearcoatMapUv),_.push(E.clearcoatNormalMapUv),_.push(E.clearcoatRoughnessMapUv),_.push(E.iridescenceMapUv),_.push(E.iridescenceThicknessMapUv),_.push(E.sheenColorMapUv),_.push(E.sheenRoughnessMapUv),_.push(E.specularMapUv),_.push(E.specularColorMapUv),_.push(E.specularIntensityMapUv),_.push(E.transmissionMapUv),_.push(E.thicknessMapUv),_.push(E.combine),_.push(E.fogExp2),_.push(E.sizeAttenuation),_.push(E.morphTargetsCount),_.push(E.morphAttributeCount),_.push(E.numDirLights),_.push(E.numPointLights),_.push(E.numSpotLights),_.push(E.numSpotLightMaps),_.push(E.numHemiLights),_.push(E.numRectAreaLights),_.push(E.numDirLightShadows),_.push(E.numPointLightShadows),_.push(E.numSpotLightShadows),_.push(E.numSpotLightShadowsWithMaps),_.push(E.numLightProbes),_.push(E.shadowMapType),_.push(E.toneMapping),_.push(E.numClippingPlanes),_.push(E.numClipIntersection),_.push(E.depthPacking)}function M(_,E){a.disableAll(),E.instancing&&a.enable(0),E.instancingColor&&a.enable(1),E.instancingMorph&&a.enable(2),E.matcap&&a.enable(3),E.envMap&&a.enable(4),E.normalMapObjectSpace&&a.enable(5),E.normalMapTangentSpace&&a.enable(6),E.clearcoat&&a.enable(7),E.iridescence&&a.enable(8),E.alphaTest&&a.enable(9),E.vertexColors&&a.enable(10),E.vertexAlphas&&a.enable(11),E.vertexUv1s&&a.enable(12),E.vertexUv2s&&a.enable(13),E.vertexUv3s&&a.enable(14),E.vertexTangents&&a.enable(15),E.anisotropy&&a.enable(16),E.alphaHash&&a.enable(17),E.batching&&a.enable(18),E.dispersion&&a.enable(19),E.batchingColor&&a.enable(20),E.gradientMap&&a.enable(21),E.packedNormalMap&&a.enable(22),E.vertexNormals&&a.enable(23),_.push(a.mask),a.disableAll(),E.fog&&a.enable(0),E.useFog&&a.enable(1),E.flatShading&&a.enable(2),E.logarithmicDepthBuffer&&a.enable(3),E.reversedDepthBuffer&&a.enable(4),E.skinning&&a.enable(5),E.morphTargets&&a.enable(6),E.morphNormals&&a.enable(7),E.morphColors&&a.enable(8),E.premultipliedAlpha&&a.enable(9),E.shadowMapEnabled&&a.enable(10),E.doubleSided&&a.enable(11),E.flipSided&&a.enable(12),E.useDepthPacking&&a.enable(13),E.dithering&&a.enable(14),E.transmission&&a.enable(15),E.sheen&&a.enable(16),E.opaque&&a.enable(17),E.pointsUvs&&a.enable(18),E.decodeVideoTexture&&a.enable(19),E.decodeVideoTextureEmissive&&a.enable(20),E.alphaToCoverage&&a.enable(21),E.numLightProbeGrids>0&&a.enable(22),E.hasPositionAttribute&&a.enable(23),_.push(a.mask)}function b(_){let E=f[_.type],P;if(E){let I=pi[E];P=fi.clone(I.uniforms)}else P=_.uniforms;return P}function v(_,E){let P=h.get(E);return P!==void 0?++P.usedTimes:(P=new uy(n,E,_,s),l.push(P),h.set(E,P)),P}function T(_){if(--_.usedTimes===0){let E=l.indexOf(_);l[E]=l[l.length-1],l.pop(),h.delete(_.cacheKey),_.destroy()}}function w(_){o.remove(_)}function C(){o.dispose()}return{getParameters:x,getProgramCacheKey:p,getUniforms:b,acquireProgram:v,releaseProgram:T,releaseShaderCache:w,programs:l,dispose:C}}function my(){let n=new WeakMap;function e(a){return n.has(a)}function t(a){let o=n.get(a);return o===void 0&&(o={},n.set(a,o)),o}function i(a){n.delete(a)}function s(a,o,c){n.get(a)[o]=c}function r(){n=new WeakMap}return{has:e,get:t,remove:i,update:s,dispose:r}}function gy(n,e){return n.groupOrder!==e.groupOrder?n.groupOrder-e.groupOrder:n.renderOrder!==e.renderOrder?n.renderOrder-e.renderOrder:n.material.id!==e.material.id?n.material.id-e.material.id:n.materialVariant!==e.materialVariant?n.materialVariant-e.materialVariant:n.z!==e.z?n.z-e.z:n.id-e.id}function hp(n,e){return n.groupOrder!==e.groupOrder?n.groupOrder-e.groupOrder:n.renderOrder!==e.renderOrder?n.renderOrder-e.renderOrder:n.z!==e.z?e.z-n.z:n.id-e.id}function up(){let n=[],e=0,t=[],i=[],s=[];function r(){e=0,t.length=0,i.length=0,s.length=0}function a(u){let f=0;return u.isInstancedMesh&&(f+=2),u.isSkinnedMesh&&(f+=1),f}function o(u,f,g,x,p,m){let M=n[e];return M===void 0?(M={id:u.id,object:u,geometry:f,material:g,materialVariant:a(u),groupOrder:x,renderOrder:u.renderOrder,z:p,group:m},n[e]=M):(M.id=u.id,M.object=u,M.geometry=f,M.material=g,M.materialVariant=a(u),M.groupOrder=x,M.renderOrder=u.renderOrder,M.z=p,M.group=m),e++,M}function c(u,f,g,x,p,m){let M=o(u,f,g,x,p,m);g.transmission>0?i.push(M):g.transparent===!0?s.push(M):t.push(M)}function l(u,f,g,x,p,m){let M=o(u,f,g,x,p,m);g.transmission>0?i.unshift(M):g.transparent===!0?s.unshift(M):t.unshift(M)}function h(u,f,g){t.length>1&&t.sort(u||gy),i.length>1&&i.sort(f||hp),s.length>1&&s.sort(f||hp),g&&(t.reverse(),i.reverse(),s.reverse())}function d(){for(let u=e,f=n.length;u<f;u++){let g=n[u];if(g.id===null)break;g.id=null,g.object=null,g.geometry=null,g.material=null,g.group=null}}return{opaque:t,transmissive:i,transparent:s,init:r,push:c,unshift:l,finish:d,sort:h}}function _y(){let n=new WeakMap;function e(i,s){let r=n.get(i),a;return r===void 0?(a=new up,n.set(i,[a])):s>=r.length?(a=new up,r.push(a)):a=r[s],a}function t(){n=new WeakMap}return{get:e,dispose:t}}function xy(){let n={};return{get:function(e){if(n[e.id]!==void 0)return n[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new A,color:new Le};break;case"SpotLight":t={position:new A,direction:new A,color:new Le,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new A,color:new Le,distance:0,decay:0};break;case"HemisphereLight":t={direction:new A,skyColor:new Le,groundColor:new Le};break;case"RectAreaLight":t={color:new Le,position:new A,halfWidth:new A,halfHeight:new A};break}return n[e.id]=t,t}}}function vy(){let n={};return{get:function(e){if(n[e.id]!==void 0)return n[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new te};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new te};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new te,shadowCameraNear:1,shadowCameraFar:1e3};break}return n[e.id]=t,t}}}var yy=0;function My(n,e){return(e.castShadow?2:0)-(n.castShadow?2:0)+(e.map?1:0)-(n.map?1:0)}function by(n){let e=new xy,t=vy(),i={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)i.probe.push(new A);let s=new A,r=new rt,a=new rt;function o(l){let h=0,d=0,u=0;for(let E=0;E<9;E++)i.probe[E].set(0,0,0);let f=0,g=0,x=0,p=0,m=0,M=0,b=0,v=0,T=0,w=0,C=0;l.sort(My);for(let E=0,P=l.length;E<P;E++){let I=l[E],L=I.color,X=I.intensity,W=I.distance,U=null;if(I.shadow&&I.shadow.map&&(I.shadow.map.texture.format===us?U=I.shadow.map.texture:U=I.shadow.map.depthTexture||I.shadow.map.texture),I.isAmbientLight)h+=L.r*X,d+=L.g*X,u+=L.b*X;else if(I.isLightProbe){for(let z=0;z<9;z++)i.probe[z].addScaledVector(I.sh.coefficients[z],X);C++}else if(I.isDirectionalLight){let z=e.get(I);if(z.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){let H=I.shadow,Q=t.get(I);Q.shadowIntensity=H.intensity,Q.shadowBias=H.bias,Q.shadowNormalBias=H.normalBias,Q.shadowRadius=H.radius,Q.shadowMapSize=H.mapSize,i.directionalShadow[f]=Q,i.directionalShadowMap[f]=U,i.directionalShadowMatrix[f]=I.shadow.matrix,M++}i.directional[f]=z,f++}else if(I.isSpotLight){let z=e.get(I);z.position.setFromMatrixPosition(I.matrixWorld),z.color.copy(L).multiplyScalar(X),z.distance=W,z.coneCos=Math.cos(I.angle),z.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),z.decay=I.decay,i.spot[x]=z;let H=I.shadow;if(I.map&&(i.spotLightMap[T]=I.map,T++,H.updateMatrices(I),I.castShadow&&w++),i.spotLightMatrix[x]=H.matrix,I.castShadow){let Q=t.get(I);Q.shadowIntensity=H.intensity,Q.shadowBias=H.bias,Q.shadowNormalBias=H.normalBias,Q.shadowRadius=H.radius,Q.shadowMapSize=H.mapSize,i.spotShadow[x]=Q,i.spotShadowMap[x]=U,v++}x++}else if(I.isRectAreaLight){let z=e.get(I);z.color.copy(L).multiplyScalar(X),z.halfWidth.set(I.width*.5,0,0),z.halfHeight.set(0,I.height*.5,0),i.rectArea[p]=z,p++}else if(I.isPointLight){let z=e.get(I);if(z.color.copy(I.color).multiplyScalar(I.intensity),z.distance=I.distance,z.decay=I.decay,I.castShadow){let H=I.shadow,Q=t.get(I);Q.shadowIntensity=H.intensity,Q.shadowBias=H.bias,Q.shadowNormalBias=H.normalBias,Q.shadowRadius=H.radius,Q.shadowMapSize=H.mapSize,Q.shadowCameraNear=H.camera.near,Q.shadowCameraFar=H.camera.far,i.pointShadow[g]=Q,i.pointShadowMap[g]=U,i.pointShadowMatrix[g]=I.shadow.matrix,b++}i.point[g]=z,g++}else if(I.isHemisphereLight){let z=e.get(I);z.skyColor.copy(I.color).multiplyScalar(X),z.groundColor.copy(I.groundColor).multiplyScalar(X),i.hemi[m]=z,m++}}p>0&&(n.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=ye.LTC_FLOAT_1,i.rectAreaLTC2=ye.LTC_FLOAT_2):(i.rectAreaLTC1=ye.LTC_HALF_1,i.rectAreaLTC2=ye.LTC_HALF_2)),i.ambient[0]=h,i.ambient[1]=d,i.ambient[2]=u;let _=i.hash;(_.directionalLength!==f||_.pointLength!==g||_.spotLength!==x||_.rectAreaLength!==p||_.hemiLength!==m||_.numDirectionalShadows!==M||_.numPointShadows!==b||_.numSpotShadows!==v||_.numSpotMaps!==T||_.numLightProbes!==C)&&(i.directional.length=f,i.spot.length=x,i.rectArea.length=p,i.point.length=g,i.hemi.length=m,i.directionalShadow.length=M,i.directionalShadowMap.length=M,i.pointShadow.length=b,i.pointShadowMap.length=b,i.spotShadow.length=v,i.spotShadowMap.length=v,i.directionalShadowMatrix.length=M,i.pointShadowMatrix.length=b,i.spotLightMatrix.length=v+T-w,i.spotLightMap.length=T,i.numSpotLightShadowsWithMaps=w,i.numLightProbes=C,_.directionalLength=f,_.pointLength=g,_.spotLength=x,_.rectAreaLength=p,_.hemiLength=m,_.numDirectionalShadows=M,_.numPointShadows=b,_.numSpotShadows=v,_.numSpotMaps=T,_.numLightProbes=C,i.version=yy++)}function c(l,h){let d=0,u=0,f=0,g=0,x=0,p=h.matrixWorldInverse;for(let m=0,M=l.length;m<M;m++){let b=l[m];if(b.isDirectionalLight){let v=i.directional[d];v.direction.setFromMatrixPosition(b.matrixWorld),s.setFromMatrixPosition(b.target.matrixWorld),v.direction.sub(s),v.direction.transformDirection(p),d++}else if(b.isSpotLight){let v=i.spot[f];v.position.setFromMatrixPosition(b.matrixWorld),v.position.applyMatrix4(p),v.direction.setFromMatrixPosition(b.matrixWorld),s.setFromMatrixPosition(b.target.matrixWorld),v.direction.sub(s),v.direction.transformDirection(p),f++}else if(b.isRectAreaLight){let v=i.rectArea[g];v.position.setFromMatrixPosition(b.matrixWorld),v.position.applyMatrix4(p),a.identity(),r.copy(b.matrixWorld),r.premultiply(p),a.extractRotation(r),v.halfWidth.set(b.width*.5,0,0),v.halfHeight.set(0,b.height*.5,0),v.halfWidth.applyMatrix4(a),v.halfHeight.applyMatrix4(a),g++}else if(b.isPointLight){let v=i.point[u];v.position.setFromMatrixPosition(b.matrixWorld),v.position.applyMatrix4(p),u++}else if(b.isHemisphereLight){let v=i.hemi[x];v.direction.setFromMatrixPosition(b.matrixWorld),v.direction.transformDirection(p),x++}}}return{setup:o,setupView:c,state:i}}function dp(n){let e=new by(n),t=[],i=[],s=[];function r(u){d.camera=u,t.length=0,i.length=0,s.length=0}function a(u){t.push(u)}function o(u){i.push(u)}function c(u){s.push(u)}function l(){e.setup(t)}function h(u){e.setupView(t,u)}let d={lightsArray:t,shadowsArray:i,lightProbeGridArray:s,camera:null,lights:e,transmissionRenderTarget:{},textureUnits:0};return{init:r,state:d,setupLights:l,setupLightsView:h,pushLight:a,pushShadow:o,pushLightProbeGrid:c}}function Sy(n){let e=new WeakMap;function t(s,r=0){let a=e.get(s),o;return a===void 0?(o=new dp(n),e.set(s,[o])):r>=a.length?(o=new dp(n),a.push(o)):o=a[r],o}function i(){e=new WeakMap}return{get:t,dispose:i}}var Ey=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,wy=`uniform sampler2D shadow_pass;
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
}`,Ty=[new A(1,0,0),new A(-1,0,0),new A(0,1,0),new A(0,-1,0),new A(0,0,1),new A(0,0,-1)],Ay=[new A(0,-1,0),new A(0,-1,0),new A(0,0,1),new A(0,0,-1),new A(0,-1,0),new A(0,-1,0)],fp=new rt,eo=new A,Au=new A;function Ry(n,e,t){let i=new vr,s=new te,r=new te,a=new gt,o=new yl,c=new Ml,l={},h=t.maxTextureSize,d={[ji]:Qt,[Qt]:ji,[xi]:xi},u=new Rt({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new te},radius:{value:4}},vertexShader:Ey,fragmentShader:wy}),f=u.clone();f.defines.HORIZONTAL_PASS=1;let g=new mt;g.setAttribute("position",new Yt(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let x=new tt(g,u),p=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Ds;let m=this.type;this.render=function(w,C,_){if(p.enabled===!1||p.autoUpdate===!1&&p.needsUpdate===!1||w.length===0)return;this.type===cf&&(Ye("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),this.type=Ds);let E=n.getRenderTarget(),P=n.getActiveCubeFace(),I=n.getActiveMipmapLevel(),L=n.state;L.setBlending(zt),L.buffers.depth.getReversed()===!0?L.buffers.color.setClear(0,0,0,0):L.buffers.color.setClear(1,1,1,1),L.buffers.depth.setTest(!0),L.setScissorTest(!1);let X=m!==this.type;X&&C.traverse(function(W){W.material&&(Array.isArray(W.material)?W.material.forEach(U=>U.needsUpdate=!0):W.material.needsUpdate=!0)});for(let W=0,U=w.length;W<U;W++){let z=w[W],H=z.shadow;if(H===void 0){Ye("WebGLShadowMap:",z,"has no shadow.");continue}if(H.autoUpdate===!1&&H.needsUpdate===!1)continue;s.copy(H.mapSize);let Q=H.getFrameExtents();s.multiply(Q),r.copy(H.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/Q.x),s.x=r.x*Q.x,H.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/Q.y),s.y=r.y*Q.y,H.mapSize.y=r.y));let ie=n.state.buffers.depth.getReversed();if(H.camera._reversedDepth=ie,H.map===null||X===!0){if(H.map!==null&&(H.map.depthTexture!==null&&(H.map.depthTexture.dispose(),H.map.depthTexture=null),H.map.dispose()),this.type===Ar){if(z.isPointLight){Ye("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}H.map=new Ht(s.x,s.y,{format:us,type:ei,minFilter:jt,magFilter:jt,generateMipmaps:!1}),H.map.texture.name=z.name+".shadowMap",H.map.depthTexture=new en(s.x,s.y,Vi),H.map.depthTexture.name=z.name+".shadowMapDepth",H.map.depthTexture.format=_n,H.map.depthTexture.compareFunction=null,H.map.depthTexture.minFilter=Ot,H.map.depthTexture.magFilter=Ot}else z.isPointLight?(H.map=new Rc(s.x),H.map.depthTexture=new fl(s.x,an)):(H.map=new Ht(s.x,s.y),H.map.depthTexture=new en(s.x,s.y,an)),H.map.depthTexture.name=z.name+".shadowMap",H.map.depthTexture.format=_n,this.type===Ds?(H.map.depthTexture.compareFunction=ie?wc:Ec,H.map.depthTexture.minFilter=jt,H.map.depthTexture.magFilter=jt):(H.map.depthTexture.compareFunction=null,H.map.depthTexture.minFilter=Ot,H.map.depthTexture.magFilter=Ot);H.camera.updateProjectionMatrix()}let q=H.map.isWebGLCubeRenderTarget?6:1;for(let Z=0;Z<q;Z++){if(H.map.isWebGLCubeRenderTarget)n.setRenderTarget(H.map,Z),n.clear();else{Z===0&&(n.setRenderTarget(H.map),n.clear());let j=H.getViewport(Z);a.set(r.x*j.x,r.y*j.y,r.x*j.z,r.y*j.w),L.viewport(a)}if(z.isPointLight){let j=H.camera,de=H.matrix,Ge=z.distance||j.far;Ge!==j.far&&(j.far=Ge,j.updateProjectionMatrix()),eo.setFromMatrixPosition(z.matrixWorld),j.position.copy(eo),Au.copy(j.position),Au.add(Ty[Z]),j.up.copy(Ay[Z]),j.lookAt(Au),j.updateMatrixWorld(),de.makeTranslation(-eo.x,-eo.y,-eo.z),fp.multiplyMatrices(j.projectionMatrix,j.matrixWorldInverse),H._frustum.setFromProjectionMatrix(fp,j.coordinateSystem,j.reversedDepth)}else H.updateMatrices(z);i=H.getFrustum(),v(C,_,H.camera,z,this.type)}H.isPointLightShadow!==!0&&this.type===Ar&&M(H,_),H.needsUpdate=!1}m=this.type,p.needsUpdate=!1,n.setRenderTarget(E,P,I)};function M(w,C){let _=e.update(x);u.defines.VSM_SAMPLES!==w.blurSamples&&(u.defines.VSM_SAMPLES=w.blurSamples,f.defines.VSM_SAMPLES=w.blurSamples,u.needsUpdate=!0,f.needsUpdate=!0),w.mapPass===null&&(w.mapPass=new Ht(s.x,s.y,{format:us,type:ei})),u.uniforms.shadow_pass.value=w.map.depthTexture,u.uniforms.resolution.value=w.mapSize,u.uniforms.radius.value=w.radius,n.setRenderTarget(w.mapPass),n.clear(),n.renderBufferDirect(C,null,_,u,x,null),f.uniforms.shadow_pass.value=w.mapPass.texture,f.uniforms.resolution.value=w.mapSize,f.uniforms.radius.value=w.radius,n.setRenderTarget(w.map),n.clear(),n.renderBufferDirect(C,null,_,f,x,null)}function b(w,C,_,E){let P=null,I=_.isPointLight===!0?w.customDistanceMaterial:w.customDepthMaterial;if(I!==void 0)P=I;else if(P=_.isPointLight===!0?c:o,n.localClippingEnabled&&C.clipShadows===!0&&Array.isArray(C.clippingPlanes)&&C.clippingPlanes.length!==0||C.displacementMap&&C.displacementScale!==0||C.alphaMap&&C.alphaTest>0||C.map&&C.alphaTest>0||C.alphaToCoverage===!0){let L=P.uuid,X=C.uuid,W=l[L];W===void 0&&(W={},l[L]=W);let U=W[X];U===void 0&&(U=P.clone(),W[X]=U,C.addEventListener("dispose",T)),P=U}if(P.visible=C.visible,P.wireframe=C.wireframe,E===Ar?P.side=C.shadowSide!==null?C.shadowSide:C.side:P.side=C.shadowSide!==null?C.shadowSide:d[C.side],P.alphaMap=C.alphaMap,P.alphaTest=C.alphaToCoverage===!0?.5:C.alphaTest,P.map=C.map,P.clipShadows=C.clipShadows,P.clippingPlanes=C.clippingPlanes,P.clipIntersection=C.clipIntersection,P.displacementMap=C.displacementMap,P.displacementScale=C.displacementScale,P.displacementBias=C.displacementBias,P.wireframeLinewidth=C.wireframeLinewidth,P.linewidth=C.linewidth,_.isPointLight===!0&&P.isMeshDistanceMaterial===!0){let L=n.properties.get(P);L.light=_}return P}function v(w,C,_,E,P){if(w.visible===!1)return;if(w.layers.test(C.layers)&&(w.isMesh||w.isLine||w.isPoints)&&(w.castShadow||w.receiveShadow&&P===Ar)&&(!w.frustumCulled||i.intersectsObject(w))){w.modelViewMatrix.multiplyMatrices(_.matrixWorldInverse,w.matrixWorld);let X=e.update(w),W=w.material;if(Array.isArray(W)){let U=X.groups;for(let z=0,H=U.length;z<H;z++){let Q=U[z],ie=W[Q.materialIndex];if(ie&&ie.visible){let q=b(w,ie,E,P);w.onBeforeShadow(n,w,C,_,X,q,Q),n.renderBufferDirect(_,null,X,q,w,Q),w.onAfterShadow(n,w,C,_,X,q,Q)}}}else if(W.visible){let U=b(w,W,E,P);w.onBeforeShadow(n,w,C,_,X,U,null),n.renderBufferDirect(_,null,X,U,w,null),w.onAfterShadow(n,w,C,_,X,U,null)}}let L=w.children;for(let X=0,W=L.length;X<W;X++)v(L[X],C,_,E,P)}function T(w){w.target.removeEventListener("dispose",T);for(let _ in l){let E=l[_],P=w.target.uuid;P in E&&(E[P].dispose(),delete E[P])}}}function Cy(n,e){function t(){let N=!1,be=new gt,re=null,Ee=new gt(0,0,0,0);return{setMask:function(Ce){re!==Ce&&!N&&(n.colorMask(Ce,Ce,Ce,Ce),re=Ce)},setLocked:function(Ce){N=Ce},setClear:function(Ce,he,Ve,Fe,Dt){Dt===!0&&(Ce*=Fe,he*=Fe,Ve*=Fe),be.set(Ce,he,Ve,Fe),Ee.equals(be)===!1&&(n.clearColor(Ce,he,Ve,Fe),Ee.copy(be))},reset:function(){N=!1,re=null,Ee.set(-1,0,0,0)}}}function i(){let N=!1,be=!1,re=null,Ee=null,Ce=null;return{setReversed:function(he){if(be!==he){let Ve=e.get("EXT_clip_control");he?Ve.clipControlEXT(Ve.LOWER_LEFT_EXT,Ve.ZERO_TO_ONE_EXT):Ve.clipControlEXT(Ve.LOWER_LEFT_EXT,Ve.NEGATIVE_ONE_TO_ONE_EXT),be=he;let Fe=Ce;Ce=null,this.setClear(Fe)}},getReversed:function(){return be},setTest:function(he){he?ae(n.DEPTH_TEST):Te(n.DEPTH_TEST)},setMask:function(he){re!==he&&!N&&(n.depthMask(he),re=he)},setFunc:function(he){if(be&&(he=Bf[he]),Ee!==he){switch(he){case Ko:n.depthFunc(n.NEVER);break;case jo:n.depthFunc(n.ALWAYS);break;case Qo:n.depthFunc(n.LESS);break;case Rs:n.depthFunc(n.LEQUAL);break;case el:n.depthFunc(n.EQUAL);break;case tl:n.depthFunc(n.GEQUAL);break;case il:n.depthFunc(n.GREATER);break;case nl:n.depthFunc(n.NOTEQUAL);break;default:n.depthFunc(n.LEQUAL)}Ee=he}},setLocked:function(he){N=he},setClear:function(he){Ce!==he&&(Ce=he,be&&(he=1-he),n.clearDepth(he))},reset:function(){N=!1,re=null,Ee=null,Ce=null,be=!1}}}function s(){let N=!1,be=null,re=null,Ee=null,Ce=null,he=null,Ve=null,Fe=null,Dt=null;return{setTest:function(St){N||(St?ae(n.STENCIL_TEST):Te(n.STENCIL_TEST))},setMask:function(St){be!==St&&!N&&(n.stencilMask(St),be=St)},setFunc:function(St,hn,un){(re!==St||Ee!==hn||Ce!==un)&&(n.stencilFunc(St,hn,un),re=St,Ee=hn,Ce=un)},setOp:function(St,hn,un){(he!==St||Ve!==hn||Fe!==un)&&(n.stencilOp(St,hn,un),he=St,Ve=hn,Fe=un)},setLocked:function(St){N=St},setClear:function(St){Dt!==St&&(n.clearStencil(St),Dt=St)},reset:function(){N=!1,be=null,re=null,Ee=null,Ce=null,he=null,Ve=null,Fe=null,Dt=null}}}let r=new t,a=new i,o=new s,c=new WeakMap,l=new WeakMap,h={},d={},u={},f=new WeakMap,g=[],x=null,p=!1,m=null,M=null,b=null,v=null,T=null,w=null,C=null,_=new Le(0,0,0),E=0,P=!1,I=null,L=null,X=null,W=null,U=null,z=n.getParameter(n.MAX_COMBINED_TEXTURE_IMAGE_UNITS),H=!1,Q=0,ie=n.getParameter(n.VERSION);ie.indexOf("WebGL")!==-1?(Q=parseFloat(/^WebGL (\d)/.exec(ie)[1]),H=Q>=1):ie.indexOf("OpenGL ES")!==-1&&(Q=parseFloat(/^OpenGL ES (\d)/.exec(ie)[1]),H=Q>=2);let q=null,Z={},j=n.getParameter(n.SCISSOR_BOX),de=n.getParameter(n.VIEWPORT),Ge=new gt().fromArray(j),me=new gt().fromArray(de);function k(N,be,re,Ee){let Ce=new Uint8Array(4),he=n.createTexture();n.bindTexture(N,he),n.texParameteri(N,n.TEXTURE_MIN_FILTER,n.NEAREST),n.texParameteri(N,n.TEXTURE_MAG_FILTER,n.NEAREST);for(let Ve=0;Ve<re;Ve++)N===n.TEXTURE_3D||N===n.TEXTURE_2D_ARRAY?n.texImage3D(be,0,n.RGBA,1,1,Ee,0,n.RGBA,n.UNSIGNED_BYTE,Ce):n.texImage2D(be+Ve,0,n.RGBA,1,1,0,n.RGBA,n.UNSIGNED_BYTE,Ce);return he}let ce={};ce[n.TEXTURE_2D]=k(n.TEXTURE_2D,n.TEXTURE_2D,1),ce[n.TEXTURE_CUBE_MAP]=k(n.TEXTURE_CUBE_MAP,n.TEXTURE_CUBE_MAP_POSITIVE_X,6),ce[n.TEXTURE_2D_ARRAY]=k(n.TEXTURE_2D_ARRAY,n.TEXTURE_2D_ARRAY,1,1),ce[n.TEXTURE_3D]=k(n.TEXTURE_3D,n.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),ae(n.DEPTH_TEST),a.setFunc(Rs),J(!1),se(Qh),ae(n.CULL_FACE),ee(zt);function ae(N){h[N]!==!0&&(n.enable(N),h[N]=!0)}function Te(N){h[N]!==!1&&(n.disable(N),h[N]=!1)}function Ue(N,be){return u[N]!==be?(n.bindFramebuffer(N,be),u[N]=be,N===n.DRAW_FRAMEBUFFER&&(u[n.FRAMEBUFFER]=be),N===n.FRAMEBUFFER&&(u[n.DRAW_FRAMEBUFFER]=be),!0):!1}function Oe(N,be){let re=g,Ee=!1;if(N){re=f.get(be),re===void 0&&(re=[],f.set(be,re));let Ce=N.textures;if(re.length!==Ce.length||re[0]!==n.COLOR_ATTACHMENT0){for(let he=0,Ve=Ce.length;he<Ve;he++)re[he]=n.COLOR_ATTACHMENT0+he;re.length=Ce.length,Ee=!0}}else re[0]!==n.BACK&&(re[0]=n.BACK,Ee=!0);Ee&&n.drawBuffers(re)}function st(N){return x!==N?(n.useProgram(N),x=N,!0):!1}let He={[Ti]:n.FUNC_ADD,[hf]:n.FUNC_SUBTRACT,[uf]:n.FUNC_REVERSE_SUBTRACT};He[df]=n.MIN,He[ff]=n.MAX;let oe={[Ls]:n.ZERO,[pf]:n.ONE,[mf]:n.SRC_COLOR,[Zo]:n.SRC_ALPHA,[vf]:n.SRC_ALPHA_SATURATE,[za]:n.DST_COLOR,[Ba]:n.DST_ALPHA,[gf]:n.ONE_MINUS_SRC_COLOR,[Jo]:n.ONE_MINUS_SRC_ALPHA,[xf]:n.ONE_MINUS_DST_COLOR,[_f]:n.ONE_MINUS_DST_ALPHA,[yf]:n.CONSTANT_COLOR,[Mf]:n.ONE_MINUS_CONSTANT_COLOR,[bf]:n.CONSTANT_ALPHA,[Sf]:n.ONE_MINUS_CONSTANT_ALPHA};function ee(N,be,re,Ee,Ce,he,Ve,Fe,Dt,St){if(N===zt){p===!0&&(Te(n.BLEND),p=!1);return}if(p===!1&&(ae(n.BLEND),p=!0),N!==Fl){if(N!==m||St!==P){if((M!==Ti||T!==Ti)&&(n.blendEquation(n.FUNC_ADD),M=Ti,T=Ti),St)switch(N){case As:n.blendFuncSeparate(n.ONE,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case eu:n.blendFunc(n.ONE,n.ONE);break;case tu:n.blendFuncSeparate(n.ZERO,n.ONE_MINUS_SRC_COLOR,n.ZERO,n.ONE);break;case iu:n.blendFuncSeparate(n.DST_COLOR,n.ONE_MINUS_SRC_ALPHA,n.ZERO,n.ONE);break;default:$e("WebGLState: Invalid blending: ",N);break}else switch(N){case As:n.blendFuncSeparate(n.SRC_ALPHA,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case eu:n.blendFuncSeparate(n.SRC_ALPHA,n.ONE,n.ONE,n.ONE);break;case tu:$e("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case iu:$e("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:$e("WebGLState: Invalid blending: ",N);break}b=null,v=null,w=null,C=null,_.set(0,0,0),E=0,m=N,P=St}return}Ce=Ce||be,he=he||re,Ve=Ve||Ee,(be!==M||Ce!==T)&&(n.blendEquationSeparate(He[be],He[Ce]),M=be,T=Ce),(re!==b||Ee!==v||he!==w||Ve!==C)&&(n.blendFuncSeparate(oe[re],oe[Ee],oe[he],oe[Ve]),b=re,v=Ee,w=he,C=Ve),(Fe.equals(_)===!1||Dt!==E)&&(n.blendColor(Fe.r,Fe.g,Fe.b,Dt),_.copy(Fe),E=Dt),m=N,P=!1}function le(N,be){N.side===xi?Te(n.CULL_FACE):ae(n.CULL_FACE);let re=N.side===Qt;be&&(re=!re),J(re),N.blending===As&&N.transparent===!1?ee(zt):ee(N.blending,N.blendEquation,N.blendSrc,N.blendDst,N.blendEquationAlpha,N.blendSrcAlpha,N.blendDstAlpha,N.blendColor,N.blendAlpha,N.premultipliedAlpha),a.setFunc(N.depthFunc),a.setTest(N.depthTest),a.setMask(N.depthWrite),r.setMask(N.colorWrite);let Ee=N.stencilWrite;o.setTest(Ee),Ee&&(o.setMask(N.stencilWriteMask),o.setFunc(N.stencilFunc,N.stencilRef,N.stencilFuncMask),o.setOp(N.stencilFail,N.stencilZFail,N.stencilZPass)),ge(N.polygonOffset,N.polygonOffsetFactor,N.polygonOffsetUnits),N.alphaToCoverage===!0?ae(n.SAMPLE_ALPHA_TO_COVERAGE):Te(n.SAMPLE_ALPHA_TO_COVERAGE)}function J(N){I!==N&&(N?n.frontFace(n.CW):n.frontFace(n.CCW),I=N)}function se(N){N!==of?(ae(n.CULL_FACE),N!==L&&(N===Qh?n.cullFace(n.BACK):N===lf?n.cullFace(n.FRONT):n.cullFace(n.FRONT_AND_BACK))):Te(n.CULL_FACE),L=N}function fe(N){N!==X&&(H&&n.lineWidth(N),X=N)}function ge(N,be,re){N?(ae(n.POLYGON_OFFSET_FILL),(W!==be||U!==re)&&(W=be,U=re,a.getReversed()&&(be=-be),n.polygonOffset(be,re))):Te(n.POLYGON_OFFSET_FILL)}function we(N){N?ae(n.SCISSOR_TEST):Te(n.SCISSOR_TEST)}function Se(N){N===void 0&&(N=n.TEXTURE0+z-1),q!==N&&(n.activeTexture(N),q=N)}function D(N,be,re){re===void 0&&(q===null?re=n.TEXTURE0+z-1:re=q);let Ee=Z[re];Ee===void 0&&(Ee={type:void 0,texture:void 0},Z[re]=Ee),(Ee.type!==N||Ee.texture!==be)&&(q!==re&&(n.activeTexture(re),q=re),n.bindTexture(N,be||ce[N]),Ee.type=N,Ee.texture=be)}function Pe(){let N=Z[q];N!==void 0&&N.type!==void 0&&(n.bindTexture(N.type,null),N.type=void 0,N.texture=void 0)}function Ze(){try{n.compressedTexImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function R(){try{n.compressedTexImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function y(){try{n.texSubImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function F(){try{n.texSubImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function B(){try{n.compressedTexSubImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function Y(){try{n.compressedTexSubImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function pe(){try{n.texStorage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function _e(){try{n.texStorage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function K(){try{n.texImage2D(...arguments)}catch(N){$e("WebGLState:",N)}}function ne(){try{n.texImage3D(...arguments)}catch(N){$e("WebGLState:",N)}}function Me(N){return d[N]!==void 0?d[N]:n.getParameter(N)}function ke(N,be){d[N]!==be&&(n.pixelStorei(N,be),d[N]=be)}function ve(N){Ge.equals(N)===!1&&(n.scissor(N.x,N.y,N.z,N.w),Ge.copy(N))}function xe(N){me.equals(N)===!1&&(n.viewport(N.x,N.y,N.z,N.w),me.copy(N))}function Be(N,be){let re=l.get(be);re===void 0&&(re=new WeakMap,l.set(be,re));let Ee=re.get(N);Ee===void 0&&(Ee=n.getUniformBlockIndex(be,N.name),re.set(N,Ee))}function Xe(N,be){let Ee=l.get(be).get(N);c.get(be)!==Ee&&(n.uniformBlockBinding(be,Ee,N.__bindingPointIndex),c.set(be,Ee))}function Je(){n.disable(n.BLEND),n.disable(n.CULL_FACE),n.disable(n.DEPTH_TEST),n.disable(n.POLYGON_OFFSET_FILL),n.disable(n.SCISSOR_TEST),n.disable(n.STENCIL_TEST),n.disable(n.SAMPLE_ALPHA_TO_COVERAGE),n.blendEquation(n.FUNC_ADD),n.blendFunc(n.ONE,n.ZERO),n.blendFuncSeparate(n.ONE,n.ZERO,n.ONE,n.ZERO),n.blendColor(0,0,0,0),n.colorMask(!0,!0,!0,!0),n.clearColor(0,0,0,0),n.depthMask(!0),n.depthFunc(n.LESS),a.setReversed(!1),n.clearDepth(1),n.stencilMask(4294967295),n.stencilFunc(n.ALWAYS,0,4294967295),n.stencilOp(n.KEEP,n.KEEP,n.KEEP),n.clearStencil(0),n.cullFace(n.BACK),n.frontFace(n.CCW),n.polygonOffset(0,0),n.activeTexture(n.TEXTURE0),n.bindFramebuffer(n.FRAMEBUFFER,null),n.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),n.bindFramebuffer(n.READ_FRAMEBUFFER,null),n.useProgram(null),n.lineWidth(1),n.scissor(0,0,n.canvas.width,n.canvas.height),n.viewport(0,0,n.canvas.width,n.canvas.height),n.pixelStorei(n.PACK_ALIGNMENT,4),n.pixelStorei(n.UNPACK_ALIGNMENT,4),n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,!1),n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,n.BROWSER_DEFAULT_WEBGL),n.pixelStorei(n.PACK_ROW_LENGTH,0),n.pixelStorei(n.PACK_SKIP_PIXELS,0),n.pixelStorei(n.PACK_SKIP_ROWS,0),n.pixelStorei(n.UNPACK_ROW_LENGTH,0),n.pixelStorei(n.UNPACK_IMAGE_HEIGHT,0),n.pixelStorei(n.UNPACK_SKIP_PIXELS,0),n.pixelStorei(n.UNPACK_SKIP_ROWS,0),n.pixelStorei(n.UNPACK_SKIP_IMAGES,0),h={},d={},q=null,Z={},u={},f=new WeakMap,g=[],x=null,p=!1,m=null,M=null,b=null,v=null,T=null,w=null,C=null,_=new Le(0,0,0),E=0,P=!1,I=null,L=null,X=null,W=null,U=null,Ge.set(0,0,n.canvas.width,n.canvas.height),me.set(0,0,n.canvas.width,n.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:ae,disable:Te,bindFramebuffer:Ue,drawBuffers:Oe,useProgram:st,setBlending:ee,setMaterial:le,setFlipSided:J,setCullFace:se,setLineWidth:fe,setPolygonOffset:ge,setScissorTest:we,activeTexture:Se,bindTexture:D,unbindTexture:Pe,compressedTexImage2D:Ze,compressedTexImage3D:R,texImage2D:K,texImage3D:ne,pixelStorei:ke,getParameter:Me,updateUBOMapping:Be,uniformBlockBinding:Xe,texStorage2D:pe,texStorage3D:_e,texSubImage2D:y,texSubImage3D:F,compressedTexSubImage2D:B,compressedTexSubImage3D:Y,scissor:ve,viewport:xe,reset:Je}}function Py(n,e,t,i,s,r,a){let o=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new te,h=new WeakMap,d=new Set,u,f=new WeakMap,g=!1;try{g=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(R,y){return g?new OffscreenCanvas(R,y):sa("canvas")}function p(R,y,F){let B=1,Y=Ze(R);if((Y.width>F||Y.height>F)&&(B=F/Math.max(Y.width,Y.height)),B<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){let pe=Math.floor(B*Y.width),_e=Math.floor(B*Y.height);u===void 0&&(u=x(pe,_e));let K=y?x(pe,_e):u;return K.width=pe,K.height=_e,K.getContext("2d").drawImage(R,0,0,pe,_e),Ye("WebGLRenderer: Texture has been resized from ("+Y.width+"x"+Y.height+") to ("+pe+"x"+_e+")."),K}else return"data"in R&&Ye("WebGLRenderer: Image in DataTexture is too big ("+Y.width+"x"+Y.height+")."),R;return R}function m(R){return R.generateMipmaps}function M(R){n.generateMipmap(R)}function b(R){return R.isWebGLCubeRenderTarget?n.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?n.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?n.TEXTURE_2D_ARRAY:n.TEXTURE_2D}function v(R,y,F,B,Y,pe=!1){if(R!==null){if(n[R]!==void 0)return n[R];Ye("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let _e;B&&(_e=e.get("EXT_texture_norm16"),_e||Ye("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let K=y;if(y===n.RED&&(F===n.FLOAT&&(K=n.R32F),F===n.HALF_FLOAT&&(K=n.R16F),F===n.UNSIGNED_BYTE&&(K=n.R8),F===n.UNSIGNED_SHORT&&_e&&(K=_e.R16_EXT),F===n.SHORT&&_e&&(K=_e.R16_SNORM_EXT)),y===n.RED_INTEGER&&(F===n.UNSIGNED_BYTE&&(K=n.R8UI),F===n.UNSIGNED_SHORT&&(K=n.R16UI),F===n.UNSIGNED_INT&&(K=n.R32UI),F===n.BYTE&&(K=n.R8I),F===n.SHORT&&(K=n.R16I),F===n.INT&&(K=n.R32I)),y===n.RG&&(F===n.FLOAT&&(K=n.RG32F),F===n.HALF_FLOAT&&(K=n.RG16F),F===n.UNSIGNED_BYTE&&(K=n.RG8),F===n.UNSIGNED_SHORT&&_e&&(K=_e.RG16_EXT),F===n.SHORT&&_e&&(K=_e.RG16_SNORM_EXT)),y===n.RG_INTEGER&&(F===n.UNSIGNED_BYTE&&(K=n.RG8UI),F===n.UNSIGNED_SHORT&&(K=n.RG16UI),F===n.UNSIGNED_INT&&(K=n.RG32UI),F===n.BYTE&&(K=n.RG8I),F===n.SHORT&&(K=n.RG16I),F===n.INT&&(K=n.RG32I)),y===n.RGB_INTEGER&&(F===n.UNSIGNED_BYTE&&(K=n.RGB8UI),F===n.UNSIGNED_SHORT&&(K=n.RGB16UI),F===n.UNSIGNED_INT&&(K=n.RGB32UI),F===n.BYTE&&(K=n.RGB8I),F===n.SHORT&&(K=n.RGB16I),F===n.INT&&(K=n.RGB32I)),y===n.RGBA_INTEGER&&(F===n.UNSIGNED_BYTE&&(K=n.RGBA8UI),F===n.UNSIGNED_SHORT&&(K=n.RGBA16UI),F===n.UNSIGNED_INT&&(K=n.RGBA32UI),F===n.BYTE&&(K=n.RGBA8I),F===n.SHORT&&(K=n.RGBA16I),F===n.INT&&(K=n.RGBA32I)),y===n.RGB&&(F===n.UNSIGNED_SHORT&&_e&&(K=_e.RGB16_EXT),F===n.SHORT&&_e&&(K=_e.RGB16_SNORM_EXT),F===n.UNSIGNED_INT_5_9_9_9_REV&&(K=n.RGB9_E5),F===n.UNSIGNED_INT_10F_11F_11F_REV&&(K=n.R11F_G11F_B10F)),y===n.RGBA){let ne=pe?na:ht.getTransfer(Y);F===n.FLOAT&&(K=n.RGBA32F),F===n.HALF_FLOAT&&(K=n.RGBA16F),F===n.UNSIGNED_BYTE&&(K=ne===ft?n.SRGB8_ALPHA8:n.RGBA8),F===n.UNSIGNED_SHORT&&_e&&(K=_e.RGBA16_EXT),F===n.SHORT&&_e&&(K=_e.RGBA16_SNORM_EXT),F===n.UNSIGNED_SHORT_4_4_4_4&&(K=n.RGBA4),F===n.UNSIGNED_SHORT_5_5_5_1&&(K=n.RGB5_A1)}return(K===n.R16F||K===n.R32F||K===n.RG16F||K===n.RG32F||K===n.RGBA16F||K===n.RGBA32F)&&e.get("EXT_color_buffer_float"),K}function T(R,y){let F;return R?y===null||y===an||y===hs?F=n.DEPTH24_STENCIL8:y===Vi?F=n.DEPTH32F_STENCIL8:y===Rr&&(F=n.DEPTH24_STENCIL8,Ye("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):y===null||y===an||y===hs?F=n.DEPTH_COMPONENT24:y===Vi?F=n.DEPTH_COMPONENT32F:y===Rr&&(F=n.DEPTH_COMPONENT16),F}function w(R,y){return m(R)===!0||R.isFramebufferTexture&&R.minFilter!==Ot&&R.minFilter!==jt?Math.log2(Math.max(y.width,y.height))+1:R.mipmaps!==void 0&&R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?y.mipmaps.length:1}function C(R){let y=R.target;y.removeEventListener("dispose",C),E(y),y.isVideoTexture&&h.delete(y),y.isHTMLTexture&&d.delete(y)}function _(R){let y=R.target;y.removeEventListener("dispose",_),I(y)}function E(R){let y=i.get(R);if(y.__webglInit===void 0)return;let F=R.source,B=f.get(F);if(B){let Y=B[y.__cacheKey];Y.usedTimes--,Y.usedTimes===0&&P(R),Object.keys(B).length===0&&f.delete(F)}i.remove(R)}function P(R){let y=i.get(R);n.deleteTexture(y.__webglTexture);let F=R.source,B=f.get(F);delete B[y.__cacheKey],a.memory.textures--}function I(R){let y=i.get(R);if(R.depthTexture&&(R.depthTexture.dispose(),i.remove(R.depthTexture)),R.isWebGLCubeRenderTarget)for(let B=0;B<6;B++){if(Array.isArray(y.__webglFramebuffer[B]))for(let Y=0;Y<y.__webglFramebuffer[B].length;Y++)n.deleteFramebuffer(y.__webglFramebuffer[B][Y]);else n.deleteFramebuffer(y.__webglFramebuffer[B]);y.__webglDepthbuffer&&n.deleteRenderbuffer(y.__webglDepthbuffer[B])}else{if(Array.isArray(y.__webglFramebuffer))for(let B=0;B<y.__webglFramebuffer.length;B++)n.deleteFramebuffer(y.__webglFramebuffer[B]);else n.deleteFramebuffer(y.__webglFramebuffer);if(y.__webglDepthbuffer&&n.deleteRenderbuffer(y.__webglDepthbuffer),y.__webglMultisampledFramebuffer&&n.deleteFramebuffer(y.__webglMultisampledFramebuffer),y.__webglColorRenderbuffer)for(let B=0;B<y.__webglColorRenderbuffer.length;B++)y.__webglColorRenderbuffer[B]&&n.deleteRenderbuffer(y.__webglColorRenderbuffer[B]);y.__webglDepthRenderbuffer&&n.deleteRenderbuffer(y.__webglDepthRenderbuffer)}let F=R.textures;for(let B=0,Y=F.length;B<Y;B++){let pe=i.get(F[B]);pe.__webglTexture&&(n.deleteTexture(pe.__webglTexture),a.memory.textures--),i.remove(F[B])}i.remove(R)}let L=0;function X(){L=0}function W(){return L}function U(R){L=R}function z(){let R=L;return R>=s.maxTextures&&Ye("WebGLTextures: Trying to use "+R+" texture units while this GPU supports only "+s.maxTextures),L+=1,R}function H(R){let y=[];return y.push(R.wrapS),y.push(R.wrapT),y.push(R.wrapR||0),y.push(R.magFilter),y.push(R.minFilter),y.push(R.anisotropy),y.push(R.internalFormat),y.push(R.format),y.push(R.type),y.push(R.generateMipmaps),y.push(R.premultiplyAlpha),y.push(R.flipY),y.push(R.unpackAlignment),y.push(R.colorSpace),y.join()}function Q(R,y){let F=i.get(R);if(R.isVideoTexture&&D(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&F.__version!==R.version){let B=R.image;if(B===null)Ye("WebGLRenderer: Texture marked for update but no image data found.");else if(B.complete===!1)Ye("WebGLRenderer: Texture marked for update but image is incomplete");else{Te(F,R,y);return}}else R.isExternalTexture&&(F.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(n.TEXTURE_2D,F.__webglTexture,n.TEXTURE0+y)}function ie(R,y){let F=i.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&F.__version!==R.version){Te(F,R,y);return}else R.isExternalTexture&&(F.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(n.TEXTURE_2D_ARRAY,F.__webglTexture,n.TEXTURE0+y)}function q(R,y){let F=i.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&F.__version!==R.version){Te(F,R,y);return}t.bindTexture(n.TEXTURE_3D,F.__webglTexture,n.TEXTURE0+y)}function Z(R,y){let F=i.get(R);if(R.isCubeDepthTexture!==!0&&R.version>0&&F.__version!==R.version){Ue(F,R,y);return}t.bindTexture(n.TEXTURE_CUBE_MAP,F.__webglTexture,n.TEXTURE0+y)}let j={[zi]:n.REPEAT,[mn]:n.CLAMP_TO_EDGE,[sl]:n.MIRRORED_REPEAT},de={[Ot]:n.NEAREST,[Tf]:n.NEAREST_MIPMAP_NEAREST,[qa]:n.NEAREST_MIPMAP_LINEAR,[jt]:n.LINEAR,[kl]:n.LINEAR_MIPMAP_NEAREST,[cs]:n.LINEAR_MIPMAP_LINEAR},Ge={[Cf]:n.NEVER,[Nf]:n.ALWAYS,[Pf]:n.LESS,[Ec]:n.LEQUAL,[If]:n.EQUAL,[wc]:n.GEQUAL,[Df]:n.GREATER,[Lf]:n.NOTEQUAL};function me(R,y){if(y.type===Vi&&e.has("OES_texture_float_linear")===!1&&(y.magFilter===jt||y.magFilter===kl||y.magFilter===qa||y.magFilter===cs||y.minFilter===jt||y.minFilter===kl||y.minFilter===qa||y.minFilter===cs)&&Ye("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),n.texParameteri(R,n.TEXTURE_WRAP_S,j[y.wrapS]),n.texParameteri(R,n.TEXTURE_WRAP_T,j[y.wrapT]),(R===n.TEXTURE_3D||R===n.TEXTURE_2D_ARRAY)&&n.texParameteri(R,n.TEXTURE_WRAP_R,j[y.wrapR]),n.texParameteri(R,n.TEXTURE_MAG_FILTER,de[y.magFilter]),n.texParameteri(R,n.TEXTURE_MIN_FILTER,de[y.minFilter]),y.compareFunction&&(n.texParameteri(R,n.TEXTURE_COMPARE_MODE,n.COMPARE_REF_TO_TEXTURE),n.texParameteri(R,n.TEXTURE_COMPARE_FUNC,Ge[y.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(y.magFilter===Ot||y.minFilter!==qa&&y.minFilter!==cs||y.type===Vi&&e.has("OES_texture_float_linear")===!1)return;if(y.anisotropy>1||i.get(y).__currentAnisotropy){let F=e.get("EXT_texture_filter_anisotropic");n.texParameterf(R,F.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(y.anisotropy,s.getMaxAnisotropy())),i.get(y).__currentAnisotropy=y.anisotropy}}}function k(R,y){let F=!1;R.__webglInit===void 0&&(R.__webglInit=!0,y.addEventListener("dispose",C));let B=y.source,Y=f.get(B);Y===void 0&&(Y={},f.set(B,Y));let pe=H(y);if(pe!==R.__cacheKey){Y[pe]===void 0&&(Y[pe]={texture:n.createTexture(),usedTimes:0},a.memory.textures++,F=!0),Y[pe].usedTimes++;let _e=Y[R.__cacheKey];_e!==void 0&&(Y[R.__cacheKey].usedTimes--,_e.usedTimes===0&&P(y)),R.__cacheKey=pe,R.__webglTexture=Y[pe].texture}return F}function ce(R,y,F){return Math.floor(Math.floor(R/F)/y)}function ae(R,y,F,B){let pe=R.updateRanges;if(pe.length===0)t.texSubImage2D(n.TEXTURE_2D,0,0,0,y.width,y.height,F,B,y.data);else{pe.sort((ke,ve)=>ke.start-ve.start);let _e=0;for(let ke=1;ke<pe.length;ke++){let ve=pe[_e],xe=pe[ke],Be=ve.start+ve.count,Xe=ce(xe.start,y.width,4),Je=ce(ve.start,y.width,4);xe.start<=Be+1&&Xe===Je&&ce(xe.start+xe.count-1,y.width,4)===Xe?ve.count=Math.max(ve.count,xe.start+xe.count-ve.start):(++_e,pe[_e]=xe)}pe.length=_e+1;let K=t.getParameter(n.UNPACK_ROW_LENGTH),ne=t.getParameter(n.UNPACK_SKIP_PIXELS),Me=t.getParameter(n.UNPACK_SKIP_ROWS);t.pixelStorei(n.UNPACK_ROW_LENGTH,y.width);for(let ke=0,ve=pe.length;ke<ve;ke++){let xe=pe[ke],Be=Math.floor(xe.start/4),Xe=Math.ceil(xe.count/4),Je=Be%y.width,N=Math.floor(Be/y.width),be=Xe,re=1;t.pixelStorei(n.UNPACK_SKIP_PIXELS,Je),t.pixelStorei(n.UNPACK_SKIP_ROWS,N),t.texSubImage2D(n.TEXTURE_2D,0,Je,N,be,re,F,B,y.data)}R.clearUpdateRanges(),t.pixelStorei(n.UNPACK_ROW_LENGTH,K),t.pixelStorei(n.UNPACK_SKIP_PIXELS,ne),t.pixelStorei(n.UNPACK_SKIP_ROWS,Me)}}function Te(R,y,F){let B=n.TEXTURE_2D;(y.isDataArrayTexture||y.isCompressedArrayTexture)&&(B=n.TEXTURE_2D_ARRAY),y.isData3DTexture&&(B=n.TEXTURE_3D);let Y=k(R,y),pe=y.source;t.bindTexture(B,R.__webglTexture,n.TEXTURE0+F);let _e=i.get(pe);if(pe.version!==_e.__version||Y===!0){if(t.activeTexture(n.TEXTURE0+F),(typeof ImageBitmap<"u"&&y.image instanceof ImageBitmap)===!1){let re=ht.getPrimaries(ht.workingColorSpace),Ee=y.colorSpace===Un?null:ht.getPrimaries(y.colorSpace),Ce=y.colorSpace===Un||re===Ee?n.NONE:n.BROWSER_DEFAULT_WEBGL;t.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,y.flipY),t.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,y.premultiplyAlpha),t.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ce)}t.pixelStorei(n.UNPACK_ALIGNMENT,y.unpackAlignment);let ne=p(y.image,!1,s.maxTextureSize);ne=Pe(y,ne);let Me=r.convert(y.format,y.colorSpace),ke=r.convert(y.type),ve=v(y.internalFormat,Me,ke,y.normalized,y.colorSpace,y.isVideoTexture);me(B,y);let xe,Be=y.mipmaps,Xe=y.isVideoTexture!==!0,Je=_e.__version===void 0||Y===!0,N=pe.dataReady,be=w(y,ne);if(y.isDepthTexture)ve=T(y.format===xn,y.type),Je&&(Xe?t.texStorage2D(n.TEXTURE_2D,1,ve,ne.width,ne.height):t.texImage2D(n.TEXTURE_2D,0,ve,ne.width,ne.height,0,Me,ke,null));else if(y.isDataTexture)if(Be.length>0){Xe&&Je&&t.texStorage2D(n.TEXTURE_2D,be,ve,Be[0].width,Be[0].height);for(let re=0,Ee=Be.length;re<Ee;re++)xe=Be[re],Xe?N&&t.texSubImage2D(n.TEXTURE_2D,re,0,0,xe.width,xe.height,Me,ke,xe.data):t.texImage2D(n.TEXTURE_2D,re,ve,xe.width,xe.height,0,Me,ke,xe.data);y.generateMipmaps=!1}else Xe?(Je&&t.texStorage2D(n.TEXTURE_2D,be,ve,ne.width,ne.height),N&&ae(y,ne,Me,ke)):t.texImage2D(n.TEXTURE_2D,0,ve,ne.width,ne.height,0,Me,ke,ne.data);else if(y.isCompressedTexture)if(y.isCompressedArrayTexture){Xe&&Je&&t.texStorage3D(n.TEXTURE_2D_ARRAY,be,ve,Be[0].width,Be[0].height,ne.depth);for(let re=0,Ee=Be.length;re<Ee;re++)if(xe=Be[re],y.format!==vi)if(Me!==null)if(Xe){if(N)if(y.layerUpdates.size>0){let Ce=gu(xe.width,xe.height,y.format,y.type);for(let he of y.layerUpdates){let Ve=xe.data.subarray(he*Ce/xe.data.BYTES_PER_ELEMENT,(he+1)*Ce/xe.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,re,0,0,he,xe.width,xe.height,1,Me,Ve)}y.clearLayerUpdates()}else t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,re,0,0,0,xe.width,xe.height,ne.depth,Me,xe.data)}else t.compressedTexImage3D(n.TEXTURE_2D_ARRAY,re,ve,xe.width,xe.height,ne.depth,0,xe.data,0,0);else Ye("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else Xe?N&&t.texSubImage3D(n.TEXTURE_2D_ARRAY,re,0,0,0,xe.width,xe.height,ne.depth,Me,ke,xe.data):t.texImage3D(n.TEXTURE_2D_ARRAY,re,ve,xe.width,xe.height,ne.depth,0,Me,ke,xe.data)}else{Xe&&Je&&t.texStorage2D(n.TEXTURE_2D,be,ve,Be[0].width,Be[0].height);for(let re=0,Ee=Be.length;re<Ee;re++)xe=Be[re],y.format!==vi?Me!==null?Xe?N&&t.compressedTexSubImage2D(n.TEXTURE_2D,re,0,0,xe.width,xe.height,Me,xe.data):t.compressedTexImage2D(n.TEXTURE_2D,re,ve,xe.width,xe.height,0,xe.data):Ye("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Xe?N&&t.texSubImage2D(n.TEXTURE_2D,re,0,0,xe.width,xe.height,Me,ke,xe.data):t.texImage2D(n.TEXTURE_2D,re,ve,xe.width,xe.height,0,Me,ke,xe.data)}else if(y.isDataArrayTexture)if(Xe){if(Je&&t.texStorage3D(n.TEXTURE_2D_ARRAY,be,ve,ne.width,ne.height,ne.depth),N)if(y.layerUpdates.size>0){let re=gu(ne.width,ne.height,y.format,y.type);for(let Ee of y.layerUpdates){let Ce=ne.data.subarray(Ee*re/ne.data.BYTES_PER_ELEMENT,(Ee+1)*re/ne.data.BYTES_PER_ELEMENT);t.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,Ee,ne.width,ne.height,1,Me,ke,Ce)}y.clearLayerUpdates()}else t.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,0,ne.width,ne.height,ne.depth,Me,ke,ne.data)}else t.texImage3D(n.TEXTURE_2D_ARRAY,0,ve,ne.width,ne.height,ne.depth,0,Me,ke,ne.data);else if(y.isData3DTexture)Xe?(Je&&t.texStorage3D(n.TEXTURE_3D,be,ve,ne.width,ne.height,ne.depth),N&&t.texSubImage3D(n.TEXTURE_3D,0,0,0,0,ne.width,ne.height,ne.depth,Me,ke,ne.data)):t.texImage3D(n.TEXTURE_3D,0,ve,ne.width,ne.height,ne.depth,0,Me,ke,ne.data);else if(y.isFramebufferTexture){if(Je)if(Xe)t.texStorage2D(n.TEXTURE_2D,be,ve,ne.width,ne.height);else{let re=ne.width,Ee=ne.height;for(let Ce=0;Ce<be;Ce++)t.texImage2D(n.TEXTURE_2D,Ce,ve,re,Ee,0,Me,ke,null),re>>=1,Ee>>=1}}else if(y.isHTMLTexture){if("texElementImage2D"in n){let re=n.canvas;if(re.hasAttribute("layoutsubtree")||re.setAttribute("layoutsubtree","true"),ne.parentNode!==re){re.appendChild(ne),d.add(y),re.onpaint=Ee=>{let Ce=Ee.changedElements;for(let he of d)Ce.includes(he.image)&&(he.needsUpdate=!0)},re.requestPaint();return}if(n.texElementImage2D.length===3)n.texElementImage2D(n.TEXTURE_2D,n.RGBA8,ne);else{let Ce=n.RGBA,he=n.RGBA,Ve=n.UNSIGNED_BYTE;n.texElementImage2D(n.TEXTURE_2D,0,Ce,he,Ve,ne)}n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MIN_FILTER,n.LINEAR),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE)}}else if(Be.length>0){if(Xe&&Je){let re=Ze(Be[0]);t.texStorage2D(n.TEXTURE_2D,be,ve,re.width,re.height)}for(let re=0,Ee=Be.length;re<Ee;re++)xe=Be[re],Xe?N&&t.texSubImage2D(n.TEXTURE_2D,re,0,0,Me,ke,xe):t.texImage2D(n.TEXTURE_2D,re,ve,Me,ke,xe);y.generateMipmaps=!1}else if(Xe){if(Je){let re=Ze(ne);t.texStorage2D(n.TEXTURE_2D,be,ve,re.width,re.height)}N&&t.texSubImage2D(n.TEXTURE_2D,0,0,0,Me,ke,ne)}else t.texImage2D(n.TEXTURE_2D,0,ve,Me,ke,ne);m(y)&&M(B),_e.__version=pe.version,y.onUpdate&&y.onUpdate(y)}R.__version=y.version}function Ue(R,y,F){if(y.image.length!==6)return;let B=k(R,y),Y=y.source;t.bindTexture(n.TEXTURE_CUBE_MAP,R.__webglTexture,n.TEXTURE0+F);let pe=i.get(Y);if(Y.version!==pe.__version||B===!0){t.activeTexture(n.TEXTURE0+F);let _e=ht.getPrimaries(ht.workingColorSpace),K=y.colorSpace===Un?null:ht.getPrimaries(y.colorSpace),ne=y.colorSpace===Un||_e===K?n.NONE:n.BROWSER_DEFAULT_WEBGL;t.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,y.flipY),t.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,y.premultiplyAlpha),t.pixelStorei(n.UNPACK_ALIGNMENT,y.unpackAlignment),t.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,ne);let Me=y.isCompressedTexture||y.image[0].isCompressedTexture,ke=y.image[0]&&y.image[0].isDataTexture,ve=[];for(let he=0;he<6;he++)!Me&&!ke?ve[he]=p(y.image[he],!0,s.maxCubemapSize):ve[he]=ke?y.image[he].image:y.image[he],ve[he]=Pe(y,ve[he]);let xe=ve[0],Be=r.convert(y.format,y.colorSpace),Xe=r.convert(y.type),Je=v(y.internalFormat,Be,Xe,y.normalized,y.colorSpace),N=y.isVideoTexture!==!0,be=pe.__version===void 0||B===!0,re=Y.dataReady,Ee=w(y,xe);me(n.TEXTURE_CUBE_MAP,y);let Ce;if(Me){N&&be&&t.texStorage2D(n.TEXTURE_CUBE_MAP,Ee,Je,xe.width,xe.height);for(let he=0;he<6;he++){Ce=ve[he].mipmaps;for(let Ve=0;Ve<Ce.length;Ve++){let Fe=Ce[Ve];y.format!==vi?Be!==null?N?re&&t.compressedTexSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve,0,0,Fe.width,Fe.height,Be,Fe.data):t.compressedTexImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve,Je,Fe.width,Fe.height,0,Fe.data):Ye("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):N?re&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve,0,0,Fe.width,Fe.height,Be,Xe,Fe.data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve,Je,Fe.width,Fe.height,0,Be,Xe,Fe.data)}}}else{if(Ce=y.mipmaps,N&&be){Ce.length>0&&Ee++;let he=Ze(ve[0]);t.texStorage2D(n.TEXTURE_CUBE_MAP,Ee,Je,he.width,he.height)}for(let he=0;he<6;he++)if(ke){N?re&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,0,0,ve[he].width,ve[he].height,Be,Xe,ve[he].data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,Je,ve[he].width,ve[he].height,0,Be,Xe,ve[he].data);for(let Ve=0;Ve<Ce.length;Ve++){let Dt=Ce[Ve].image[he].image;N?re&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve+1,0,0,Dt.width,Dt.height,Be,Xe,Dt.data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve+1,Je,Dt.width,Dt.height,0,Be,Xe,Dt.data)}}else{N?re&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,0,0,Be,Xe,ve[he]):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,Je,Be,Xe,ve[he]);for(let Ve=0;Ve<Ce.length;Ve++){let Fe=Ce[Ve];N?re&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve+1,0,0,Be,Xe,Fe.image[he]):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+he,Ve+1,Je,Be,Xe,Fe.image[he])}}}m(y)&&M(n.TEXTURE_CUBE_MAP),pe.__version=Y.version,y.onUpdate&&y.onUpdate(y)}R.__version=y.version}function Oe(R,y,F,B,Y,pe){let _e=r.convert(F.format,F.colorSpace),K=r.convert(F.type),ne=v(F.internalFormat,_e,K,F.normalized,F.colorSpace),Me=i.get(y),ke=i.get(F);if(ke.__renderTarget=y,!Me.__hasExternalTextures){let ve=Math.max(1,y.width>>pe),xe=Math.max(1,y.height>>pe);Y===n.TEXTURE_3D||Y===n.TEXTURE_2D_ARRAY?t.texImage3D(Y,pe,ne,ve,xe,y.depth,0,_e,K,null):t.texImage2D(Y,pe,ne,ve,xe,0,_e,K,null)}t.bindFramebuffer(n.FRAMEBUFFER,R),Se(y)?o.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,B,Y,ke.__webglTexture,0,we(y)):(Y===n.TEXTURE_2D||Y>=n.TEXTURE_CUBE_MAP_POSITIVE_X&&Y<=n.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&n.framebufferTexture2D(n.FRAMEBUFFER,B,Y,ke.__webglTexture,pe),t.bindFramebuffer(n.FRAMEBUFFER,null)}function st(R,y,F){if(n.bindRenderbuffer(n.RENDERBUFFER,R),y.depthBuffer){let B=y.depthTexture,Y=B&&B.isDepthTexture?B.type:null,pe=T(y.stencilBuffer,Y),_e=y.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;Se(y)?o.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,we(y),pe,y.width,y.height):F?n.renderbufferStorageMultisample(n.RENDERBUFFER,we(y),pe,y.width,y.height):n.renderbufferStorage(n.RENDERBUFFER,pe,y.width,y.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,_e,n.RENDERBUFFER,R)}else{let B=y.textures;for(let Y=0;Y<B.length;Y++){let pe=B[Y],_e=r.convert(pe.format,pe.colorSpace),K=r.convert(pe.type),ne=v(pe.internalFormat,_e,K,pe.normalized,pe.colorSpace);Se(y)?o.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,we(y),ne,y.width,y.height):F?n.renderbufferStorageMultisample(n.RENDERBUFFER,we(y),ne,y.width,y.height):n.renderbufferStorage(n.RENDERBUFFER,ne,y.width,y.height)}}n.bindRenderbuffer(n.RENDERBUFFER,null)}function He(R,y,F){let B=y.isWebGLCubeRenderTarget===!0;if(t.bindFramebuffer(n.FRAMEBUFFER,R),!(y.depthTexture&&y.depthTexture.isDepthTexture))throw new Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");let Y=i.get(y.depthTexture);if(Y.__renderTarget=y,(!Y.__webglTexture||y.depthTexture.image.width!==y.width||y.depthTexture.image.height!==y.height)&&(y.depthTexture.image.width=y.width,y.depthTexture.image.height=y.height,y.depthTexture.needsUpdate=!0),B){if(Y.__webglInit===void 0&&(Y.__webglInit=!0,y.depthTexture.addEventListener("dispose",C)),Y.__webglTexture===void 0){Y.__webglTexture=n.createTexture(),t.bindTexture(n.TEXTURE_CUBE_MAP,Y.__webglTexture),me(n.TEXTURE_CUBE_MAP,y.depthTexture);let Me=r.convert(y.depthTexture.format),ke=r.convert(y.depthTexture.type),ve;y.depthTexture.format===_n?ve=n.DEPTH_COMPONENT24:y.depthTexture.format===xn&&(ve=n.DEPTH24_STENCIL8);for(let xe=0;xe<6;xe++)n.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+xe,0,ve,y.width,y.height,0,Me,ke,null)}}else Q(y.depthTexture,0);let pe=Y.__webglTexture,_e=we(y),K=B?n.TEXTURE_CUBE_MAP_POSITIVE_X+F:n.TEXTURE_2D,ne=y.depthTexture.format===xn?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;if(y.depthTexture.format===_n)Se(y)?o.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,ne,K,pe,0,_e):n.framebufferTexture2D(n.FRAMEBUFFER,ne,K,pe,0);else if(y.depthTexture.format===xn)Se(y)?o.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,ne,K,pe,0,_e):n.framebufferTexture2D(n.FRAMEBUFFER,ne,K,pe,0);else throw new Error("THREE.WebGLTextures: Unknown depthTexture format.")}function oe(R){let y=i.get(R),F=R.isWebGLCubeRenderTarget===!0;if(y.__boundDepthTexture!==R.depthTexture){let B=R.depthTexture;if(y.__depthDisposeCallback&&y.__depthDisposeCallback(),B){let Y=()=>{delete y.__boundDepthTexture,delete y.__depthDisposeCallback,B.removeEventListener("dispose",Y)};B.addEventListener("dispose",Y),y.__depthDisposeCallback=Y}y.__boundDepthTexture=B}if(R.depthTexture&&!y.__autoAllocateDepthBuffer)if(F)for(let B=0;B<6;B++)He(y.__webglFramebuffer[B],R,B);else{let B=R.texture.mipmaps;B&&B.length>0?He(y.__webglFramebuffer[0],R,0):He(y.__webglFramebuffer,R,0)}else if(F){y.__webglDepthbuffer=[];for(let B=0;B<6;B++)if(t.bindFramebuffer(n.FRAMEBUFFER,y.__webglFramebuffer[B]),y.__webglDepthbuffer[B]===void 0)y.__webglDepthbuffer[B]=n.createRenderbuffer(),st(y.__webglDepthbuffer[B],R,!1);else{let Y=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,pe=y.__webglDepthbuffer[B];n.bindRenderbuffer(n.RENDERBUFFER,pe),n.framebufferRenderbuffer(n.FRAMEBUFFER,Y,n.RENDERBUFFER,pe)}}else{let B=R.texture.mipmaps;if(B&&B.length>0?t.bindFramebuffer(n.FRAMEBUFFER,y.__webglFramebuffer[0]):t.bindFramebuffer(n.FRAMEBUFFER,y.__webglFramebuffer),y.__webglDepthbuffer===void 0)y.__webglDepthbuffer=n.createRenderbuffer(),st(y.__webglDepthbuffer,R,!1);else{let Y=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,pe=y.__webglDepthbuffer;n.bindRenderbuffer(n.RENDERBUFFER,pe),n.framebufferRenderbuffer(n.FRAMEBUFFER,Y,n.RENDERBUFFER,pe)}}t.bindFramebuffer(n.FRAMEBUFFER,null)}function ee(R,y,F){let B=i.get(R);y!==void 0&&Oe(B.__webglFramebuffer,R,R.texture,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,0),F!==void 0&&oe(R)}function le(R){let y=R.texture,F=i.get(R),B=i.get(y);R.addEventListener("dispose",_);let Y=R.textures,pe=R.isWebGLCubeRenderTarget===!0,_e=Y.length>1;if(_e||(B.__webglTexture===void 0&&(B.__webglTexture=n.createTexture()),B.__version=y.version,a.memory.textures++),pe){F.__webglFramebuffer=[];for(let K=0;K<6;K++)if(y.mipmaps&&y.mipmaps.length>0){F.__webglFramebuffer[K]=[];for(let ne=0;ne<y.mipmaps.length;ne++)F.__webglFramebuffer[K][ne]=n.createFramebuffer()}else F.__webglFramebuffer[K]=n.createFramebuffer()}else{if(y.mipmaps&&y.mipmaps.length>0){F.__webglFramebuffer=[];for(let K=0;K<y.mipmaps.length;K++)F.__webglFramebuffer[K]=n.createFramebuffer()}else F.__webglFramebuffer=n.createFramebuffer();if(_e)for(let K=0,ne=Y.length;K<ne;K++){let Me=i.get(Y[K]);Me.__webglTexture===void 0&&(Me.__webglTexture=n.createTexture(),a.memory.textures++)}if(R.samples>0&&Se(R)===!1){F.__webglMultisampledFramebuffer=n.createFramebuffer(),F.__webglColorRenderbuffer=[],t.bindFramebuffer(n.FRAMEBUFFER,F.__webglMultisampledFramebuffer);for(let K=0;K<Y.length;K++){let ne=Y[K];F.__webglColorRenderbuffer[K]=n.createRenderbuffer(),n.bindRenderbuffer(n.RENDERBUFFER,F.__webglColorRenderbuffer[K]);let Me=r.convert(ne.format,ne.colorSpace),ke=r.convert(ne.type),ve=v(ne.internalFormat,Me,ke,ne.normalized,ne.colorSpace,R.isXRRenderTarget===!0),xe=we(R);n.renderbufferStorageMultisample(n.RENDERBUFFER,xe,ve,R.width,R.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+K,n.RENDERBUFFER,F.__webglColorRenderbuffer[K])}n.bindRenderbuffer(n.RENDERBUFFER,null),R.depthBuffer&&(F.__webglDepthRenderbuffer=n.createRenderbuffer(),st(F.__webglDepthRenderbuffer,R,!0)),t.bindFramebuffer(n.FRAMEBUFFER,null)}}if(pe){t.bindTexture(n.TEXTURE_CUBE_MAP,B.__webglTexture),me(n.TEXTURE_CUBE_MAP,y);for(let K=0;K<6;K++)if(y.mipmaps&&y.mipmaps.length>0)for(let ne=0;ne<y.mipmaps.length;ne++)Oe(F.__webglFramebuffer[K][ne],R,y,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+K,ne);else Oe(F.__webglFramebuffer[K],R,y,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+K,0);m(y)&&M(n.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(_e){for(let K=0,ne=Y.length;K<ne;K++){let Me=Y[K],ke=i.get(Me),ve=n.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(ve=R.isWebGL3DRenderTarget?n.TEXTURE_3D:n.TEXTURE_2D_ARRAY),t.bindTexture(ve,ke.__webglTexture),me(ve,Me),Oe(F.__webglFramebuffer,R,Me,n.COLOR_ATTACHMENT0+K,ve,0),m(Me)&&M(ve)}t.unbindTexture()}else{let K=n.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(K=R.isWebGL3DRenderTarget?n.TEXTURE_3D:n.TEXTURE_2D_ARRAY),t.bindTexture(K,B.__webglTexture),me(K,y),y.mipmaps&&y.mipmaps.length>0)for(let ne=0;ne<y.mipmaps.length;ne++)Oe(F.__webglFramebuffer[ne],R,y,n.COLOR_ATTACHMENT0,K,ne);else Oe(F.__webglFramebuffer,R,y,n.COLOR_ATTACHMENT0,K,0);m(y)&&M(K),t.unbindTexture()}R.depthBuffer&&oe(R)}function J(R){let y=R.textures;for(let F=0,B=y.length;F<B;F++){let Y=y[F];if(m(Y)){let pe=b(R),_e=i.get(Y).__webglTexture;t.bindTexture(pe,_e),M(pe),t.unbindTexture()}}}let se=[],fe=[];function ge(R){if(R.samples>0){if(Se(R)===!1){let y=R.textures,F=R.width,B=R.height,Y=n.COLOR_BUFFER_BIT,pe=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,_e=i.get(R),K=y.length>1;if(K)for(let Me=0;Me<y.length;Me++)t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+Me,n.RENDERBUFFER,null),t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+Me,n.TEXTURE_2D,null,0);t.bindFramebuffer(n.READ_FRAMEBUFFER,_e.__webglMultisampledFramebuffer);let ne=R.texture.mipmaps;ne&&ne.length>0?t.bindFramebuffer(n.DRAW_FRAMEBUFFER,_e.__webglFramebuffer[0]):t.bindFramebuffer(n.DRAW_FRAMEBUFFER,_e.__webglFramebuffer);for(let Me=0;Me<y.length;Me++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(Y|=n.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(Y|=n.STENCIL_BUFFER_BIT)),K){n.framebufferRenderbuffer(n.READ_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.RENDERBUFFER,_e.__webglColorRenderbuffer[Me]);let ke=i.get(y[Me]).__webglTexture;n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,ke,0)}n.blitFramebuffer(0,0,F,B,0,0,F,B,Y,n.NEAREST),c===!0&&(se.length=0,fe.length=0,se.push(n.COLOR_ATTACHMENT0+Me),R.depthBuffer&&R.resolveDepthBuffer===!1&&(se.push(pe),fe.push(pe),n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,fe)),n.invalidateFramebuffer(n.READ_FRAMEBUFFER,se))}if(t.bindFramebuffer(n.READ_FRAMEBUFFER,null),t.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),K)for(let Me=0;Me<y.length;Me++){t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+Me,n.RENDERBUFFER,_e.__webglColorRenderbuffer[Me]);let ke=i.get(y[Me]).__webglTexture;t.bindFramebuffer(n.FRAMEBUFFER,_e.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+Me,n.TEXTURE_2D,ke,0)}t.bindFramebuffer(n.DRAW_FRAMEBUFFER,_e.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.resolveDepthBuffer===!1&&c){let y=R.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,[y])}}}function we(R){return Math.min(s.maxSamples,R.samples)}function Se(R){let y=i.get(R);return R.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&y.__useRenderToTexture!==!1}function D(R){let y=a.render.frame;h.get(R)!==y&&(h.set(R,y),R.update())}function Pe(R,y){let F=R.colorSpace,B=R.format,Y=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||F!==ia&&F!==Un&&(ht.getTransfer(F)===ft?(B!==vi||Y!==li)&&Ye("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):$e("WebGLTextures: Unsupported texture color space:",F)),y}function Ze(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(l.width=R.naturalWidth||R.width,l.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(l.width=R.displayWidth,l.height=R.displayHeight):(l.width=R.width,l.height=R.height),l}this.allocateTextureUnit=z,this.resetTextureUnits=X,this.getTextureUnits=W,this.setTextureUnits=U,this.setTexture2D=Q,this.setTexture2DArray=ie,this.setTexture3D=q,this.setTextureCube=Z,this.rebindTextures=ee,this.setupRenderTarget=le,this.updateRenderTargetMipmap=J,this.updateMultisampleRenderTarget=ge,this.setupDepthRenderbuffer=oe,this.setupFrameBufferTexture=Oe,this.useMultisampledRTT=Se,this.isReversedDepthBuffer=function(){return t.buffers.depth.getReversed()}}function Iy(n,e){function t(i,s=Un){let r,a=ht.getTransfer(s);if(i===li)return n.UNSIGNED_BYTE;if(i===Vl)return n.UNSIGNED_SHORT_4_4_4_4;if(i===Gl)return n.UNSIGNED_SHORT_5_5_5_1;if(i===au)return n.UNSIGNED_INT_5_9_9_9_REV;if(i===ou)return n.UNSIGNED_INT_10F_11F_11F_REV;if(i===su)return n.BYTE;if(i===ru)return n.SHORT;if(i===Rr)return n.UNSIGNED_SHORT;if(i===Hl)return n.INT;if(i===an)return n.UNSIGNED_INT;if(i===Vi)return n.FLOAT;if(i===ei)return n.HALF_FLOAT;if(i===lu)return n.ALPHA;if(i===cu)return n.RGB;if(i===vi)return n.RGBA;if(i===_n)return n.DEPTH_COMPONENT;if(i===xn)return n.DEPTH_STENCIL;if(i===Wl)return n.RED;if(i===Xl)return n.RED_INTEGER;if(i===us)return n.RG;if(i===ql)return n.RG_INTEGER;if(i===Yl)return n.RGBA_INTEGER;if(i===Ya||i===$a||i===Za||i===Ja)if(a===ft)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(i===Ya)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===$a)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===Za)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===Ja)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(i===Ya)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===$a)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===Za)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===Ja)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===$l||i===Zl||i===Jl||i===Kl)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(i===$l)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===Zl)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===Jl)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===Kl)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===jl||i===Ql||i===ec||i===tc||i===ic||i===Ka||i===nc)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(i===jl||i===Ql)return a===ft?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(i===ec)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC;if(i===tc)return r.COMPRESSED_R11_EAC;if(i===ic)return r.COMPRESSED_SIGNED_R11_EAC;if(i===Ka)return r.COMPRESSED_RG11_EAC;if(i===nc)return r.COMPRESSED_SIGNED_RG11_EAC}else return null;if(i===sc||i===rc||i===ac||i===oc||i===lc||i===cc||i===hc||i===uc||i===dc||i===fc||i===pc||i===mc||i===gc||i===_c)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(i===sc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===rc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===ac)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===oc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===lc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===cc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===hc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===uc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===dc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===fc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===pc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===mc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===gc)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===_c)return a===ft?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===xc||i===vc||i===yc)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(i===xc)return a===ft?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===vc)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===yc)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===Mc||i===bc||i===ja||i===Sc)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(i===Mc)return r.COMPRESSED_RED_RGTC1_EXT;if(i===bc)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===ja)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===Sc)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===hs?n.UNSIGNED_INT_24_8:n[i]!==void 0?n[i]:null}return{convert:t}}var Dy=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,Ly=`
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

}`,Uu=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){let i=new pa(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=i}}getMesh(e){if(this.texture!==null&&this.mesh===null){let t=e.cameras[0].viewport,i=new Rt({vertexShader:Dy,fragmentShader:Ly,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new tt(new sn(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},Fu=class extends Qi{constructor(e,t){super();let i=this,s=null,r=1,a=null,o="local-floor",c=1,l=null,h=null,d=null,u=null,f=null,g=null,x=typeof XRWebGLBinding<"u",p=new Uu,m={},M=t.getContextAttributes(),b=null,v=null,T=[],w=[],C=new te,_=null,E=new Kt;E.viewport=new gt;let P=new Kt;P.viewport=new gt;let I=[E,P],L=new Nl,X=null,W=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(k){let ce=T[k];return ce===void 0&&(ce=new _r,T[k]=ce),ce.getTargetRaySpace()},this.getControllerGrip=function(k){let ce=T[k];return ce===void 0&&(ce=new _r,T[k]=ce),ce.getGripSpace()},this.getHand=function(k){let ce=T[k];return ce===void 0&&(ce=new _r,T[k]=ce),ce.getHandSpace()};function U(k){let ce=w.indexOf(k.inputSource);if(ce===-1)return;let ae=T[ce];ae!==void 0&&(ae.update(k.inputSource,k.frame,l||a),ae.dispatchEvent({type:k.type,data:k.inputSource}))}function z(){s.removeEventListener("select",U),s.removeEventListener("selectstart",U),s.removeEventListener("selectend",U),s.removeEventListener("squeeze",U),s.removeEventListener("squeezestart",U),s.removeEventListener("squeezeend",U),s.removeEventListener("end",z),s.removeEventListener("inputsourceschange",H);for(let k=0;k<T.length;k++){let ce=w[k];ce!==null&&(w[k]=null,T[k].disconnect(ce))}X=null,W=null,p.reset();for(let k in m)delete m[k];e.setRenderTarget(b),f=null,u=null,d=null,s=null,v=null,me.stop(),i.isPresenting=!1,e.setPixelRatio(_),e.setSize(C.width,C.height,!1),i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(k){r=k,i.isPresenting===!0&&Ye("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(k){o=k,i.isPresenting===!0&&Ye("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||a},this.setReferenceSpace=function(k){l=k},this.getBaseLayer=function(){return u!==null?u:f},this.getBinding=function(){return d===null&&x&&(d=new XRWebGLBinding(s,t)),d},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(k){if(s=k,s!==null){if(b=e.getRenderTarget(),s.addEventListener("select",U),s.addEventListener("selectstart",U),s.addEventListener("selectend",U),s.addEventListener("squeeze",U),s.addEventListener("squeezestart",U),s.addEventListener("squeezeend",U),s.addEventListener("end",z),s.addEventListener("inputsourceschange",H),M.xrCompatible!==!0&&await t.makeXRCompatible(),_=e.getPixelRatio(),e.getSize(C),x&&"createProjectionLayer"in XRWebGLBinding.prototype){let ae=null,Te=null,Ue=null;M.depth&&(Ue=M.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,ae=M.stencil?xn:_n,Te=M.stencil?hs:an);let Oe={colorFormat:t.RGBA8,depthFormat:Ue,scaleFactor:r};d=this.getBinding(),u=d.createProjectionLayer(Oe),s.updateRenderState({layers:[u]}),e.setPixelRatio(1),e.setSize(u.textureWidth,u.textureHeight,!1),v=new Ht(u.textureWidth,u.textureHeight,{format:vi,type:li,depthTexture:new en(u.textureWidth,u.textureHeight,Te,void 0,void 0,void 0,void 0,void 0,void 0,ae),stencilBuffer:M.stencil,colorSpace:e.outputColorSpace,samples:M.antialias?4:0,resolveDepthBuffer:u.ignoreDepthValues===!1,resolveStencilBuffer:u.ignoreDepthValues===!1})}else{let ae={antialias:M.antialias,alpha:!0,depth:M.depth,stencil:M.stencil,framebufferScaleFactor:r};f=new XRWebGLLayer(s,t,ae),s.updateRenderState({baseLayer:f}),e.setPixelRatio(1),e.setSize(f.framebufferWidth,f.framebufferHeight,!1),v=new Ht(f.framebufferWidth,f.framebufferHeight,{format:vi,type:li,colorSpace:e.outputColorSpace,stencilBuffer:M.stencil,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}v.isXRRenderTarget=!0,this.setFoveation(c),l=null,a=await s.requestReferenceSpace(o),me.setContext(s),me.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return p.getDepthTexture()};function H(k){for(let ce=0;ce<k.removed.length;ce++){let ae=k.removed[ce],Te=w.indexOf(ae);Te>=0&&(w[Te]=null,T[Te].disconnect(ae))}for(let ce=0;ce<k.added.length;ce++){let ae=k.added[ce],Te=w.indexOf(ae);if(Te===-1){for(let Oe=0;Oe<T.length;Oe++)if(Oe>=w.length){w.push(ae),Te=Oe;break}else if(w[Oe]===null){w[Oe]=ae,Te=Oe;break}if(Te===-1)break}let Ue=T[Te];Ue&&Ue.connect(ae)}}let Q=new A,ie=new A;function q(k,ce,ae){Q.setFromMatrixPosition(ce.matrixWorld),ie.setFromMatrixPosition(ae.matrixWorld);let Te=Q.distanceTo(ie),Ue=ce.projectionMatrix.elements,Oe=ae.projectionMatrix.elements,st=Ue[14]/(Ue[10]-1),He=Ue[14]/(Ue[10]+1),oe=(Ue[9]+1)/Ue[5],ee=(Ue[9]-1)/Ue[5],le=(Ue[8]-1)/Ue[0],J=(Oe[8]+1)/Oe[0],se=st*le,fe=st*J,ge=Te/(-le+J),we=ge*-le;if(ce.matrixWorld.decompose(k.position,k.quaternion,k.scale),k.translateX(we),k.translateZ(ge),k.matrixWorld.compose(k.position,k.quaternion,k.scale),k.matrixWorldInverse.copy(k.matrixWorld).invert(),Ue[10]===-1)k.projectionMatrix.copy(ce.projectionMatrix),k.projectionMatrixInverse.copy(ce.projectionMatrixInverse);else{let Se=st+ge,D=He+ge,Pe=se-we,Ze=fe+(Te-we),R=oe*He/D*Se,y=ee*He/D*Se;k.projectionMatrix.makePerspective(Pe,Ze,R,y,Se,D),k.projectionMatrixInverse.copy(k.projectionMatrix).invert()}}function Z(k,ce){ce===null?k.matrixWorld.copy(k.matrix):k.matrixWorld.multiplyMatrices(ce.matrixWorld,k.matrix),k.matrixWorldInverse.copy(k.matrixWorld).invert()}this.updateCamera=function(k){if(s===null)return;let ce=k.near,ae=k.far;p.texture!==null&&(p.depthNear>0&&(ce=p.depthNear),p.depthFar>0&&(ae=p.depthFar)),L.near=P.near=E.near=ce,L.far=P.far=E.far=ae,(X!==L.near||W!==L.far)&&(s.updateRenderState({depthNear:L.near,depthFar:L.far}),X=L.near,W=L.far),L.layers.mask=k.layers.mask|6,E.layers.mask=L.layers.mask&-5,P.layers.mask=L.layers.mask&-3;let Te=k.parent,Ue=L.cameras;Z(L,Te);for(let Oe=0;Oe<Ue.length;Oe++)Z(Ue[Oe],Te);Ue.length===2?q(L,E,P):L.projectionMatrix.copy(E.projectionMatrix),j(k,L,Te)};function j(k,ce,ae){ae===null?k.matrix.copy(ce.matrixWorld):(k.matrix.copy(ae.matrixWorld),k.matrix.invert(),k.matrix.multiply(ce.matrixWorld)),k.matrix.decompose(k.position,k.quaternion,k.scale),k.updateMatrixWorld(!0),k.projectionMatrix.copy(ce.projectionMatrix),k.projectionMatrixInverse.copy(ce.projectionMatrixInverse),k.isPerspectiveCamera&&(k.fov=pr*2*Math.atan(1/k.projectionMatrix.elements[5]),k.zoom=1)}this.getCamera=function(){return L},this.getFoveation=function(){if(!(u===null&&f===null))return c},this.setFoveation=function(k){c=k,u!==null&&(u.fixedFoveation=k),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=k)},this.hasDepthSensing=function(){return p.texture!==null},this.getDepthSensingMesh=function(){return p.getMesh(L)},this.getCameraTexture=function(k){return m[k]};let de=null;function Ge(k,ce){if(h=ce.getViewerPose(l||a),g=ce,h!==null){let ae=h.views;f!==null&&(e.setRenderTargetFramebuffer(v,f.framebuffer),e.setRenderTarget(v));let Te=!1;ae.length!==L.cameras.length&&(L.cameras.length=0,Te=!0);for(let He=0;He<ae.length;He++){let oe=ae[He],ee=null;if(f!==null)ee=f.getViewport(oe);else{let J=d.getViewSubImage(u,oe);ee=J.viewport,He===0&&(e.setRenderTargetTextures(v,J.colorTexture,J.depthStencilTexture),e.setRenderTarget(v))}let le=I[He];le===void 0&&(le=new Kt,le.layers.enable(He),le.viewport=new gt,I[He]=le),le.matrix.fromArray(oe.transform.matrix),le.matrix.decompose(le.position,le.quaternion,le.scale),le.projectionMatrix.fromArray(oe.projectionMatrix),le.projectionMatrixInverse.copy(le.projectionMatrix).invert(),le.viewport.set(ee.x,ee.y,ee.width,ee.height),He===0&&(L.matrix.copy(le.matrix),L.matrix.decompose(L.position,L.quaternion,L.scale)),Te===!0&&L.cameras.push(le)}let Ue=s.enabledFeatures;if(Ue&&Ue.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&x){d=i.getBinding();let He=d.getDepthInformation(ae[0]);He&&He.isValid&&He.texture&&p.init(He,s.renderState)}if(Ue&&Ue.includes("camera-access")&&x){e.state.unbindTexture(),d=i.getBinding();for(let He=0;He<ae.length;He++){let oe=ae[He].camera;if(oe){let ee=m[oe];ee||(ee=new pa,m[oe]=ee);let le=d.getCameraImage(oe);ee.sourceTexture=le}}}}for(let ae=0;ae<T.length;ae++){let Te=w[ae],Ue=T[ae];Te!==null&&Ue!==void 0&&Ue.update(Te,ce,l||a)}de&&de(k,ce),ce.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:ce}),g=null}let me=new pp;me.setAnimationLoop(Ge),this.setAnimationLoop=function(k){de=k},this.dispose=function(){}}},Ny=new rt,yp=new je;yp.set(-1,0,0,0,1,0,0,0,1);function Uy(n,e){function t(p,m){p.matrixAutoUpdate===!0&&p.updateMatrix(),m.value.copy(p.matrix)}function i(p,m){m.color.getRGB(p.fogColor.value,fu(n)),m.isFog?(p.fogNear.value=m.near,p.fogFar.value=m.far):m.isFogExp2&&(p.fogDensity.value=m.density)}function s(p,m,M,b,v){m.isNodeMaterial?m.uniformsNeedUpdate=!1:m.isMeshBasicMaterial?r(p,m):m.isMeshLambertMaterial?(r(p,m),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)):m.isMeshToonMaterial?(r(p,m),d(p,m)):m.isMeshPhongMaterial?(r(p,m),h(p,m),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)):m.isMeshStandardMaterial?(r(p,m),u(p,m),m.isMeshPhysicalMaterial&&f(p,m,v)):m.isMeshMatcapMaterial?(r(p,m),g(p,m)):m.isMeshDepthMaterial?r(p,m):m.isMeshDistanceMaterial?(r(p,m),x(p,m)):m.isMeshNormalMaterial?r(p,m):m.isLineBasicMaterial?(a(p,m),m.isLineDashedMaterial&&o(p,m)):m.isPointsMaterial?c(p,m,M,b):m.isSpriteMaterial?l(p,m):m.isShadowMaterial?(p.color.value.copy(m.color),p.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(p,m){p.opacity.value=m.opacity,m.color&&p.diffuse.value.copy(m.color),m.emissive&&p.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(p.map.value=m.map,t(m.map,p.mapTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.bumpMap&&(p.bumpMap.value=m.bumpMap,t(m.bumpMap,p.bumpMapTransform),p.bumpScale.value=m.bumpScale,m.side===Qt&&(p.bumpScale.value*=-1)),m.normalMap&&(p.normalMap.value=m.normalMap,t(m.normalMap,p.normalMapTransform),p.normalScale.value.copy(m.normalScale),m.side===Qt&&p.normalScale.value.negate()),m.displacementMap&&(p.displacementMap.value=m.displacementMap,t(m.displacementMap,p.displacementMapTransform),p.displacementScale.value=m.displacementScale,p.displacementBias.value=m.displacementBias),m.emissiveMap&&(p.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,p.emissiveMapTransform)),m.specularMap&&(p.specularMap.value=m.specularMap,t(m.specularMap,p.specularMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest);let M=e.get(m),b=M.envMap,v=M.envMapRotation;b&&(p.envMap.value=b,p.envMapRotation.value.setFromMatrix4(Ny.makeRotationFromEuler(v)).transpose(),b.isCubeTexture&&b.isRenderTargetTexture===!1&&p.envMapRotation.value.premultiply(yp),p.reflectivity.value=m.reflectivity,p.ior.value=m.ior,p.refractionRatio.value=m.refractionRatio),m.lightMap&&(p.lightMap.value=m.lightMap,p.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,p.lightMapTransform)),m.aoMap&&(p.aoMap.value=m.aoMap,p.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,p.aoMapTransform))}function a(p,m){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,m.map&&(p.map.value=m.map,t(m.map,p.mapTransform))}function o(p,m){p.dashSize.value=m.dashSize,p.totalSize.value=m.dashSize+m.gapSize,p.scale.value=m.scale}function c(p,m,M,b){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,p.size.value=m.size*M,p.scale.value=b*.5,m.map&&(p.map.value=m.map,t(m.map,p.uvTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest)}function l(p,m){p.diffuse.value.copy(m.color),p.opacity.value=m.opacity,p.rotation.value=m.rotation,m.map&&(p.map.value=m.map,t(m.map,p.mapTransform)),m.alphaMap&&(p.alphaMap.value=m.alphaMap,t(m.alphaMap,p.alphaMapTransform)),m.alphaTest>0&&(p.alphaTest.value=m.alphaTest)}function h(p,m){p.specular.value.copy(m.specular),p.shininess.value=Math.max(m.shininess,1e-4)}function d(p,m){m.gradientMap&&(p.gradientMap.value=m.gradientMap)}function u(p,m){p.metalness.value=m.metalness,m.metalnessMap&&(p.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,p.metalnessMapTransform)),p.roughness.value=m.roughness,m.roughnessMap&&(p.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,p.roughnessMapTransform)),m.envMap&&(p.envMapIntensity.value=m.envMapIntensity)}function f(p,m,M){p.ior.value=m.ior,m.sheen>0&&(p.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),p.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(p.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,p.sheenColorMapTransform)),m.sheenRoughnessMap&&(p.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,p.sheenRoughnessMapTransform))),m.clearcoat>0&&(p.clearcoat.value=m.clearcoat,p.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(p.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,p.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(p.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,p.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(p.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,p.clearcoatNormalMapTransform),p.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===Qt&&p.clearcoatNormalScale.value.negate())),m.dispersion>0&&(p.dispersion.value=m.dispersion),m.iridescence>0&&(p.iridescence.value=m.iridescence,p.iridescenceIOR.value=m.iridescenceIOR,p.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],p.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(p.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,p.iridescenceMapTransform)),m.iridescenceThicknessMap&&(p.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,p.iridescenceThicknessMapTransform))),m.transmission>0&&(p.transmission.value=m.transmission,p.transmissionSamplerMap.value=M.texture,p.transmissionSamplerSize.value.set(M.width,M.height),m.transmissionMap&&(p.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,p.transmissionMapTransform)),p.thickness.value=m.thickness,m.thicknessMap&&(p.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,p.thicknessMapTransform)),p.attenuationDistance.value=m.attenuationDistance,p.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(p.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(p.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,p.anisotropyMapTransform))),p.specularIntensity.value=m.specularIntensity,p.specularColor.value.copy(m.specularColor),m.specularColorMap&&(p.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,p.specularColorMapTransform)),m.specularIntensityMap&&(p.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,p.specularIntensityMapTransform))}function g(p,m){m.matcap&&(p.matcap.value=m.matcap)}function x(p,m){let M=e.get(m).light;p.referencePosition.value.setFromMatrixPosition(M.matrixWorld),p.nearDistance.value=M.shadow.camera.near,p.farDistance.value=M.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:s}}function Fy(n,e,t,i){let s={},r={},a=[],o=n.getParameter(n.MAX_UNIFORM_BUFFER_BINDINGS);function c(v,T){let w=T.program;i.uniformBlockBinding(v,w)}function l(v,T){let w=s[v.id];w===void 0&&(p(v),w=h(v),s[v.id]=w,v.addEventListener("dispose",M));let C=T.program;i.updateUBOMapping(v,C);let _=e.render.frame;r[v.id]!==_&&(u(v),r[v.id]=_)}function h(v){let T=d();v.__bindingPointIndex=T;let w=n.createBuffer(),C=v.__size,_=v.usage;return n.bindBuffer(n.UNIFORM_BUFFER,w),n.bufferData(n.UNIFORM_BUFFER,C,_),n.bindBuffer(n.UNIFORM_BUFFER,null),n.bindBufferBase(n.UNIFORM_BUFFER,T,w),w}function d(){for(let v=0;v<o;v++)if(a.indexOf(v)===-1)return a.push(v),v;return $e("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function u(v){let T=s[v.id],w=v.uniforms,C=v.__cache;n.bindBuffer(n.UNIFORM_BUFFER,T);for(let _=0,E=w.length;_<E;_++){let P=w[_];if(Array.isArray(P))for(let I=0,L=P.length;I<L;I++)f(P[I],_,I,C);else f(P,_,0,C)}n.bindBuffer(n.UNIFORM_BUFFER,null)}function f(v,T,w,C){if(x(v,T,w,C)===!0){let _=v.__offset,E=v.value;if(Array.isArray(E)){let P=0;for(let I=0;I<E.length;I++){let L=E[I],X=m(L);g(L,v.__data,P),typeof L!="number"&&typeof L!="boolean"&&!L.isMatrix3&&!ArrayBuffer.isView(L)&&(P+=X.storage/Float32Array.BYTES_PER_ELEMENT)}}else g(E,v.__data,0);n.bufferSubData(n.UNIFORM_BUFFER,_,v.__data)}}function g(v,T,w){typeof v=="number"||typeof v=="boolean"?T[0]=v:v.isMatrix3?(T[0]=v.elements[0],T[1]=v.elements[1],T[2]=v.elements[2],T[3]=0,T[4]=v.elements[3],T[5]=v.elements[4],T[6]=v.elements[5],T[7]=0,T[8]=v.elements[6],T[9]=v.elements[7],T[10]=v.elements[8],T[11]=0):ArrayBuffer.isView(v)?T.set(new v.constructor(v.buffer,v.byteOffset,T.length)):v.toArray(T,w)}function x(v,T,w,C){let _=v.value,E=T+"_"+w;if(C[E]===void 0)return typeof _=="number"||typeof _=="boolean"?C[E]=_:ArrayBuffer.isView(_)?C[E]=_.slice():C[E]=_.clone(),!0;{let P=C[E];if(typeof _=="number"||typeof _=="boolean"){if(P!==_)return C[E]=_,!0}else{if(ArrayBuffer.isView(_))return!0;if(P.equals(_)===!1)return P.copy(_),!0}}return!1}function p(v){let T=v.uniforms,w=0,C=16;for(let E=0,P=T.length;E<P;E++){let I=Array.isArray(T[E])?T[E]:[T[E]];for(let L=0,X=I.length;L<X;L++){let W=I[L],U=Array.isArray(W.value)?W.value:[W.value];for(let z=0,H=U.length;z<H;z++){let Q=U[z],ie=m(Q),q=w%C,Z=q%ie.boundary,j=q+Z;w+=Z,j!==0&&C-j<ie.storage&&(w+=C-j),W.__data=new Float32Array(ie.storage/Float32Array.BYTES_PER_ELEMENT),W.__offset=w,w+=ie.storage}}}let _=w%C;return _>0&&(w+=C-_),v.__size=w,v.__cache={},this}function m(v){let T={boundary:0,storage:0};return typeof v=="number"||typeof v=="boolean"?(T.boundary=4,T.storage=4):v.isVector2?(T.boundary=8,T.storage=8):v.isVector3||v.isColor?(T.boundary=16,T.storage=12):v.isVector4?(T.boundary=16,T.storage=16):v.isMatrix3?(T.boundary=48,T.storage=48):v.isMatrix4?(T.boundary=64,T.storage=64):v.isTexture?Ye("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(v)?(T.boundary=16,T.storage=v.byteLength):Ye("WebGLRenderer: Unsupported uniform value type.",v),T}function M(v){let T=v.target;T.removeEventListener("dispose",M);let w=a.indexOf(T.__bindingPointIndex);a.splice(w,1),n.deleteBuffer(s[T.id]),delete s[T.id],delete r[T.id]}function b(){for(let v in s)n.deleteBuffer(s[v]);a=[],s={},r={}}return{bind:c,update:l,dispose:b}}var Oy=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),vn=null;function By(){return vn===null&&(vn=new Dn(Oy,16,16,us,ei),vn.name="DFG_LUT",vn.minFilter=jt,vn.magFilter=jt,vn.wrapS=mn,vn.wrapT=mn,vn.generateMipmaps=!1,vn.needsUpdate=!0),vn}var Cc=class{constructor(e={}){let{canvas:t=Uf(),context:i=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:d=!1,reversedDepthBuffer:u=!1,outputBufferType:f=li}=e;this.isWebGLRenderer=!0;let g;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");g=i.getContextAttributes().alpha}else g=a;let x=f,p=new Set([Yl,ql,Xl]),m=new Set([li,an,Rr,hs,Vl,Gl]),M=new Uint32Array(4),b=new Int32Array(4),v=new A,T=null,w=null,C=[],_=[],E=null;this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=rn,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let P=this,I=!1,L=null,X=null,W=null,U=null;this._outputColorSpace=Ft;let z=0,H=0,Q=null,ie=-1,q=null,Z=new gt,j=new gt,de=null,Ge=new Le(0),me=0,k=t.width,ce=t.height,ae=1,Te=null,Ue=null,Oe=new gt(0,0,k,ce),st=new gt(0,0,k,ce),He=!1,oe=new vr,ee=!1,le=!1,J=new rt,se=new A,fe=new gt,ge={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},we=!1;function Se(){return Q===null?ae:1}let D=i;function Pe(S,O){return t.getContext(S,O)}try{let S={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:h,failIfMajorPerformanceCaveat:d};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${"185"}`),t.addEventListener("webglcontextlost",Dt,!1),t.addEventListener("webglcontextrestored",St,!1),t.addEventListener("webglcontextcreationerror",hn,!1),D===null){let O="webgl2";if(D=Pe(O,S),D===null)throw Pe(O)?new Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes."):new Error("THREE.WebGLRenderer: Error creating WebGL context.")}}catch(S){throw $e("WebGLRenderer: "+S.message),S}let Ze,R,y,F,B,Y,pe,_e,K,ne,Me,ke,ve,xe,Be,Xe,Je,N,be,re,Ee,Ce,he;function Ve(){Ze=new Xx(D),Ze.init(),Ee=new Iy(D,Ze),R=new Ox(D,Ze,e,Ee),y=new Cy(D,Ze),R.reversedDepthBuffer&&u&&y.buffers.depth.setReversed(!0),X=D.createFramebuffer(),W=D.createFramebuffer(),U=D.createFramebuffer(),F=new $x(D),B=new my,Y=new Py(D,Ze,y,B,R,Ee,F),pe=new Wx(P),_e=new jg(D),Ce=new Ux(D,_e),K=new qx(D,_e,F,Ce),ne=new Jx(D,K,_e,Ce,F),N=new Zx(D,R,Y),Be=new Bx(B),Me=new py(P,pe,Ze,R,Ce,Be),ke=new Uy(P,B),ve=new _y,xe=new Sy(Ze),Je=new Nx(P,pe,y,ne,g,c),Xe=new Ry(P,ne,R),he=new Fy(D,F,R,y),be=new Fx(D,Ze,F),re=new Yx(D,Ze,F),F.programs=Me.programs,P.capabilities=R,P.extensions=Ze,P.properties=B,P.renderLists=ve,P.shadowMap=Xe,P.state=y,P.info=F}Ve(),x!==li&&(E=new jx(x,t.width,t.height,o,s,r));let Fe=new Fu(P,D);this.xr=Fe,this.getContext=function(){return D},this.getContextAttributes=function(){return D.getContextAttributes()},this.forceContextLoss=function(){let S=Ze.get("WEBGL_lose_context");S&&S.loseContext()},this.forceContextRestore=function(){let S=Ze.get("WEBGL_lose_context");S&&S.restoreContext()},this.getPixelRatio=function(){return ae},this.setPixelRatio=function(S){S!==void 0&&(ae=S,this.setSize(k,ce,!1))},this.getSize=function(S){return S.set(k,ce)},this.setSize=function(S,O,$=!0){if(Fe.isPresenting){Ye("WebGLRenderer: Can't change size while VR device is presenting.");return}k=S,ce=O,t.width=Math.floor(S*ae),t.height=Math.floor(O*ae),$===!0&&(t.style.width=S+"px",t.style.height=O+"px"),E!==null&&E.setSize(t.width,t.height),this.setViewport(0,0,S,O)},this.getDrawingBufferSize=function(S){return S.set(k*ae,ce*ae).floor()},this.setDrawingBufferSize=function(S,O,$){k=S,ce=O,ae=$,t.width=Math.floor(S*$),t.height=Math.floor(O*$),this.setViewport(0,0,S,O)},this.setEffects=function(S){if(x===li){$e("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(S){for(let O=0;O<S.length;O++)if(S[O].isOutputPass===!0){Ye("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}E.setEffects(S||[])},this.getCurrentViewport=function(S){return S.copy(Z)},this.getViewport=function(S){return S.copy(Oe)},this.setViewport=function(S,O,$,V){S.isVector4?Oe.set(S.x,S.y,S.z,S.w):Oe.set(S,O,$,V),y.viewport(Z.copy(Oe).multiplyScalar(ae).round())},this.getScissor=function(S){return S.copy(st)},this.setScissor=function(S,O,$,V){S.isVector4?st.set(S.x,S.y,S.z,S.w):st.set(S,O,$,V),y.scissor(j.copy(st).multiplyScalar(ae).round())},this.getScissorTest=function(){return He},this.setScissorTest=function(S){y.setScissorTest(He=S)},this.setOpaqueSort=function(S){Te=S},this.setTransparentSort=function(S){Ue=S},this.getClearColor=function(S){return S.copy(Je.getClearColor())},this.setClearColor=function(){Je.setClearColor(...arguments)},this.getClearAlpha=function(){return Je.getClearAlpha()},this.setClearAlpha=function(){Je.setClearAlpha(...arguments)},this.clear=function(S=!0,O=!0,$=!0){let V=0;if(S){let G=!1;if(Q!==null){let Re=Q.texture.format;G=p.has(Re)}if(G){let Re=Q.texture.type,De=m.has(Re),Ae=Je.getClearColor(),ze=Je.getClearAlpha(),We=Ae.r,it=Ae.g,lt=Ae.b;De?(M[0]=We,M[1]=it,M[2]=lt,M[3]=ze,D.clearBufferuiv(D.COLOR,0,M)):(b[0]=We,b[1]=it,b[2]=lt,b[3]=ze,D.clearBufferiv(D.COLOR,0,b))}else V|=D.COLOR_BUFFER_BIT}O&&(V|=D.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),$&&(V|=D.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),V!==0&&D.clear(V)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(S){S.setRenderer(this),L=S},this.dispose=function(){t.removeEventListener("webglcontextlost",Dt,!1),t.removeEventListener("webglcontextrestored",St,!1),t.removeEventListener("webglcontextcreationerror",hn,!1),Je.dispose(),ve.dispose(),xe.dispose(),B.dispose(),pe.dispose(),ne.dispose(),Ce.dispose(),he.dispose(),Me.dispose(),Fe.dispose(),Fe.removeEventListener("sessionstart",fd),Fe.removeEventListener("sessionend",pd),vs.stop()};function Dt(S){S.preventDefault(),ra("WebGLRenderer: Context Lost."),I=!0}function St(){ra("WebGLRenderer: Context Restored."),I=!1;let S=F.autoReset,O=Xe.enabled,$=Xe.autoUpdate,V=Xe.needsUpdate,G=Xe.type;Ve(),F.autoReset=S,Xe.enabled=O,Xe.autoUpdate=$,Xe.needsUpdate=V,Xe.type=G}function hn(S){$e("WebGLRenderer: A WebGL context could not be created. Reason: ",S.statusMessage)}function un(S){let O=S.target;O.removeEventListener("dispose",un),ym(O)}function ym(S){Mm(S),B.remove(S)}function Mm(S){let O=B.get(S).programs;O!==void 0&&(O.forEach(function($){Me.releaseProgram($)}),S.isShaderMaterial&&Me.releaseShaderCache(S))}this.renderBufferDirect=function(S,O,$,V,G,Re){O===null&&(O=ge);let De=G.isMesh&&G.matrixWorld.determinantAffine()<0,Ae=Em(S,O,$,V,G);y.setMaterial(V,De);let ze=$.index,We=1;if(V.wireframe===!0){if(ze=K.getWireframeAttribute($),ze===void 0)return;We=2}let it=$.drawRange,lt=$.attributes.position,qe=it.start*We,vt=(it.start+it.count)*We;Re!==null&&(qe=Math.max(qe,Re.start*We),vt=Math.min(vt,(Re.start+Re.count)*We)),ze!==null?(qe=Math.max(qe,0),vt=Math.min(vt,ze.count)):lt!=null&&(qe=Math.max(qe,0),vt=Math.min(vt,lt.count));let Nt=vt-qe;if(Nt<0||Nt===1/0)return;Ce.setup(G,V,Ae,$,ze);let Lt,Mt=be;if(ze!==null&&(Lt=_e.get(ze),Mt=re,Mt.setIndex(Lt)),G.isMesh)V.wireframe===!0?(y.setLineWidth(V.wireframeLinewidth*Se()),Mt.setMode(D.LINES)):Mt.setMode(D.TRIANGLES);else if(G.isLine){let ri=V.linewidth;ri===void 0&&(ri=1),y.setLineWidth(ri*Se()),G.isLineSegments?Mt.setMode(D.LINES):G.isLineLoop?Mt.setMode(D.LINE_LOOP):Mt.setMode(D.LINE_STRIP)}else G.isPoints?Mt.setMode(D.POINTS):G.isSprite&&Mt.setMode(D.TRIANGLES);if(G.isBatchedMesh)if(Ze.get("WEBGL_multi_draw"))Mt.renderMultiDraw(G._multiDrawStarts,G._multiDrawCounts,G._multiDrawCount);else{let ri=G._multiDrawStarts,Ie=G._multiDrawCounts,Si=G._multiDrawCount,dt=ze?_e.get(ze).bytesPerElement:1,Fi=B.get(V).currentProgram.getUniforms();for(let dn=0;dn<Si;dn++)Fi.setValue(D,"_gl_DrawID",dn),Mt.render(ri[dn]/dt,Ie[dn])}else if(G.isInstancedMesh)Mt.renderInstances(qe,Nt,G.count);else if($.isInstancedBufferGeometry){let ri=$._maxInstanceCount!==void 0?$._maxInstanceCount:1/0,Ie=Math.min($.instanceCount,ri);Mt.renderInstances(qe,Nt,Ie)}else Mt.render(qe,Nt)};function dd(S,O,$){S.transparent===!0&&S.side===xi&&S.forceSinglePass===!1?(S.side=Qt,S.needsUpdate=!0,mo(S,O,$),S.side=ji,S.needsUpdate=!0,mo(S,O,$),S.side=xi):mo(S,O,$)}this.compile=function(S,O,$=null){$===null&&($=S),w=xe.get($),w.init(O),_.push(w),$.traverseVisible(function(G){G.isLight&&G.layers.test(O.layers)&&(w.pushLight(G),G.castShadow&&w.pushShadow(G))}),S!==$&&S.traverseVisible(function(G){G.isLight&&G.layers.test(O.layers)&&(w.pushLight(G),G.castShadow&&w.pushShadow(G))}),w.setupLights();let V=new Set;return S.traverse(function(G){if(!(G.isMesh||G.isPoints||G.isLine||G.isSprite))return;let Re=G.material;if(Re)if(Array.isArray(Re))for(let De=0;De<Re.length;De++){let Ae=Re[De];dd(Ae,$,G),V.add(Ae)}else dd(Re,$,G),V.add(Re)}),w=_.pop(),V},this.compileAsync=function(S,O,$=null){let V=this.compile(S,O,$);return new Promise(G=>{function Re(){if(V.forEach(function(De){B.get(De).currentProgram.isReady()&&V.delete(De)}),V.size===0){G(S);return}setTimeout(Re,10)}Ze.get("KHR_parallel_shader_compile")!==null?Re():setTimeout(Re,10)})};let lh=null;function bm(S){lh&&lh(S)}function fd(){vs.stop()}function pd(){vs.start()}let vs=new pp;vs.setAnimationLoop(bm),typeof self<"u"&&vs.setContext(self),this.setAnimationLoop=function(S){lh=S,Fe.setAnimationLoop(S),S===null?vs.stop():vs.start()},Fe.addEventListener("sessionstart",fd),Fe.addEventListener("sessionend",pd),this.render=function(S,O){if(O!==void 0&&O.isCamera!==!0){$e("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(I===!0)return;L!==null&&L.renderStart(S,O);let $=Fe.enabled===!0&&Fe.isPresenting===!0,V=E!==null&&(Q===null||$)&&E.begin(P,Q);if(S.matrixWorldAutoUpdate===!0&&S.updateMatrixWorld(),O.parent===null&&O.matrixWorldAutoUpdate===!0&&O.updateMatrixWorld(),Fe.enabled===!0&&Fe.isPresenting===!0&&(E===null||E.isCompositing()===!1)&&(Fe.cameraAutoUpdate===!0&&Fe.updateCamera(O),O=Fe.getCamera()),S.isScene===!0&&S.onBeforeRender(P,S,O,Q),w=xe.get(S,_.length),w.init(O),w.state.textureUnits=Y.getTextureUnits(),_.push(w),J.multiplyMatrices(O.projectionMatrix,O.matrixWorldInverse),oe.setFromProjectionMatrix(J,Ki,O.reversedDepth),le=this.localClippingEnabled,ee=Be.init(this.clippingPlanes,le),T=ve.get(S,C.length),T.init(),C.push(T),Fe.enabled===!0&&Fe.isPresenting===!0){let De=P.xr.getDepthSensingMesh();De!==null&&ch(De,O,-1/0,P.sortObjects)}ch(S,O,0,P.sortObjects),T.finish(),P.sortObjects===!0&&T.sort(Te,Ue,O.reversedDepth),we=Fe.enabled===!1||Fe.isPresenting===!1||Fe.hasDepthSensing()===!1,we&&Je.addToRenderList(T,S),this.info.render.frame++,this.info.autoReset===!0&&this.info.reset(),ee===!0&&Be.beginShadows();let G=w.state.shadowsArray;if(Xe.render(G,S,O),ee===!0&&Be.endShadows(),(V&&E.hasRenderPass())===!1){let De=T.opaque,Ae=T.transmissive;if(w.setupLights(),O.isArrayCamera){let ze=O.cameras;if(Ae.length>0)for(let We=0,it=ze.length;We<it;We++){let lt=ze[We];gd(De,Ae,S,lt)}we&&Je.render(S);for(let We=0,it=ze.length;We<it;We++){let lt=ze[We];md(T,S,lt,lt.viewport)}}else Ae.length>0&&gd(De,Ae,S,O),we&&Je.render(S),md(T,S,O)}Q!==null&&H===0&&(Y.updateMultisampleRenderTarget(Q),Y.updateRenderTargetMipmap(Q)),V&&E.end(P),S.isScene===!0&&S.onAfterRender(P,S,O),Ce.resetDefaultState(),ie=-1,q=null,_.pop(),_.length>0?(w=_[_.length-1],Y.setTextureUnits(w.state.textureUnits),ee===!0&&Be.setGlobalState(P.clippingPlanes,w.state.camera)):w=null,C.pop(),C.length>0?T=C[C.length-1]:T=null,L!==null&&L.renderEnd()};function ch(S,O,$,V){if(S.visible===!1)return;if(S.layers.test(O.layers)){if(S.isGroup)$=S.renderOrder;else if(S.isLOD)S.autoUpdate===!0&&S.update(O);else if(S.isLightProbeGrid)w.pushLightProbeGrid(S);else if(S.isLight)w.pushLight(S),S.castShadow&&w.pushShadow(S);else if(S.isSprite){if(!S.frustumCulled||oe.intersectsSprite(S)){V&&fe.setFromMatrixPosition(S.matrixWorld).applyMatrix4(J);let De=ne.update(S),Ae=S.material;Ae.visible&&T.push(S,De,Ae,$,fe.z,null)}}else if((S.isMesh||S.isLine||S.isPoints)&&(!S.frustumCulled||oe.intersectsObject(S))){let De=ne.update(S),Ae=S.material;if(V&&(S.boundingSphere!==void 0?(S.boundingSphere===null&&S.computeBoundingSphere(),fe.copy(S.boundingSphere.center)):(De.boundingSphere===null&&De.computeBoundingSphere(),fe.copy(De.boundingSphere.center)),fe.applyMatrix4(S.matrixWorld).applyMatrix4(J)),Array.isArray(Ae)){let ze=De.groups;for(let We=0,it=ze.length;We<it;We++){let lt=ze[We],qe=Ae[lt.materialIndex];qe&&qe.visible&&T.push(S,De,qe,$,fe.z,lt)}}else Ae.visible&&T.push(S,De,Ae,$,fe.z,null)}}let Re=S.children;for(let De=0,Ae=Re.length;De<Ae;De++)ch(Re[De],O,$,V)}function md(S,O,$,V){let{opaque:G,transmissive:Re,transparent:De}=S;w.setupLightsView($),ee===!0&&Be.setGlobalState(P.clippingPlanes,$),V&&y.viewport(Z.copy(V)),G.length>0&&po(G,O,$),Re.length>0&&po(Re,O,$),De.length>0&&po(De,O,$),y.buffers.depth.setTest(!0),y.buffers.depth.setMask(!0),y.buffers.color.setMask(!0),y.setPolygonOffset(!1)}function gd(S,O,$,V){if(($.isScene===!0?$.overrideMaterial:null)!==null)return;if(w.state.transmissionRenderTarget[V.id]===void 0){let qe=Ze.has("EXT_color_buffer_half_float")||Ze.has("EXT_color_buffer_float");w.state.transmissionRenderTarget[V.id]=new Ht(1,1,{generateMipmaps:!0,type:qe?ei:li,minFilter:cs,samples:Math.max(4,R.samples),stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:ht.workingColorSpace})}let Re=w.state.transmissionRenderTarget[V.id],De=V.viewport||Z;Re.setSize(De.z*P.transmissionResolutionScale,De.w*P.transmissionResolutionScale);let Ae=P.getRenderTarget(),ze=P.getActiveCubeFace(),We=P.getActiveMipmapLevel();P.setRenderTarget(Re),P.getClearColor(Ge),me=P.getClearAlpha(),me<1&&P.setClearColor(16777215,.5),P.clear(),we&&Je.render($);let it=P.toneMapping;P.toneMapping=rn;let lt=V.viewport;if(V.viewport!==void 0&&(V.viewport=void 0),w.setupLightsView(V),ee===!0&&Be.setGlobalState(P.clippingPlanes,V),po(S,$,V),Y.updateMultisampleRenderTarget(Re),Y.updateRenderTargetMipmap(Re),Ze.has("WEBGL_multisampled_render_to_texture")===!1){let qe=!1;for(let vt=0,Nt=O.length;vt<Nt;vt++){let Lt=O[vt],{object:Mt,geometry:ri,material:Ie,group:Si}=Lt;if(Ie.side===xi&&Mt.layers.test(V.layers)){let dt=Ie.side;Ie.side=Qt,Ie.needsUpdate=!0,_d(Mt,$,V,ri,Ie,Si),Ie.side=dt,Ie.needsUpdate=!0,qe=!0}}qe===!0&&(Y.updateMultisampleRenderTarget(Re),Y.updateRenderTargetMipmap(Re))}P.setRenderTarget(Ae,ze,We),P.setClearColor(Ge,me),lt!==void 0&&(V.viewport=lt),P.toneMapping=it}function po(S,O,$){let V=O.isScene===!0?O.overrideMaterial:null;for(let G=0,Re=S.length;G<Re;G++){let De=S[G],{object:Ae,geometry:ze,group:We}=De,it=De.material;it.allowOverride===!0&&V!==null&&(it=V),Ae.layers.test($.layers)&&_d(Ae,O,$,ze,it,We)}}function _d(S,O,$,V,G,Re){S.onBeforeRender(P,O,$,V,G,Re),S.modelViewMatrix.multiplyMatrices($.matrixWorldInverse,S.matrixWorld),S.normalMatrix.getNormalMatrix(S.modelViewMatrix),G.onBeforeRender(P,O,$,V,S,Re),G.transparent===!0&&G.side===xi&&G.forceSinglePass===!1?(G.side=Qt,G.needsUpdate=!0,P.renderBufferDirect($,O,V,G,S,Re),G.side=ji,G.needsUpdate=!0,P.renderBufferDirect($,O,V,G,S,Re),G.side=xi):P.renderBufferDirect($,O,V,G,S,Re),S.onAfterRender(P,O,$,V,G,Re)}function mo(S,O,$){O.isScene!==!0&&(O=ge);let V=B.get(S),G=w.state.lights,Re=w.state.shadowsArray,De=G.state.version,Ae=Me.getParameters(S,G.state,Re,O,$,w.state.lightProbeGridArray),ze=Me.getProgramCacheKey(Ae),We=V.programs;V.environment=S.isMeshStandardMaterial||S.isMeshLambertMaterial||S.isMeshPhongMaterial?O.environment:null,V.fog=O.fog;let it=S.isMeshStandardMaterial||S.isMeshLambertMaterial&&!S.envMap||S.isMeshPhongMaterial&&!S.envMap;V.envMap=pe.get(S.envMap||V.environment,it),V.envMapRotation=V.environment!==null&&S.envMap===null?O.environmentRotation:S.envMapRotation,We===void 0&&(S.addEventListener("dispose",un),We=new Map,V.programs=We);let lt=We.get(ze);if(lt!==void 0){if(V.currentProgram===lt&&V.lightsStateVersion===De)return vd(S,Ae),lt}else Ae.uniforms=Me.getUniforms(S),L!==null&&S.isNodeMaterial&&L.build(S,$,Ae),S.onBeforeCompile(Ae,P),lt=Me.acquireProgram(Ae,ze),We.set(ze,lt),V.uniforms=Ae.uniforms;let qe=V.uniforms;return(!S.isShaderMaterial&&!S.isRawShaderMaterial||S.clipping===!0)&&(qe.clippingPlanes=Be.uniform),vd(S,Ae),V.needsLights=Tm(S),V.lightsStateVersion=De,V.needsLights&&(qe.ambientLightColor.value=G.state.ambient,qe.lightProbe.value=G.state.probe,qe.directionalLights.value=G.state.directional,qe.directionalLightShadows.value=G.state.directionalShadow,qe.spotLights.value=G.state.spot,qe.spotLightShadows.value=G.state.spotShadow,qe.rectAreaLights.value=G.state.rectArea,qe.ltc_1.value=G.state.rectAreaLTC1,qe.ltc_2.value=G.state.rectAreaLTC2,qe.pointLights.value=G.state.point,qe.pointLightShadows.value=G.state.pointShadow,qe.hemisphereLights.value=G.state.hemi,qe.directionalShadowMatrix.value=G.state.directionalShadowMatrix,qe.spotLightMatrix.value=G.state.spotLightMatrix,qe.spotLightMap.value=G.state.spotLightMap,qe.pointShadowMatrix.value=G.state.pointShadowMatrix),V.lightProbeGrid=w.state.lightProbeGridArray.length>0,V.currentProgram=lt,V.uniformsList=null,lt}function xd(S){if(S.uniformsList===null){let O=S.currentProgram.getUniforms();S.uniformsList=Dr.seqWithValue(O.seq,S.uniforms)}return S.uniformsList}function vd(S,O){let $=B.get(S);$.outputColorSpace=O.outputColorSpace,$.batching=O.batching,$.batchingColor=O.batchingColor,$.instancing=O.instancing,$.instancingColor=O.instancingColor,$.instancingMorph=O.instancingMorph,$.skinning=O.skinning,$.morphTargets=O.morphTargets,$.morphNormals=O.morphNormals,$.morphColors=O.morphColors,$.morphTargetsCount=O.morphTargetsCount,$.numClippingPlanes=O.numClippingPlanes,$.numIntersection=O.numClipIntersection,$.vertexAlphas=O.vertexAlphas,$.vertexTangents=O.vertexTangents,$.toneMapping=O.toneMapping}function Sm(S,O){if(S.length===0)return null;if(S.length===1)return S[0].texture!==null?S[0]:null;v.setFromMatrixPosition(O.matrixWorld);for(let $=0,V=S.length;$<V;$++){let G=S[$];if(G.texture!==null&&G.boundingBox.containsPoint(v))return G}return null}function Em(S,O,$,V,G){O.isScene!==!0&&(O=ge),Y.resetTextureUnits();let Re=O.fog,De=V.isMeshStandardMaterial||V.isMeshLambertMaterial||V.isMeshPhongMaterial?O.environment:null,Ae=Q===null?P.outputColorSpace:Q.isXRRenderTarget===!0?Q.texture.colorSpace:ht.workingColorSpace,ze=V.isMeshStandardMaterial||V.isMeshLambertMaterial&&!V.envMap||V.isMeshPhongMaterial&&!V.envMap,We=pe.get(V.envMap||De,ze),it=V.vertexColors===!0&&!!$.attributes.color&&$.attributes.color.itemSize===4,lt=!!$.attributes.tangent&&(!!V.normalMap||V.anisotropy>0),qe=!!$.morphAttributes.position,vt=!!$.morphAttributes.normal,Nt=!!$.morphAttributes.color,Lt=rn;V.toneMapped&&(Q===null||Q.isXRRenderTarget===!0)&&(Lt=P.toneMapping);let Mt=$.morphAttributes.position||$.morphAttributes.normal||$.morphAttributes.color,ri=Mt!==void 0?Mt.length:0,Ie=B.get(V),Si=w.state.lights;if(ee===!0&&(le===!0||S!==q)){let Et=S===q&&V.id===ie;Be.setState(V,S,Et)}let dt=!1;V.version===Ie.__version?(Ie.needsLights&&Ie.lightsStateVersion!==Si.state.version||Ie.outputColorSpace!==Ae||G.isBatchedMesh&&Ie.batching===!1||!G.isBatchedMesh&&Ie.batching===!0||G.isBatchedMesh&&Ie.batchingColor===!0&&G.colorTexture===null||G.isBatchedMesh&&Ie.batchingColor===!1&&G.colorTexture!==null||G.isInstancedMesh&&Ie.instancing===!1||!G.isInstancedMesh&&Ie.instancing===!0||G.isSkinnedMesh&&Ie.skinning===!1||!G.isSkinnedMesh&&Ie.skinning===!0||G.isInstancedMesh&&Ie.instancingColor===!0&&G.instanceColor===null||G.isInstancedMesh&&Ie.instancingColor===!1&&G.instanceColor!==null||G.isInstancedMesh&&Ie.instancingMorph===!0&&G.morphTexture===null||G.isInstancedMesh&&Ie.instancingMorph===!1&&G.morphTexture!==null||Ie.envMap!==We||V.fog===!0&&Ie.fog!==Re||Ie.numClippingPlanes!==void 0&&(Ie.numClippingPlanes!==Be.numPlanes||Ie.numIntersection!==Be.numIntersection)||Ie.vertexAlphas!==it||Ie.vertexTangents!==lt||Ie.morphTargets!==qe||Ie.morphNormals!==vt||Ie.morphColors!==Nt||Ie.toneMapping!==Lt||Ie.morphTargetsCount!==ri||!!Ie.lightProbeGrid!=w.state.lightProbeGridArray.length>0)&&(dt=!0):(dt=!0,Ie.__version=V.version);let Fi=Ie.currentProgram;dt===!0&&(Fi=mo(V,O,G),L&&V.isNodeMaterial&&L.onUpdateProgram(V,Fi,Ie));let dn=!1,Gn=!1,Gs=!1,bt=Fi.getUniforms(),Ut=Ie.uniforms;if(y.useProgram(Fi.program)&&(dn=!0,Gn=!0,Gs=!0),V.id!==ie&&(ie=V.id,Gn=!0),Ie.needsLights){let Et=Sm(w.state.lightProbeGridArray,G);Ie.lightProbeGrid!==Et&&(Ie.lightProbeGrid=Et,Gn=!0)}if(dn||q!==S){y.buffers.depth.getReversed()&&S.reversedDepth!==!0&&(S._reversedDepth=!0,S.updateProjectionMatrix()),bt.setValue(D,"projectionMatrix",S.projectionMatrix),bt.setValue(D,"viewMatrix",S.matrixWorldInverse);let Xn=bt.map.cameraPosition;Xn!==void 0&&Xn.setValue(D,se.setFromMatrixPosition(S.matrixWorld)),R.logarithmicDepthBuffer&&bt.setValue(D,"logDepthBufFC",2/(Math.log(S.far+1)/Math.LN2)),(V.isMeshPhongMaterial||V.isMeshToonMaterial||V.isMeshLambertMaterial||V.isMeshBasicMaterial||V.isMeshStandardMaterial||V.isShaderMaterial)&&bt.setValue(D,"isOrthographic",S.isOrthographicCamera===!0),q!==S&&(q=S,Gn=!0,Gs=!0)}if(Ie.needsLights&&(Si.state.directionalShadowMap.length>0&&bt.setValue(D,"directionalShadowMap",Si.state.directionalShadowMap,Y),Si.state.spotShadowMap.length>0&&bt.setValue(D,"spotShadowMap",Si.state.spotShadowMap,Y),Si.state.pointShadowMap.length>0&&bt.setValue(D,"pointShadowMap",Si.state.pointShadowMap,Y)),G.isSkinnedMesh){bt.setOptional(D,G,"bindMatrix"),bt.setOptional(D,G,"bindMatrixInverse");let Et=G.skeleton;Et&&(Et.boneTexture===null&&Et.computeBoneTexture(),bt.setValue(D,"boneTexture",Et.boneTexture,Y))}G.isBatchedMesh&&(bt.setOptional(D,G,"batchingTexture"),bt.setValue(D,"batchingTexture",G._matricesTexture,Y),bt.setOptional(D,G,"batchingIdTexture"),bt.setValue(D,"batchingIdTexture",G._indirectTexture,Y),bt.setOptional(D,G,"batchingColorTexture"),G._colorsTexture!==null&&bt.setValue(D,"batchingColorTexture",G._colorsTexture,Y));let Wn=$.morphAttributes;if((Wn.position!==void 0||Wn.normal!==void 0||Wn.color!==void 0)&&N.update(G,$,Fi),(Gn||Ie.receiveShadow!==G.receiveShadow)&&(Ie.receiveShadow=G.receiveShadow,bt.setValue(D,"receiveShadow",G.receiveShadow)),(V.isMeshStandardMaterial||V.isMeshLambertMaterial||V.isMeshPhongMaterial)&&V.envMap===null&&O.environment!==null&&(Ut.envMapIntensity.value=O.environmentIntensity),Ut.dfgLUT!==void 0&&(Ut.dfgLUT.value=By()),Gn){if(bt.setValue(D,"toneMappingExposure",P.toneMappingExposure),Ie.needsLights&&wm(Ut,Gs),Re&&V.fog===!0&&ke.refreshFogUniforms(Ut,Re),ke.refreshMaterialUniforms(Ut,V,ae,ce,w.state.transmissionRenderTarget[S.id]),Ie.needsLights&&Ie.lightProbeGrid){let Et=Ie.lightProbeGrid;Ut.probesSH.value=Et.texture,Ut.probesMin.value.copy(Et.boundingBox.min),Ut.probesMax.value.copy(Et.boundingBox.max),Ut.probesResolution.value.copy(Et.resolution)}Dr.upload(D,xd(Ie),Ut,Y)}if(V.isShaderMaterial&&V.uniformsNeedUpdate===!0&&(Dr.upload(D,xd(Ie),Ut,Y),V.uniformsNeedUpdate=!1),V.isSpriteMaterial&&bt.setValue(D,"center",G.center),bt.setValue(D,"modelViewMatrix",G.modelViewMatrix),bt.setValue(D,"normalMatrix",G.normalMatrix),bt.setValue(D,"modelMatrix",G.matrixWorld),V.uniformsGroups!==void 0){let Et=V.uniformsGroups;for(let Xn=0,Ws=Et.length;Xn<Ws;Xn++){let yd=Et[Xn];he.update(yd,Fi),he.bind(yd,Fi)}}return Fi}function wm(S,O){S.ambientLightColor.needsUpdate=O,S.lightProbe.needsUpdate=O,S.directionalLights.needsUpdate=O,S.directionalLightShadows.needsUpdate=O,S.pointLights.needsUpdate=O,S.pointLightShadows.needsUpdate=O,S.spotLights.needsUpdate=O,S.spotLightShadows.needsUpdate=O,S.rectAreaLights.needsUpdate=O,S.hemisphereLights.needsUpdate=O}function Tm(S){return S.isMeshLambertMaterial||S.isMeshToonMaterial||S.isMeshPhongMaterial||S.isMeshStandardMaterial||S.isShadowMaterial||S.isShaderMaterial&&S.lights===!0}this.getActiveCubeFace=function(){return z},this.getActiveMipmapLevel=function(){return H},this.getRenderTarget=function(){return Q},this.setRenderTargetTextures=function(S,O,$){let V=B.get(S);V.__autoAllocateDepthBuffer=S.resolveDepthBuffer===!1,V.__autoAllocateDepthBuffer===!1&&(V.__useRenderToTexture=!1),B.get(S.texture).__webglTexture=O,B.get(S.depthTexture).__webglTexture=V.__autoAllocateDepthBuffer?void 0:$,V.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(S,O){let $=B.get(S);$.__webglFramebuffer=O,$.__useDefaultFramebuffer=O===void 0},this.setRenderTarget=function(S,O=0,$=0){Q=S,z=O,H=$;let V=null,G=!1,Re=!1;if(S){let Ae=B.get(S);if(Ae.__useDefaultFramebuffer!==void 0){y.bindFramebuffer(D.FRAMEBUFFER,Ae.__webglFramebuffer),Z.copy(S.viewport),j.copy(S.scissor),de=S.scissorTest,y.viewport(Z),y.scissor(j),y.setScissorTest(de),ie=-1;return}else if(Ae.__webglFramebuffer===void 0)Y.setupRenderTarget(S);else if(Ae.__hasExternalTextures)Y.rebindTextures(S,B.get(S.texture).__webglTexture,B.get(S.depthTexture).__webglTexture);else if(S.depthBuffer){let it=S.depthTexture;if(Ae.__boundDepthTexture!==it){if(it!==null&&B.has(it)&&(S.width!==it.image.width||S.height!==it.image.height))throw new Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");Y.setupDepthRenderbuffer(S)}}let ze=S.texture;(ze.isData3DTexture||ze.isDataArrayTexture||ze.isCompressedArrayTexture)&&(Re=!0);let We=B.get(S).__webglFramebuffer;S.isWebGLCubeRenderTarget?(Array.isArray(We[O])?V=We[O][$]:V=We[O],G=!0):S.samples>0&&Y.useMultisampledRTT(S)===!1?V=B.get(S).__webglMultisampledFramebuffer:Array.isArray(We)?V=We[$]:V=We,Z.copy(S.viewport),j.copy(S.scissor),de=S.scissorTest}else Z.copy(Oe).multiplyScalar(ae).floor(),j.copy(st).multiplyScalar(ae).floor(),de=He;if($!==0&&(V=X),y.bindFramebuffer(D.FRAMEBUFFER,V)&&y.drawBuffers(S,V),y.viewport(Z),y.scissor(j),y.setScissorTest(de),G){let Ae=B.get(S.texture);D.framebufferTexture2D(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_CUBE_MAP_POSITIVE_X+O,Ae.__webglTexture,$)}else if(Re){let Ae=O;for(let ze=0;ze<S.textures.length;ze++){let We=B.get(S.textures[ze]);D.framebufferTextureLayer(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0+ze,We.__webglTexture,$,Ae)}}else if(S!==null&&$!==0){let Ae=B.get(S.texture);D.framebufferTexture2D(D.FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Ae.__webglTexture,$)}ie=-1},this.readRenderTargetPixels=function(S,O,$,V,G,Re,De,Ae=0){if(!(S&&S.isWebGLRenderTarget)){$e("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ze=B.get(S).__webglFramebuffer;if(S.isWebGLCubeRenderTarget&&De!==void 0&&(ze=ze[De]),ze){y.bindFramebuffer(D.FRAMEBUFFER,ze);try{let We=S.textures[Ae],it=We.format,lt=We.type;if(S.textures.length>1&&D.readBuffer(D.COLOR_ATTACHMENT0+Ae),!R.textureFormatReadable(it)){$e("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!R.textureTypeReadable(lt)){$e("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}O>=0&&O<=S.width-V&&$>=0&&$<=S.height-G&&D.readPixels(O,$,V,G,Ee.convert(it),Ee.convert(lt),Re)}finally{let We=Q!==null?B.get(Q).__webglFramebuffer:null;y.bindFramebuffer(D.FRAMEBUFFER,We)}}},this.readRenderTargetPixelsAsync=async function(S,O,$,V,G,Re,De,Ae=0){if(!(S&&S.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ze=B.get(S).__webglFramebuffer;if(S.isWebGLCubeRenderTarget&&De!==void 0&&(ze=ze[De]),ze)if(O>=0&&O<=S.width-V&&$>=0&&$<=S.height-G){y.bindFramebuffer(D.FRAMEBUFFER,ze);let We=S.textures[Ae],it=We.format,lt=We.type;if(S.textures.length>1&&D.readBuffer(D.COLOR_ATTACHMENT0+Ae),!R.textureFormatReadable(it))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!R.textureTypeReadable(lt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");let qe=D.createBuffer();D.bindBuffer(D.PIXEL_PACK_BUFFER,qe),D.bufferData(D.PIXEL_PACK_BUFFER,Re.byteLength,D.STREAM_READ),D.readPixels(O,$,V,G,Ee.convert(it),Ee.convert(lt),0);let vt=Q!==null?B.get(Q).__webglFramebuffer:null;y.bindFramebuffer(D.FRAMEBUFFER,vt);let Nt=D.fenceSync(D.SYNC_GPU_COMMANDS_COMPLETE,0);return D.flush(),await Of(D,Nt,4),D.bindBuffer(D.PIXEL_PACK_BUFFER,qe),D.getBufferSubData(D.PIXEL_PACK_BUFFER,0,Re),D.deleteBuffer(qe),D.deleteSync(Nt),Re}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(S,O=null,$=0){let V=Math.pow(2,-$),G=Math.floor(S.image.width*V),Re=Math.floor(S.image.height*V),De=O!==null?O.x:0,Ae=O!==null?O.y:0;Y.setTexture2D(S,0),D.copyTexSubImage2D(D.TEXTURE_2D,$,0,0,De,Ae,G,Re),y.unbindTexture()},this.copyTextureToTexture=function(S,O,$=null,V=null,G=0,Re=0){let De,Ae,ze,We,it,lt,qe,vt,Nt,Lt=S.isCompressedTexture?S.mipmaps[Re]:S.image;if($!==null)De=$.max.x-$.min.x,Ae=$.max.y-$.min.y,ze=$.isBox3?$.max.z-$.min.z:1,We=$.min.x,it=$.min.y,lt=$.isBox3?$.min.z:0;else{let Ut=Math.pow(2,-G);De=Math.floor(Lt.width*Ut),Ae=Math.floor(Lt.height*Ut),S.isDataArrayTexture?ze=Lt.depth:S.isData3DTexture?ze=Math.floor(Lt.depth*Ut):ze=1,We=0,it=0,lt=0}V!==null?(qe=V.x,vt=V.y,Nt=V.z):(qe=0,vt=0,Nt=0);let Mt=Ee.convert(O.format),ri=Ee.convert(O.type),Ie;O.isData3DTexture?(Y.setTexture3D(O,0),Ie=D.TEXTURE_3D):O.isDataArrayTexture||O.isCompressedArrayTexture?(Y.setTexture2DArray(O,0),Ie=D.TEXTURE_2D_ARRAY):(Y.setTexture2D(O,0),Ie=D.TEXTURE_2D),y.activeTexture(D.TEXTURE0),y.pixelStorei(D.UNPACK_FLIP_Y_WEBGL,O.flipY),y.pixelStorei(D.UNPACK_PREMULTIPLY_ALPHA_WEBGL,O.premultiplyAlpha),y.pixelStorei(D.UNPACK_ALIGNMENT,O.unpackAlignment);let Si=y.getParameter(D.UNPACK_ROW_LENGTH),dt=y.getParameter(D.UNPACK_IMAGE_HEIGHT),Fi=y.getParameter(D.UNPACK_SKIP_PIXELS),dn=y.getParameter(D.UNPACK_SKIP_ROWS),Gn=y.getParameter(D.UNPACK_SKIP_IMAGES);y.pixelStorei(D.UNPACK_ROW_LENGTH,Lt.width),y.pixelStorei(D.UNPACK_IMAGE_HEIGHT,Lt.height),y.pixelStorei(D.UNPACK_SKIP_PIXELS,We),y.pixelStorei(D.UNPACK_SKIP_ROWS,it),y.pixelStorei(D.UNPACK_SKIP_IMAGES,lt);let Gs=S.isDataArrayTexture||S.isData3DTexture,bt=O.isDataArrayTexture||O.isData3DTexture;if(S.isDepthTexture){let Ut=B.get(S),Wn=B.get(O),Et=B.get(Ut.__renderTarget),Xn=B.get(Wn.__renderTarget);y.bindFramebuffer(D.READ_FRAMEBUFFER,Et.__webglFramebuffer),y.bindFramebuffer(D.DRAW_FRAMEBUFFER,Xn.__webglFramebuffer);for(let Ws=0;Ws<ze;Ws++)Gs&&(D.framebufferTextureLayer(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,B.get(S).__webglTexture,G,lt+Ws),D.framebufferTextureLayer(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,B.get(O).__webglTexture,Re,Nt+Ws)),D.blitFramebuffer(We,it,De,Ae,qe,vt,De,Ae,D.DEPTH_BUFFER_BIT,D.NEAREST);y.bindFramebuffer(D.READ_FRAMEBUFFER,null),y.bindFramebuffer(D.DRAW_FRAMEBUFFER,null)}else if(G!==0||S.isRenderTargetTexture||B.has(S)){let Ut=B.get(S),Wn=B.get(O);y.bindFramebuffer(D.READ_FRAMEBUFFER,W),y.bindFramebuffer(D.DRAW_FRAMEBUFFER,U);for(let Et=0;Et<ze;Et++)Gs?D.framebufferTextureLayer(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,Ut.__webglTexture,G,lt+Et):D.framebufferTexture2D(D.READ_FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Ut.__webglTexture,G),bt?D.framebufferTextureLayer(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,Wn.__webglTexture,Re,Nt+Et):D.framebufferTexture2D(D.DRAW_FRAMEBUFFER,D.COLOR_ATTACHMENT0,D.TEXTURE_2D,Wn.__webglTexture,Re),G!==0?D.blitFramebuffer(We,it,De,Ae,qe,vt,De,Ae,D.COLOR_BUFFER_BIT,D.NEAREST):bt?D.copyTexSubImage3D(Ie,Re,qe,vt,Nt+Et,We,it,De,Ae):D.copyTexSubImage2D(Ie,Re,qe,vt,We,it,De,Ae);y.bindFramebuffer(D.READ_FRAMEBUFFER,null),y.bindFramebuffer(D.DRAW_FRAMEBUFFER,null)}else bt?S.isDataTexture||S.isData3DTexture?D.texSubImage3D(Ie,Re,qe,vt,Nt,De,Ae,ze,Mt,ri,Lt.data):O.isCompressedArrayTexture?D.compressedTexSubImage3D(Ie,Re,qe,vt,Nt,De,Ae,ze,Mt,Lt.data):D.texSubImage3D(Ie,Re,qe,vt,Nt,De,Ae,ze,Mt,ri,Lt):S.isDataTexture?D.texSubImage2D(D.TEXTURE_2D,Re,qe,vt,De,Ae,Mt,ri,Lt.data):S.isCompressedTexture?D.compressedTexSubImage2D(D.TEXTURE_2D,Re,qe,vt,Lt.width,Lt.height,Mt,Lt.data):D.texSubImage2D(D.TEXTURE_2D,Re,qe,vt,De,Ae,Mt,ri,Lt);y.pixelStorei(D.UNPACK_ROW_LENGTH,Si),y.pixelStorei(D.UNPACK_IMAGE_HEIGHT,dt),y.pixelStorei(D.UNPACK_SKIP_PIXELS,Fi),y.pixelStorei(D.UNPACK_SKIP_ROWS,dn),y.pixelStorei(D.UNPACK_SKIP_IMAGES,Gn),Re===0&&O.generateMipmaps&&D.generateMipmap(Ie),y.unbindTexture()},this.initRenderTarget=function(S){B.get(S).__webglFramebuffer===void 0&&Y.setupRenderTarget(S)},this.initTexture=function(S){S.isCubeTexture?Y.setTextureCube(S,0):S.isData3DTexture?Y.setTexture3D(S,0):S.isDataArrayTexture||S.isCompressedArrayTexture?Y.setTexture2DArray(S,0):Y.setTexture2D(S,0),y.unbindTexture()},this.resetState=function(){z=0,H=0,Q=null,y.reset(),Ce.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Ki}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;let t=this.getContext();t.drawingBufferColorSpace=ht._getDrawingBufferColorSpace(e),t.unpackColorSpace=ht._getUnpackColorSpace()}};var Mp={type:"change"},Bu={type:"start"},Sp={type:"end"},Dc=new jn,bp=new Bi,zy=Math.cos(70*Vt.DEG2RAD),Xt=new A,yi=2*Math.PI,yt={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},Ou=1e-6,Lc=class extends Oa{constructor(e,t=null){super(e,t),this.state=yt.NONE,this.target=new A,this.cursor=new A,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:rs.ROTATE,MIDDLE:rs.DOLLY,RIGHT:rs.PAN},this.touches={ONE:as.ROTATE,TWO:as.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._cursorStyle="auto",this._domElementKeyEvents=null,this._lastPosition=new A,this._lastQuaternion=new Ai,this._lastTargetPosition=new A,this._quat=new Ai().setFromUnitVectors(e.up,new A(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new Tr,this._sphericalDelta=new Tr,this._scale=1,this._panOffset=new A,this._rotateStart=new te,this._rotateEnd=new te,this._rotateDelta=new te,this._panStart=new te,this._panEnd=new te,this._panDelta=new te,this._dollyStart=new te,this._dollyEnd=new te,this._dollyDelta=new te,this._dollyDirection=new A,this._mouse=new te,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=Hy.bind(this),this._onPointerDown=ky.bind(this),this._onPointerUp=Vy.bind(this),this._onContextMenu=Zy.bind(this),this._onMouseWheel=Xy.bind(this),this._onKeyDown=qy.bind(this),this._onTouchStart=Yy.bind(this),this._onTouchMove=$y.bind(this),this._onMouseDown=Gy.bind(this),this._onMouseMove=Wy.bind(this),this._interceptControlDown=Jy.bind(this),this._interceptControlUp=Ky.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}set cursorStyle(e){this._cursorStyle=e,e==="grab"?this.domElement.style.cursor="grab":this.domElement.style.cursor="auto"}get cursorStyle(){return this._cursorStyle}connect(e){super.connect(e),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction=""}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(e){e.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=e}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(Mp),this.update(),this.state=yt.NONE}pan(e,t){this._pan(e,t),this.update()}dollyIn(e){this._dollyIn(e),this.update()}dollyOut(e){this._dollyOut(e),this.update()}rotateLeft(e){this._rotateLeft(e),this.update()}rotateUp(e){this._rotateUp(e),this.update()}update(e=null){let t=this.object.position;Xt.copy(t).sub(this.target),Xt.applyQuaternion(this._quat),this._spherical.setFromVector3(Xt),this.autoRotate&&this.state===yt.NONE&&this._rotateLeft(this._getAutoRotationAngle(e)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let i=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(i)&&isFinite(s)&&(i<-Math.PI?i+=yi:i>Math.PI&&(i-=yi),s<-Math.PI?s+=yi:s>Math.PI&&(s-=yi),i<=s?this._spherical.theta=Math.max(i,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(i+s)/2?Math.max(i,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{let a=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=a!=this._spherical.radius}if(Xt.setFromSpherical(this._spherical),Xt.applyQuaternion(this._quatInverse),t.copy(this.target).add(Xt),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let a=null;if(this.object.isPerspectiveCamera){let o=Xt.length();a=this._clampDistance(o*this._scale);let c=o-a;this.object.position.addScaledVector(this._dollyDirection,c),this.object.updateMatrixWorld(),r=!!c}else if(this.object.isOrthographicCamera){let o=new A(this._mouse.x,this._mouse.y,0);o.unproject(this.object);let c=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=c!==this.object.zoom;let l=new A(this._mouse.x,this._mouse.y,0);l.unproject(this.object),this.object.position.sub(l).add(o),this.object.updateMatrixWorld(),a=Xt.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;a!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(a).add(this.object.position):(Dc.origin.copy(this.object.position),Dc.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Dc.direction))<zy?this.object.lookAt(this.target):(bp.setFromNormalAndCoplanarPoint(this.object.up,this.target),Dc.intersectPlane(bp,this.target))))}else if(this.object.isOrthographicCamera){let a=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),a!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>Ou||8*(1-this._lastQuaternion.dot(this.object.quaternion))>Ou||this._lastTargetPosition.distanceToSquared(this.target)>Ou?(this.dispatchEvent(Mp),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(e){return e!==null?yi/60*this.autoRotateSpeed*e:yi/60/60*this.autoRotateSpeed}_getZoomScale(e){let t=Math.abs(e*.01);return Math.pow(.95,this.zoomSpeed*t)}_rotateLeft(e){this._sphericalDelta.theta-=e}_rotateUp(e){this._sphericalDelta.phi-=e}_panLeft(e,t){Xt.setFromMatrixColumn(t,0),Xt.multiplyScalar(-e),this._panOffset.add(Xt)}_panUp(e,t){this.screenSpacePanning===!0?Xt.setFromMatrixColumn(t,1):(Xt.setFromMatrixColumn(t,0),Xt.crossVectors(this.object.up,Xt)),Xt.multiplyScalar(e),this._panOffset.add(Xt)}_pan(e,t){let i=this.domElement;if(this.object.isPerspectiveCamera){let s=this.object.position;Xt.copy(s).sub(this.target);let r=Xt.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*e*r/i.clientHeight,this.object.matrix),this._panUp(2*t*r/i.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(e*(this.object.right-this.object.left)/this.object.zoom/i.clientWidth,this.object.matrix),this._panUp(t*(this.object.top-this.object.bottom)/this.object.zoom/i.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(e,t){if(!this.zoomToCursor)return;this._performCursorZoom=!0;let i=this.domElement.getBoundingClientRect(),s=e-i.left,r=t-i.top,a=i.width,o=i.height;this._mouse.x=s/a*2-1,this._mouse.y=-(r/o)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(e){return Math.max(this.minDistance,Math.min(this.maxDistance,e))}_handleMouseDownRotate(e){this._rotateStart.set(e.clientX,e.clientY)}_handleMouseDownDolly(e){this._updateZoomParameters(e.clientX,e.clientX),this._dollyStart.set(e.clientX,e.clientY)}_handleMouseDownPan(e){this._panStart.set(e.clientX,e.clientY)}_handleMouseMoveRotate(e){this._rotateEnd.set(e.clientX,e.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let t=this.domElement;this._rotateLeft(yi*this._rotateDelta.x/t.clientHeight),this._rotateUp(yi*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(e){this._dollyEnd.set(e.clientX,e.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(e){this._panEnd.set(e.clientX,e.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(e){this._updateZoomParameters(e.clientX,e.clientY),e.deltaY<0?this._dollyIn(this._getZoomScale(e.deltaY)):e.deltaY>0&&this._dollyOut(this._getZoomScale(e.deltaY)),this.update()}_handleKeyDown(e){let t=!1;switch(e.code){case this.keys.UP:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(yi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),t=!0;break;case this.keys.BOTTOM:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(-yi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),t=!0;break;case this.keys.LEFT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(yi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),t=!0;break;case this.keys.RIGHT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(-yi*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),this.update())}_handleTouchStartRotate(e){if(this._pointers.length===1)this._rotateStart.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._rotateStart.set(i,s)}}_handleTouchStartPan(e){if(this._pointers.length===1)this._panStart.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panStart.set(i,s)}}_handleTouchStartDolly(e){let t=this._getSecondPointerPosition(e),i=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(i*i+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enablePan&&this._handleTouchStartPan(e)}_handleTouchStartDollyRotate(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enableRotate&&this._handleTouchStartRotate(e)}_handleTouchMoveRotate(e){if(this._pointers.length==1)this._rotateEnd.set(e.pageX,e.pageY);else{let i=this._getSecondPointerPosition(e),s=.5*(e.pageX+i.x),r=.5*(e.pageY+i.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let t=this.domElement;this._rotateLeft(yi*this._rotateDelta.x/t.clientHeight),this._rotateUp(yi*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(e){if(this._pointers.length===1)this._panEnd.set(e.pageX,e.pageY);else{let t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panEnd.set(i,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(e){let t=this._getSecondPointerPosition(e),i=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(i*i+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);let a=(e.pageX+t.x)*.5,o=(e.pageY+t.y)*.5;this._updateZoomParameters(a,o)}_handleTouchMoveDollyPan(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enablePan&&this._handleTouchMovePan(e)}_handleTouchMoveDollyRotate(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enableRotate&&this._handleTouchMoveRotate(e)}_addPointer(e){this._pointers.push(e.pointerId)}_removePointer(e){delete this._pointerPositions[e.pointerId];for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId){this._pointers.splice(t,1);return}}_isTrackingPointer(e){for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId)return!0;return!1}_trackPointer(e){let t=this._pointerPositions[e.pointerId];t===void 0&&(t=new te,this._pointerPositions[e.pointerId]=t),t.set(e.pageX,e.pageY)}_getSecondPointerPosition(e){let t=e.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[t]}_customWheelEvent(e){let t=e.deltaMode,i={clientX:e.clientX,clientY:e.clientY,deltaY:e.deltaY};switch(t){case 1:i.deltaY*=16;break;case 2:i.deltaY*=100;break}return e.ctrlKey&&!this._controlActive&&(i.deltaY*=10),i}};function ky(n){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(n.pointerId),this.domElement.ownerDocument.addEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(n)&&(this._addPointer(n),n.pointerType==="touch"?this._onTouchStart(n):this._onMouseDown(n),this._cursorStyle==="grab"&&(this.domElement.style.cursor="grabbing")))}function Hy(n){this.enabled!==!1&&(n.pointerType==="touch"?this._onTouchMove(n):this._onMouseMove(n))}function Vy(n){switch(this._removePointer(n),this._pointers.length){case 0:this.domElement.releasePointerCapture(n.pointerId),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Sp),this.state=yt.NONE,this._cursorStyle==="grab"&&(this.domElement.style.cursor="grab");break;case 1:let e=this._pointers[0],t=this._pointerPositions[e];this._onTouchStart({pointerId:e,pageX:t.x,pageY:t.y});break}}function Gy(n){let e;switch(n.button){case 0:e=this.mouseButtons.LEFT;break;case 1:e=this.mouseButtons.MIDDLE;break;case 2:e=this.mouseButtons.RIGHT;break;default:e=-1}switch(e){case rs.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(n),this.state=yt.DOLLY;break;case rs.ROTATE:if(n.ctrlKey||n.metaKey||n.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(n),this.state=yt.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(n),this.state=yt.ROTATE}break;case rs.PAN:if(n.ctrlKey||n.metaKey||n.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(n),this.state=yt.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(n),this.state=yt.PAN}break;default:this.state=yt.NONE}this.state!==yt.NONE&&this.dispatchEvent(Bu)}function Wy(n){switch(this.state){case yt.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(n);break;case yt.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(n);break;case yt.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(n);break}}function Xy(n){this.enabled===!1||this.enableZoom===!1||this.state!==yt.NONE||(n.preventDefault(),this.dispatchEvent(Bu),this._handleMouseWheel(this._customWheelEvent(n)),this.dispatchEvent(Sp))}function qy(n){this.enabled!==!1&&this._handleKeyDown(n)}function Yy(n){switch(this._trackPointer(n),this._pointers.length){case 1:switch(this.touches.ONE){case as.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(n),this.state=yt.TOUCH_ROTATE;break;case as.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(n),this.state=yt.TOUCH_PAN;break;default:this.state=yt.NONE}break;case 2:switch(this.touches.TWO){case as.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(n),this.state=yt.TOUCH_DOLLY_PAN;break;case as.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(n),this.state=yt.TOUCH_DOLLY_ROTATE;break;default:this.state=yt.NONE}break;default:this.state=yt.NONE}this.state!==yt.NONE&&this.dispatchEvent(Bu)}function $y(n){switch(this._trackPointer(n),this.state){case yt.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(n),this.update();break;case yt.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(n),this.update();break;case yt.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(n),this.update();break;case yt.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(n),this.update();break;default:this.state=yt.NONE}}function Zy(n){this.enabled!==!1&&n.preventDefault()}function Jy(n){n.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function Ky(n){n.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}var Nc=class extends Cs{constructor(){super(),this.name="RoomEnvironment",this.position.y=-3.5;let e=new Bt;e.deleteAttribute("uv");let t=new Qe({side:Qt}),i=new Qe,s=new Da(16777215,900,28,2);s.position.set(.418,16.199,.3),this.add(s);let r=new tt(e,t);r.position.set(-.757,13.219,.717),r.scale.set(31.713,28.305,28.591),this.add(r);let a=new $t(e,i,6),o=new pt;o.position.set(-10.906,2.009,1.846),o.rotation.set(0,-.195,0),o.scale.set(2.328,7.905,4.651),o.updateMatrix(),a.setMatrixAt(0,o.matrix),o.position.set(-5.607,-.754,-.758),o.rotation.set(0,.994,0),o.scale.set(1.97,1.534,3.955),o.updateMatrix(),a.setMatrixAt(1,o.matrix),o.position.set(6.167,.857,7.803),o.rotation.set(0,.561,0),o.scale.set(3.927,6.285,3.687),o.updateMatrix(),a.setMatrixAt(2,o.matrix),o.position.set(-2.017,.018,6.124),o.rotation.set(0,.333,0),o.scale.set(2.002,4.566,2.064),o.updateMatrix(),a.setMatrixAt(3,o.matrix),o.position.set(2.291,-.756,-2.621),o.rotation.set(0,-.286,0),o.scale.set(1.546,1.552,1.496),o.updateMatrix(),a.setMatrixAt(4,o.matrix),o.position.set(-2.193,-.369,-5.547),o.rotation.set(0,.516,0),o.scale.set(3.875,3.487,2.986),o.updateMatrix(),a.setMatrixAt(5,o.matrix),this.add(a);let c=new tt(e,Ur(50));c.position.set(-16.116,14.37,8.208),c.scale.set(.1,2.428,2.739),this.add(c);let l=new tt(e,Ur(50));l.position.set(-16.109,18.021,-8.207),l.scale.set(.1,2.425,2.751),this.add(l);let h=new tt(e,Ur(17));h.position.set(14.904,12.198,-1.832),h.scale.set(.15,4.265,6.331),this.add(h);let d=new tt(e,Ur(43));d.position.set(-.462,8.89,14.52),d.scale.set(4.38,5.441,.088),this.add(d);let u=new tt(e,Ur(20));u.position.set(3.235,11.486,-12.541),u.scale.set(2.5,2,.1),this.add(u);let f=new tt(e,Ur(100));f.position.set(0,20,0),f.scale.set(1,.1,1),this.add(f)}dispose(){let e=new Set;this.traverse(t=>{t.isMesh&&(e.add(t.geometry),e.add(t.material))});for(let t of e)t.dispose()}};function Ur(n){return new Ra({color:0,emissive:16777215,emissiveIntensity:n})}var Ep=new di,Uc=new A,Bs=class extends La{constructor(){super(),this.isLineSegmentsGeometry=!0,this.type="LineSegmentsGeometry";let e=[-1,2,0,1,2,0,-1,1,0,1,1,0,-1,0,0,1,0,0,-1,-1,0,1,-1,0],t=[-1,2,1,2,-1,1,1,1,-1,-1,1,-1,-1,-2,1,-2],i=[0,2,1,2,3,1,2,4,3,4,5,3,4,6,5,6,7,5];this.setIndex(i),this.setAttribute("position",new nt(e,3)),this.setAttribute("uv",new nt(t,2))}applyMatrix4(e){let t=this.attributes.instanceStart,i=this.attributes.instanceEnd;return t!==void 0&&(t.applyMatrix4(e),i.applyMatrix4(e),t.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}setPositions(e){let t;e instanceof Float32Array?t=e:Array.isArray(e)&&(t=new Float32Array(e));let i=new ss(t,6,1);return this.setAttribute("instanceStart",new Pi(i,3,0)),this.setAttribute("instanceEnd",new Pi(i,3,3)),this.instanceCount=this.attributes.instanceStart.count,this.computeBoundingBox(),this.computeBoundingSphere(),this}setColors(e){let t;e instanceof Float32Array?t=e:Array.isArray(e)&&(t=new Float32Array(e));let i=new ss(t,6,1);return this.setAttribute("instanceColorStart",new Pi(i,3,0)),this.setAttribute("instanceColorEnd",new Pi(i,3,3)),this}fromWireframeGeometry(e){return this.setPositions(e.attributes.position.array),this}fromEdgesGeometry(e){return this.setPositions(e.attributes.position.array),this}fromMesh(e){return this.fromWireframeGeometry(new Ta(e.geometry)),this}fromLineSegments(e){let t=e.geometry;return this.setPositions(t.attributes.position.array),this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new di);let e=this.attributes.instanceStart,t=this.attributes.instanceEnd;e!==void 0&&t!==void 0&&(this.boundingBox.setFromBufferAttribute(e),Ep.setFromBufferAttribute(t),this.boundingBox.union(Ep))}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Ci),this.boundingBox===null&&this.computeBoundingBox();let e=this.attributes.instanceStart,t=this.attributes.instanceEnd;if(e!==void 0&&t!==void 0){let i=this.boundingSphere.center;this.boundingBox.getCenter(i);let s=0;for(let r=0,a=e.count;r<a;r++)Uc.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared(Uc)),Uc.fromBufferAttribute(t,r),s=Math.max(s,i.distanceToSquared(Uc));this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&console.error("THREE.LineSegmentsGeometry.computeBoundingSphere(): Computed radius is NaN. The instanced position data is likely to have NaN values.",this)}}toJSON(){}};ye.line={worldUnits:{value:1},linewidth:{value:1},resolution:{value:new te},dashOffset:{value:0},dashScale:{value:1},dashSize:{value:1},gapSize:{value:1}};pi.line={uniforms:fi.merge([ye.common,ye.fog,ye.line]),vertexShader:`
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
		`};var Fr=class extends Rt{constructor(e){super({type:"LineMaterial",uniforms:fi.clone(pi.line.uniforms),vertexShader:pi.line.vertexShader,fragmentShader:pi.line.fragmentShader,clipping:!0}),this.isLineMaterial=!0,this.setValues(e)}get color(){return this.uniforms.diffuse.value}set color(e){this.uniforms.diffuse.value=e}get worldUnits(){return"WORLD_UNITS"in this.defines}set worldUnits(e){e===!0!==this.worldUnits&&(this.needsUpdate=!0),e===!0?this.defines.WORLD_UNITS="":delete this.defines.WORLD_UNITS}get linewidth(){return this.uniforms.linewidth.value}set linewidth(e){this.uniforms.linewidth&&(this.uniforms.linewidth.value=e)}get dashed(){return"USE_DASH"in this.defines}set dashed(e){e===!0!==this.dashed&&(this.needsUpdate=!0),e===!0?this.defines.USE_DASH="":delete this.defines.USE_DASH}get dashScale(){return this.uniforms.dashScale.value}set dashScale(e){this.uniforms.dashScale.value=e}get dashSize(){return this.uniforms.dashSize.value}set dashSize(e){this.uniforms.dashSize.value=e}get dashOffset(){return this.uniforms.dashOffset.value}set dashOffset(e){this.uniforms.dashOffset.value=e}get gapSize(){return this.uniforms.gapSize.value}set gapSize(e){this.uniforms.gapSize.value=e}get opacity(){return this.uniforms.opacity.value}set opacity(e){this.uniforms&&(this.uniforms.opacity.value=e)}get resolution(){return this.uniforms.resolution.value}set resolution(e){this.uniforms.resolution.value.copy(e)}get alphaToCoverage(){return"USE_ALPHA_TO_COVERAGE"in this.defines}set alphaToCoverage(e){this.defines&&(e===!0!==this.alphaToCoverage&&(this.needsUpdate=!0),e===!0?this.defines.USE_ALPHA_TO_COVERAGE="":delete this.defines.USE_ALPHA_TO_COVERAGE)}};var zu=new gt,wp=new A,Tp=new A,ti=new gt,ii=new gt,yn=new gt,ku=new A,Hu=new rt,ni=new Fa,Ap=new A,Fc=new di,Oc=new Ci,Mn=new gt,bn,zs;function Rp(n,e,t){return Mn.set(0,0,-e,1).applyMatrix4(n.projectionMatrix),Mn.multiplyScalar(1/Mn.w),Mn.x=zs/t.width,Mn.y=zs/t.height,Mn.applyMatrix4(n.projectionMatrixInverse),Mn.multiplyScalar(1/Mn.w),Math.abs(Math.max(Mn.x,Mn.y))}function jy(n,e){let t=n.matrixWorld,i=n.geometry,s=i.attributes.instanceStart,r=i.attributes.instanceEnd,a=Math.min(i.instanceCount,s.count);for(let o=0,c=a;o<c;o++){ni.start.fromBufferAttribute(s,o),ni.end.fromBufferAttribute(r,o),ni.applyMatrix4(t);let l=new A,h=new A;bn.distanceSqToSegment(ni.start,ni.end,h,l),h.distanceTo(l)<zs*.5&&e.push({point:h,pointOnLine:l,distance:bn.origin.distanceTo(h),object:n,face:null,faceIndex:o,uv:null,uv1:null})}}function Qy(n,e,t){let i=e.projectionMatrix,r=n.material.resolution,a=n.matrixWorld,o=n.geometry,c=o.attributes.instanceStart,l=o.attributes.instanceEnd,h=Math.min(o.instanceCount,c.count),d=-e.near;bn.at(1,yn),yn.w=1,yn.applyMatrix4(e.matrixWorldInverse),yn.applyMatrix4(i),yn.multiplyScalar(1/yn.w),yn.x*=r.x/2,yn.y*=r.y/2,yn.z=0,ku.copy(yn),Hu.multiplyMatrices(e.matrixWorldInverse,a);for(let u=0,f=h;u<f;u++){if(ti.fromBufferAttribute(c,u),ii.fromBufferAttribute(l,u),ti.w=1,ii.w=1,ti.applyMatrix4(Hu),ii.applyMatrix4(Hu),ti.z>d&&ii.z>d)continue;if(ti.z>d){let b=ti.z-ii.z,v=(ti.z-d)/b;ti.lerp(ii,v)}else if(ii.z>d){let b=ii.z-ti.z,v=(ii.z-d)/b;ii.lerp(ti,v)}ti.applyMatrix4(i),ii.applyMatrix4(i),ti.multiplyScalar(1/ti.w),ii.multiplyScalar(1/ii.w),ti.x*=r.x/2,ti.y*=r.y/2,ii.x*=r.x/2,ii.y*=r.y/2,ni.start.copy(ti),ni.start.z=0,ni.end.copy(ii),ni.end.z=0;let x=ni.closestPointToPointParameter(ku,!0);ni.at(x,Ap);let p=Vt.lerp(ti.z,ii.z,x),m=p>=-1&&p<=1,M=ku.distanceTo(Ap)<zs*.5;if(m&&M){ni.start.fromBufferAttribute(c,u),ni.end.fromBufferAttribute(l,u),ni.start.applyMatrix4(a),ni.end.applyMatrix4(a);let b=new A,v=new A;bn.distanceSqToSegment(ni.start,ni.end,v,b),t.push({point:v,pointOnLine:b,distance:bn.origin.distanceTo(v),object:n,face:null,faceIndex:u,uv:null,uv1:null})}}}var Bc=class extends tt{constructor(e=new Bs,t=new Fr({color:Math.random()*16777215})){super(e,t),this.isLineSegments2=!0,this.type="LineSegments2"}computeLineDistances(){let e=this.geometry,t=e.attributes.instanceStart,i=e.attributes.instanceEnd,s=new Float32Array(2*t.count);for(let a=0,o=0,c=t.count;a<c;a++,o+=2)wp.fromBufferAttribute(t,a),Tp.fromBufferAttribute(i,a),s[o]=o===0?0:s[o-1],s[o+1]=s[o]+wp.distanceTo(Tp);let r=new ss(s,2,1);return e.setAttribute("instanceDistanceStart",new Pi(r,1,0)),e.setAttribute("instanceDistanceEnd",new Pi(r,1,1)),this}raycast(e,t){let i=this.material.worldUnits,s=e.camera;if(s===null&&!i&&console.error('LineSegments2: "Raycaster.camera" needs to be set in order to raycast against LineSegments2 while worldUnits is set to false.'),i===!1&&(this.material.resolution.x===0||this.material.resolution.y===0))return;let r=e.params.Line2!==void 0&&e.params.Line2.threshold||0;bn=e.ray;let a=this.matrixWorld,o=this.geometry,c=this.material;zs=c.linewidth+r,o.boundingSphere===null&&o.computeBoundingSphere(),Oc.copy(o.boundingSphere).applyMatrix4(a);let l;if(i)l=zs*.5;else{let d=Math.max(s.near,Oc.distanceToPoint(bn.origin));l=Rp(s,d,c.resolution)}if(Oc.radius+=l,bn.intersectsSphere(Oc)===!1)return;o.boundingBox===null&&o.computeBoundingBox(),Fc.copy(o.boundingBox).applyMatrix4(a);let h;if(i)h=zs*.5;else{let d=Math.max(s.near,Fc.distanceToPoint(bn.origin));h=Rp(s,d,c.resolution)}Fc.expandByScalar(h),bn.intersectsBox(Fc)!==!1&&(i?jy(this,t):Qy(this,s,t))}onBeforeRender(e){let t=this.material.uniforms;t&&t.resolution&&(e.getViewport(zu),this.material.uniforms.resolution.value.set(zu.z,zu.w))}};var io=new A;function Gi(n,e,t,i,s,r){let a=2*Math.PI*s/4,o=Math.max(r-2*s,0),c=Math.PI/4;io.copy(e),io[i]=0,io.normalize();let l=.5*a/(a+o),h=1-io.angleTo(n)/c;return Math.sign(io[t])===1?h*l:o/(a+o)+l+l*(1-h)}var zc=class n extends Bt{constructor(e=1,t=1,i=1,s=2,r=.1){let a=s*2+1;if(r=Math.min(e/2,t/2,i/2,r),super(1,1,1,a,a,a),this.type="RoundedBoxGeometry",this.parameters={width:e,height:t,depth:i,segments:s,radius:r},a===1)return;let o=this.toNonIndexed();this.index=null,this.attributes.position=o.attributes.position,this.attributes.normal=o.attributes.normal,this.attributes.uv=o.attributes.uv;let c=new A,l=new A,h=new A(e,t,i).divideScalar(2).subScalar(r),d=this.attributes.position.array,u=this.attributes.normal.array,f=this.attributes.uv.array,g=d.length/6,x=new A,p=.5/a;for(let m=0,M=0;m<d.length;m+=3,M+=2)switch(c.fromArray(d,m),l.copy(c),l.x-=Math.sign(l.x)*p,l.y-=Math.sign(l.y)*p,l.z-=Math.sign(l.z)*p,l.normalize(),d[m+0]=h.x*Math.sign(c.x)+l.x*r,d[m+1]=h.y*Math.sign(c.y)+l.y*r,d[m+2]=h.z*Math.sign(c.z)+l.z*r,u[m+0]=l.x,u[m+1]=l.y,u[m+2]=l.z,Math.floor(m/g)){case 0:x.set(1,0,0),f[M+0]=Gi(x,l,"z","y",r,i),f[M+1]=1-Gi(x,l,"y","z",r,t);break;case 1:x.set(-1,0,0),f[M+0]=1-Gi(x,l,"z","y",r,i),f[M+1]=1-Gi(x,l,"y","z",r,t);break;case 2:x.set(0,1,0),f[M+0]=1-Gi(x,l,"x","z",r,e),f[M+1]=Gi(x,l,"z","x",r,i);break;case 3:x.set(0,-1,0),f[M+0]=1-Gi(x,l,"x","z",r,e),f[M+1]=1-Gi(x,l,"z","x",r,i);break;case 4:x.set(0,0,1),f[M+0]=1-Gi(x,l,"x","y",r,e),f[M+1]=1-Gi(x,l,"y","x",r,t);break;case 5:x.set(0,0,-1),f[M+0]=Gi(x,l,"x","y",r,e),f[M+1]=1-Gi(x,l,"y","x",r,t);break}}static fromJSON(e){return new n(e.width,e.height,e.depth,e.segments,e.radius)}};var Dp=[16756767,3262128,16740193,10194175],Vu=new Map;function It(n,e={}){let t=`${n}:${JSON.stringify(e)}`;return Vu.has(t)||Vu.set(t,new Qe({color:n,roughness:.52,metalness:0,...e})),Vu.get(t)}var Ne={dark:It(3883079,{roughness:.7}),darker:It(2830133,{roughness:.8}),rubber:It(2763824,{roughness:.95}),metal:It(10989748,{roughness:.35,metalness:.55}),chrome:It(14936298,{roughness:.18,metalness:.85}),glass:It(8242390,{roughness:.08,metalness:.1,emissive:1454650,emissiveIntensity:.35}),seat:It(3093304,{roughness:.9}),soil:It(11039551,{roughness:1,flatShading:!0}),lamp:It(16773570,{emissive:16770720,emissiveIntensity:.9}),tail:It(16730682,{emissive:12590608,emissiveIntensity:.6}),white:It(16052712,{roughness:.6})},eM=[{body:16036379,accent:16765788,trim:3883079},{body:14964026,accent:16165179,trim:3883079,bed:15903035},{body:16022304,accent:16752717,trim:3093304}],$u=new Bt(1,1,1),ks=new tn(1,1,1,24),Zu=new tn(1,1,1,10);function si(n,e,t,i=0,s=0,r=0){let a=new tt(e,t);return a.position.set(i,s,r),a.castShadow=!0,a.receiveShadow=!0,n.add(a),a}function ct(n,e,t,i,s,r,a,o){let c=si(n,$u,e,t,i,s);return c.scale.set(r,a,o),c}function Ni(n,e,t,i,s,r,a,o,c){return si(n,new zc(r,a,o,3,Math.min(c,r/2,a/2,o/2)),e,t,i,s)}function gi(n,e,t,i,s,r,a,o=0,c=ks){let l=si(n,c,e,t,i,s);return l.scale.set(r,a,r),l.rotation.x=o,l}function no(n,e,t,i,s){let r=new A(...t),a=new A(...i),o=a.clone().sub(r),c=si(n,ks,e);return c.position.copy(r.add(a).multiplyScalar(.5)),c.scale.set(s,o.length(),s),c.quaternion.setFromUnitVectors(new A(0,1,0),o.normalize()),c}function Gu(n,e,t,i,s,r=!1){let a=new Di;a.moveTo(0,-t*.4),r?(a.quadraticCurveTo(-t*.14,t*.15,e*.18,t*.5),a.quadraticCurveTo(e*.24,t*.6,e*.33,t*.48),a.lineTo(e*.89,t*.2),a.quadraticCurveTo(e+t*.2,t*.2,e+t*.15,-t*.08),a.quadraticCurveTo(e+t*.1,-t*.4,e*.9,-t*.32),a.lineTo(e*.26,-t*.25)):(a.lineTo(e*.22,t*.55),a.lineTo(e*.7,t*.3),a.lineTo(e,t*.12),a.lineTo(e,-t*.32),a.lineTo(e*.24,-t*.25)),a.closePath();let o=new Hi(a,{depth:i,bevelEnabled:!0,bevelSegments:r?3:1,curveSegments:10,steps:1,bevelSize:t*.085,bevelThickness:t*.085});return si(n,o,s,0,0,-i/2)}function Wi(n,e,t,i,s){let r=new pt;return r.name=e,r.position.set(t,i,s),n.add(r),r}function mi(n,e,t,i,s,r,a){return gi(n,e,t,i,s,r,a,Math.PI/2)}function Hc(n,e,t,i){let s=t.clone().sub(e);n.position.copy(e).add(t).multiplyScalar(.5),n.scale.set(i,s.length(),i),n.quaternion.setFromUnitVectors(new A(0,1,0),s.normalize())}function Wu(n,e,t,i,s){let r=si(n,ks,Ne.dark),a=si(n,ks,Ne.chrome);return r.name=`${s}-barrel`,a.name=`${s}-piston`,{start:e,end:t,barrel:r,piston:a,update(){let o=n.worldToLocal(e.getWorldPosition(new A)),c=n.worldToLocal(t.getWorldPosition(new A));Hc(r,o,o.clone().lerp(c,.58),i),Hc(a,o.clone().lerp(c,.43),c,i*.56)}}}function tM(n,e,t,i){let s=e.x-n.x,r=e.y-n.y,a=Math.hypot(s,r);if(a<=Math.abs(t-i)||a>=t+i)throw new Error("Bucket linkage pose is outside its mechanical range.");let o=(t**2-i**2+a**2)/(2*a),c=Math.sqrt(Math.max(0,t**2-o**2));return new A(n.x+(o*s-c*r)/a,n.y+(o*r+c*s)/a,0)}function Ju(n,e,t){if(typeof document>"u")return null;let i=document.createElement("canvas");i.width=n,i.height=e,t(i.getContext("2d"));let s=new Nn(i);return s.colorSpace=Ft,s}function iM(){return Ju(128,32,n=>{n.fillStyle="#f4b21b",n.fillRect(0,0,128,32),n.fillStyle="#2b2f35";for(let e=-32;e<160;e+=24)n.beginPath(),n.moveTo(e,32),n.lineTo(e+12,32),n.lineTo(e+44,0),n.lineTo(e+32,0),n.fill()})}var Xu;function nM(n,e,t,i,s,r){let a=new et;a.position.set(e,i,t),n.add(a);let o=new et;a.add(o),gi(o,Ne.rubber,0,0,0,i*.9,s,Math.PI/2);let c=Math.sign(t)||1;gi(o,r,0,0,c*s*.47,i*.56,s*.12,Math.PI/2),gi(o,Ne.metal,0,0,c*s*.54,i*.2,s*.1,Math.PI/2,Zu);for(let l=0;l<6;l++){let h=l*Math.PI/3;ct(o,Ne.darker,Math.cos(h)*i*.36,Math.sin(h)*i*.36,c*s*.53,i*.09,i*.09,s*.05)}for(let l=0;l<14;l++){let h=l*Math.PI*2/14,d=ct(o,Ne.rubber,Math.sin(h)*i*.93,Math.cos(h)*i*.93,(l%2?.18:-.18)*s,i*.26,i*.16,s*.6);d.rotation.z=-h}return{steer:a,spin:o,radius:i}}function sM(n,e,t,i,s,r){let a=new et;a.position.z=s*t*.35,n.add(a);let o=i*.13,c=e*.34,l=i*.02,h=o+l,d=t*.2,u=4*c+2*Math.PI*o,f=new Di;f.moveTo(-c,l+o*.35),f.lineTo(c,l+o*.35),f.absarc(c,h,o*.65,-Math.PI/2,Math.PI/2,!1),f.lineTo(-c,h+o*.65),f.absarc(-c,h,o*.65,Math.PI/2,Math.PI*1.5,!1);let g=new Hi(f,{depth:d*.7,bevelEnabled:!0,bevelSegments:2,bevelSize:i*.012,bevelThickness:i*.012,steps:1});si(a,g,r,0,0,-d*.35);let x=[];for(let _=0;_<5;_++)x.push(gi(a,Ne.metal,e*(-.26+_*.13),l+o*.42,s*d*.38,i*.045,d*.12,Math.PI/2));for(let _ of[-c,c]){let E=new et;E.position.set(_,h,0),a.add(E),x.push(E),gi(E,Ne.dark,0,0,0,o*.82,d*.86,Math.PI/2),gi(E,Ne.metal,0,0,s*d*.44,o*.38,d*.06,Math.PI/2,Zu);for(let P=0;P<8;P++){let I=P*Math.PI/4;ct(E,Ne.darker,Math.cos(I)*o*.6,Math.sin(I)*o*.6,s*d*.44,o*.14,o*.14,d*.04)}}let p=Math.max(24,Math.round(u/(i*.055))),m=u/p,M=new $t($u,Ne.rubber,p);M.castShadow=!0,M.receiveShadow=!0,a.add(M);let b=new pt,v=i*.034,T=(_,E)=>{if(_=(_%u+u)%u,_<2*c){E.set(-c+_,l,-Math.PI/2);return}if(_-=2*c,_<Math.PI*o){let I=-Math.PI/2+_/o;E.set(c+Math.cos(I)*o,h+Math.sin(I)*o,I);return}if(_-=Math.PI*o,_<2*c){E.set(c-_,h+o,Math.PI/2);return}_-=2*c;let P=Math.PI/2+_/o;E.set(-c+Math.cos(P)*o,h+Math.sin(P)*o,P)},w=new A,C=_=>{for(let E=0;E<p;E++){T(E*m+_,w);let P=w.z;b.position.set(w.x+Math.cos(P)*v*.4,w.y+Math.sin(P)*v*.4,0),b.rotation.set(0,0,P-Math.PI/2),b.scale.set(m*.82,v,d),b.updateMatrix(),M.setMatrixAt(E,b.matrix)}M.instanceMatrix.needsUpdate=!0;for(let E of x)E.isGroup&&(E.rotation.z=-_/(o*.82))};return C(0),{update:C,shoes:M}}function rM(n,e,t,i,s,r,a){let o=[],c=[],l=[];if(ct(n,Ne.dark,0,i*.22,0,e*.74,i*.17,t*.6),s){let h=i*(a?.23:.19),d=t*(a?.22:.18);for(let u of[-.3,.3])for(let f of[-.4,.4]){let g=nM(n,u*e,f*t,h,d,It(r.accent));c.push(g),u>0&&o.push(g.steer)}for(let u of[-.3,.3])no(n,Ne.darker,[u*e,h,-.4*t],[u*e,h,.4*t],i*.05)}else for(let h of[-1,1])l.push({side:h,...sM(n,e,t,i,h,It(r.trim,{roughness:.7}))});return{steering:o,spinning:c,tracks:l}}function qu(n,e,t,i,s,r,a,o="x"){let c=It(16777215,{transparent:!0,opacity:.45,emissive:16777215,emissiveIntensity:.4,depthWrite:!1});for(let[l,h]of[[-.18,.16],[.1,.07]]){let d=ct(n,c,e,t,i,s,r,a);d.castShadow=!1,o==="x"?(d.scale.set(s,r*1.2,a*h),d.position.z+=a*l*2.2,d.rotation.x=.5):(d.scale.set(s*h,r*1.2,a),d.position.x+=s*l*2.2,d.rotation.z=-.5),d.userData.skipAO=!0,d.name="glass-highlight"}}function Cp(n,e,t,i,s,r,a){let o=It(a.body),c=new et;c.position.set(s,0,r),n.add(c);let l=.3*e,h=.39*t,d=.5*i;Ni(c,o,0,.12*i,0,l,.2*i,h,i*.04),Ni(c,o,0,.36*i,0,l*.96,d*.72,h*.96,i*.05).name="cab-shell";let u=.39*i,f=d*.56;ct(c,Ne.glass,l*.485,u,0,i*.012,f,h*.84),qu(c,l*.492,u,0,i*.01,f*.8,h*.84,"x");for(let g of[-1,1])ct(c,Ne.glass,-l*.04,u,g*h*.485,l*.76,f,i*.012),qu(c,-l*.04,u,g*h*.492,l*.76,f*.8,i*.01,"z");ct(c,Ne.glass,-l*.485,u+f*.1,0,i*.012,f*.6,h*.7),ct(c,Ne.seat,-l*.12,.3*i,0,l*.3,.16*i,h*.5),Ni(c,It(a.trim),0,.62*i,0,l*1.06,i*.05,h*1.06,i*.02);for(let g of[-1,1])ct(c,Ne.lamp,l*.5,.6*i,g*h*.3,i*.02,i*.035,h*.12);return c}function aM(n,e,t,i){let s=new et;n.add(s),s.name="loader-bucket",ct(s,Ne.dark,e*.07,-e*.1,0,e*.35,e*.06,t);let r=ct(s,Ne.dark,-e*.12,.01*e,0,e*.06,e*.26,t);r.rotation.z=.25,ct(s,Ne.dark,-e*.04,.14*e,0,e*.14,e*.04,t*.98).rotation.z=-.5;for(let o of[-1,1])ct(s,It(i.body),0,.025*e,o*t*.48,e*.31,e*.22,t*.055);ct(s,Ne.metal,e*.25,-e*.12,0,e*.06,e*.03,t*1.01);for(let o=0;o<6;o++)ct(s,Ne.metal,e*.29,-e*.115,(o/5-.5)*t*.86,e*.1,e*.035,t*.07);let a=si(s,Ku(3),Ne.soil,.04*e,.03*e,0);return a.scale.set(e*.2,e*.12,t*.42),a.visible=!1,{root:s,soil:a}}var kc=new Map;function Ku(n){if(kc.has(n))return kc.get(n);let e=new nn(1,1),t=e.attributes.position,i=n*9301+49297,s=()=>(i=i*16807%2147483647)/2147483647,r=new Map;for(let a=0;a<t.count;a++){let o=`${t.getX(a).toFixed(3)},${t.getY(a).toFixed(3)},${t.getZ(a).toFixed(3)}`;r.has(o)||r.set(o,.82+s()*.3);let c=r.get(o),l=t.getY(a);t.setXYZ(a,t.getX(a)*c,(l<0?l*.25:l)*c,t.getZ(a)*c)}return e.computeVertexNormals(),kc.set(n,e),e}function oM(n,e,t,i,s){let r=new et;r.name="bucket-curl",n.add(r);let a=new et;a.name="bucket-orientation",a.rotation.y=Math.PI,r.add(a);let o=new Di;o.moveTo(-.14*e,-.05*e),o.bezierCurveTo(-.34*e,-.12*e,-.34*e,-.34*e,-.18*e,-.43*e),o.quadraticCurveTo(.04*e,-.5*e,.3*e,-.4*e),o.lineTo(.32*e,-.34*e),o.quadraticCurveTo(.06*e,-.43*e,-.15*e,-.37*e),o.bezierCurveTo(-.27*e,-.3*e,-.26*e,-.16*e,-.09*e,-.105*e),o.closePath();let c=new Hi(o,{depth:t,bevelEnabled:!0,bevelSegments:3,bevelSize:e*.008,bevelThickness:e*.008,curveSegments:12,steps:1});si(a,c,Ne.dark,0,0,-t/2).name="bucket-shell";let l=new Di;l.moveTo(-.14*e,-.06*e),l.bezierCurveTo(-.34*e,-.14*e,-.34*e,-.35*e,-.17*e,-.43*e),l.quadraticCurveTo(.05*e,-.48*e,.31*e,-.39*e),l.lineTo(.19*e,-.18*e),l.quadraticCurveTo(.04*e,-.1*e,-.14*e,-.06*e);let h=t*.05,d=new Hi(l,{depth:h,bevelEnabled:!0,bevelSegments:3,bevelSize:e*.009,bevelThickness:e*.007,curveSegments:12,steps:1});for(let C of[-1,1])si(a,d,It(s.body),0,0,C*t*.475-h/2);Ni(a,Ne.metal,e*.29,-e*.38,0,e*.07,e*.06,t*1.02,e*.012).rotation.z=.1;let u=new Di;u.moveTo(0,-.035*e),u.lineTo(.17*e,-.016*e),u.lineTo(.17*e,.009*e),u.lineTo(0,.035*e),u.closePath();let f=new Hi(u,{depth:t*.085,bevelEnabled:!0,bevelSegments:2,bevelSize:e*.006,bevelThickness:e*.006,steps:1});for(let C=0;C<5;C++)si(a,f,Ne.chrome,e*.31,-e*.385,(C/4-.5)*t*.82-t*.0425);let g=new Di;g.moveTo(-.14*e,-.11*e),g.lineTo(-.14*e,.115*e),g.quadraticCurveTo(-.1*e,.18*e,-.045*e,.15*e),g.lineTo(.075*e,.025*e),g.quadraticCurveTo(.1*e,-.025*e,.055*e,-.1*e),g.closePath();let x=t*.075,p=e*.008,m=i+e*.02,M=(m+x)/2+p,b=Math.max(t*.72,m+2*x+4*p+e*.014),v=new Hi(g,{depth:x,bevelEnabled:!0,bevelSegments:3,bevelSize:e*.008,bevelThickness:p,curveSegments:10,steps:1});for(let C of[-1,1])si(a,v,It(s.accent),0,0,C*M-x/2).name=`bucket-ear-${C}`;mi(a,Ne.metal,0,0,0,e*.045,b).name="bucket-main-pin";let T=Wi(a,"bucket-link-pin",-.08*e,.12*e,0);mi(T,Ne.metal,0,0,0,e*.032,b);let w=si(a,Ku(1),Ne.soil,.005*e,-.245*e,0);return w.name="bucket-soil",w.scale.set(e*.225,e*.125,t*.405),w.visible=!1,{curl:r,orientation:a,soil:w,linkPin:T}}function lM(n,e){let t=`#${Dp[n.id%4].toString(16).padStart(6,"0")}`,i=String(n.id+1).padStart(2,"0"),s=Ju(160,96,a=>{if(a.textAlign="center",a.textBaseline="middle",e==="paper"){a.font='600 44px Inter, "Helvetica Neue", Arial, sans-serif',a.lineJoin="round",a.lineWidth=10,a.strokeStyle="#ffffffee",a.strokeText(i,80,38),a.fillStyle="#1f2426",a.fillText(i,80,38),a.fillStyle="#ffffffee",a.beginPath(),a.roundRect(52,66,56,14,7),a.fill(),a.fillStyle=t,a.beginPath(),a.roundRect(56,69,48,8,4),a.fill();return}a.fillStyle="#00000033",a.beginPath(),a.roundRect(22,12,116,58,29),a.fill(),a.fillStyle=t,a.beginPath(),a.roundRect(20,8,120,58,29),a.fill(),a.beginPath(),a.moveTo(68,62),a.lineTo(92,62),a.lineTo(80,80),a.closePath(),a.fill(),a.lineWidth=5,a.strokeStyle="#ffffffcc",a.beginPath(),a.roundRect(22.5,10.5,115,53,26.5),a.stroke(),a.font='800 38px ui-rounded, "SF Pro Rounded", system-ui, sans-serif',a.fillStyle="#1f2a2c",a.fillText(i,80,39)}),r=new ua(new xr({map:s,depthTest:!1,transparent:!0,sizeAttenuation:!1}));return r.center.set(.5,0),r.renderOrder=30,r.scale.set(.05,.03,1),r}function cM(n,e=!0){let t=`#${n.toString(16).padStart(6,"0")}`,i=Ju(256,256,r=>{if(e){let a=r.createRadialGradient(128,128,60,128,128,126);a.addColorStop(0,`${t}00`),a.addColorStop(.82,`${t}38`),a.addColorStop(1,`${t}00`),r.fillStyle=a,r.fillRect(0,0,256,256)}r.strokeStyle=t,r.lineWidth=9,r.lineCap="round";for(let a=0;a<16;a++)r.beginPath(),r.arc(128,128,112,a*Math.PI/8+.06,(a+.62)*Math.PI/8),r.stroke()}),s=new tt(new sn(1,1),new In({map:i,transparent:!0,depthWrite:!1,polygonOffset:!0,polygonOffsetFactor:-4}));return s.rotation.x=-Math.PI/2,s.renderOrder=16,s.userData.skipAO=!0,s}var Xi=n=>n*n*(3-2*n),hM=n=>Math.min(1,Math.max(0,n)),Yu=n=>1+(1.9+1)*(n-1)**3+1.9*(n-1)**2,Mi=(n,e,t)=>hM((n-e)/(t-e));function uM(n,e,t){for(let i=1;i<e.length;i++){let[s,r,a=Xi]=e[i],[o,c]=e[i-1];if(t<=s||i===e.length-1){let l=a(Mi(t,o,s)),h=n[c],d=n[r];return Object.fromEntries(Object.keys(h).map(u=>[u,h[u]+(d[u]-h[u])*l]))}}return n[e[0][1]]}var Pp={carry:{boom:.63,stick:-1.35,pitch:.04},reach:{boom:.3,stick:-1.05,pitch:-.42},scoop:{boom:.2,stick:-1.3,pitch:.62},raise:{boom:.8,stick:-1.02,pitch:.1},pour:{boom:.74,stick:-.98,pitch:.94}},Ip={dig:[[0,"carry"],[.3,"reach"],[.56,"scoop"],[1,"carry",Yu]],dump:[[0,"carry"],[.34,"raise"],[.62,"pour"],[1,"carry",Yu]]};function Lp(n,e,{labels:t=!0,style:i="diorama"}={}){let s=new et,r=n.height*e,a=n.width*e,o=Math.min(r,a),c=eM[n.type],l=n.action_type===1||n.type===1;s.name=`machine-${n.id}`;let h=rM(s,r,a,o,l,c,n.type===1),d=new et;d.name="suspension",s.add(d);let u=new et;u.position.y=.32*o,d.add(u);let f=It(c.body),g=It(c.accent),x=It(c.trim),p=new Qe({color:16753183,roughness:.3,emissive:16742912,emissiveIntensity:.2,transparent:!0,opacity:.92}),m,M,b,v,T,w,C,_,E,P=[];if(Xu||(Xu=new Qe({map:iM(),color:typeof document>"u"?16036379:16777215,roughness:.6})),n.type===0){gi(u,Ne.dark,0,.045*o,0,o*.31,o*.1),Ni(u,f,-.1*r,.16*o,0,r*.65,o*.23,a*.66,o*.065).name="excavator-upper-body",Ni(u,Ne.dark,-.33*r,.255*o,0,r*.2,o*.2,a*.65,o*.068).name="excavator-counterweight",ct(u,Xu,-.434*r,.255*o,0,r*.012,o*.09,a*.56);for(let ee of[-1,1])ct(u,Ne.tail,-.434*r,.3*o,ee*a*.29,r*.012,o*.03,a*.05);Ni(u,g,-.22*r,.29*o,.16*a,r*.26,o*.05,a*.3,o*.02);for(let ee=0;ee<5;ee++)ct(u,Ne.darker,(-.3+ee*.04)*r,.318*o,.16*a,r*.018,o*.012,a*.22);Cp(u,r,a,o,-.02*r,-.18*a,c),E=gi(u,p,-.1*r,.7*o,-.18*a,o*.032,o*.06),gi(u,Ne.dark,-.1*r,.665*o,-.18*a,o*.04,o*.02),gi(u,Ne.dark,-.28*r,.45*o,.26*a,o*.026,o*.32),gi(u,Ne.darker,-.28*r,.62*o,.26*a,o*.034,o*.03),_=Wi(u,"exhaust",-.28*r,.66*o,.26*a),no(u,Ne.metal,[-.33*r,.4*o,.32*a],[-.12*r,.4*o,.32*a],o*.012);for(let ee of[-.33,-.12])no(u,Ne.metal,[ee*r,.27*o,.32*a],[ee*r,.4*o,.32*a],o*.012);let q=Math.max(r*.63,n.reach[1]*e*.4),Z=Math.max(r*.48,n.reach[1]*e*.32);m=new et,m.name="boom-pivot",m.position.set(.16*r,.23*o,.09*a),u.add(m),Gu(m,q,o*.21,a*.12,g,!0),mi(m,Ne.dark,0,0,0,o*.095,a*.19),mi(m,Ne.metal,0,0,0,o*.05,a*.205);for(let ee of[-1,1])ct(m,Ne.lamp,q*.3,o*.1,ee*a*.065,o*.04,o*.03,o*.012);for(let ee of[-1,1]){let le=Wi(u,`boom-cylinder-${ee}-start`,.2*r,.12*o,(.09+ee*.12)*a),J=Wi(m,`boom-cylinder-${ee}-end`,q*.48,-.055*o,ee*a*.12);mi(le,f,0,0,0,o*.047,a*.055),mi(J,f,0,0,0,o*.047,a*.055),P.push(Wu(u,le,J,o*.036,`boom-cylinder-${ee}`))}M=new et,M.name="stick-pivot",M.position.x=q,m.add(M),Gu(M,Z,o*.17,a*.09,f,!0),Ni(M,f,-.07*Z,.045*o,0,.22*Z,o*.105,a*.09,o*.035),mi(M,Ne.dark,0,0,0,o*.078,a*.16),mi(M,Ne.metal,0,0,0,o*.04,a*.175);let j=Wi(m,"stick-cylinder-start",q*.4,o*.145,0),de=Wi(M,"stick-cylinder-end",-.09*Z,o*.08,0);mi(j,g,0,0,0,o*.048,a*.11),mi(de,f,0,0,0,o*.043,a*.115),P.push(Wu(u,j,de,o*.04,"stick-cylinder"));let Ge=.65,me=o*Ge,k=a*.39*Ge,ce=a*.145,ae=oM(M,me,k,ce,c);b=ae.curl,b.position.x=Z,v=ae.soil,Wi(M,"bucket-hinge",Z,0,0),mi(M,f,Z,0,0,me*.068,ce).name="bucket-hinge-housing";let Te=Wi(M,"bucket-rocker-pivot",Z-me*.24,me*.1,0);Ni(M,f,Te.position.x,.04*me,0,me*.115,me*.17,a*.105,me*.025),mi(Te,Ne.metal,0,0,0,me*.036,k*.72);let Ue=Wi(M,"bucket-rocker-joint",0,0,0);mi(Ue,Ne.metal,0,0,0,me*.036,k*.72);let Oe=me*.22,st=me*.25,He=[];for(let ee of[-1,1]){let le=si(M,ks,f),J=si(M,ks,Ne.dark);le.name=`bucket-rocker-${ee}`,J.name=`bucket-link-${ee}`,He.push({first:le,second:J,z:ee*k*.3})}let oe=Wi(M,"bucket-cylinder-start",Z*.24,o*.12,0);mi(oe,f,0,0,0,me*.038,a*.115),P.push(Wu(M,oe,Ue,me*.03,"bucket-cylinder")),C={origin:Te,joint:Ue,destination:ae.linkPin,firstLength:Oe,secondLength:st,links:He,update(){let ee=M.worldToLocal(ae.linkPin.getWorldPosition(new A)),le=tM(Te.position,ee,Oe,st);Ue.position.copy(le);for(let J of He){let se=Te.position.clone(),fe=ee.clone(),ge=le.clone();se.z=J.z,ge.z=J.z,fe.z=J.z,Hc(J.first,se,ge,me*.029),Hc(J.second,ge,fe,me*.025)}}}}else if(n.type===1){ct(u,Ne.dark,0,.02*o,0,r*.92,.09*o,a*.62);for(let de of[-1,1])ct(u,x,.05*r,.08*o,de*a*.44,r*.7,.05*o,a*.1);Ni(u,f,.36*r,.16*o,0,.22*r,.26*o,a*.82,o*.05),ct(u,Ne.darker,.475*r,.15*o,0,r*.02,.15*o,a*.5);for(let de=0;de<4;de++)ct(u,Ne.metal,.486*r,(.1+de*.035)*o,0,r*.01,o*.012,a*.44);for(let de of[-1,1])ct(u,Ne.lamp,.478*r,.24*o,de*a*.32,r*.02,.05*o,a*.1),ct(u,Ne.chrome,.44*r,.06*o,de*a*.37,r*.08,.04*o,a*.12);Cp(u,r*.92,a*1.62,o*.95,.3*r,-.14*a,c),_=Wi(u,"exhaust",.2*r,.78*o,.3*a),gi(u,Ne.chrome,.2*r,.5*o,.3*a,o*.03,o*.52);let q=new et;q.name="truck-bed",q.position.set(-.43*r,.12*o,0),u.add(q),T=q;let Z=It(c.bed);ct(q,Z,.3*r,0,0,r*.64,o*.08,a*.86);for(let de of[-1,1]){let Ge=ct(q,Z,.3*r,.2*o,de*a*.41,.66*r,o*.38,a*.05);Ge.rotation.x=de*.08;for(let me=0;me<4;me++)ct(q,g,(.06+me*.16)*r,.22*o,de*a*.44,r*.025,o*.34,a*.02);ct(q,g,.3*r,.4*o,de*a*.43,.66*r,o*.04,a*.07)}ct(q,Z,.62*r,.28*o,0,.04*r,o*.52,a*.86);let j=ct(q,Z,.72*r,.52*o,0,.22*r,o*.04,a*.86);j.rotation.z=-.06,ct(q,Ne.dark,-.02*r,.22*o,0,.03*r,o*.3,a*.78),v=si(q,Ku(2),Ne.soil,r*.3,o*.2,0),v.scale.set(r*.27,o*.2,a*.33);for(let de of[-1,1])ct(u,Ne.tail,-.47*r,.05*o,de*.32*a,.02*r,.05*o,.1*a)}else{Ni(u,f,-.06*r,.12*o,0,.72*r,.26*o,.64*a,o*.05),Ni(u,Ne.dark,-.34*r,.2*o,0,.14*r,.22*o,.6*a,o*.04);for(let me=0;me<4;me++)ct(u,Ne.darker,-.412*r,(.12+me*.045)*o,0,r*.01,o*.018,a*.46);let q=new et;q.position.set(-.06*r,.25*o,0),u.add(q);let Z=.34*r,j=.4*a,de=.46*o;for(let me of[-1,1])for(let k of[-1,1])ct(q,Ne.darker,me*Z*.47,de/2,k*j*.47,o*.035,de,o*.035);Ni(q,f,0,de,0,Z*1.06,o*.05,j*1.08,o*.02),ct(q,Ne.glass,Z*.47,de*.52,0,o*.01,de*.78,j*.86),qu(q,Z*.478,de*.52,0,o*.01,de*.6,j*.86,"x");for(let me of[-1,1])for(let k=0;k<4;k++)ct(q,Ne.darker,(-.3+k*.2)*Z,de*.55,me*j*.47,o*.012,de*.8,o*.012);ct(q,Ne.seat,-Z*.1,de*.25,0,Z*.35,de*.3,j*.5),E=gi(q,p,-Z*.3,de+o*.05,0,o*.03,o*.05),_=Wi(u,"exhaust",-.36*r,.42*o,.2*a),gi(u,Ne.dark,-.36*r,.36*o,.2*a,o*.025,o*.14),w=new et,w.name="loader-arm",w.position.set(-.18*r,.22*o,0),u.add(w);for(let me of[-1,1]){let k=new et;k.position.z=me*a*.36,w.add(k),Gu(k,r*.81,o*.13,a*.075,g),no(k,Ne.chrome,[r*.1,-.08*o,0],[r*.5,-.06*o,0],o*.022),mi(k,Ne.metal,0,0,0,o*.05,a*.09)}no(w,x,[r*.66,-.04*o,-a*.36],[r*.66,-.04*o,a*.36],o*.04);let Ge=aM(w,o*1.22,a*.92,c);b=Ge.root,b.position.set(.8*r,-.1*o,0),v=Ge.soil}E&&(E.name="beacon");let I=Dp[n.id%4],L=cM(I,i!=="paper"),X=i==="paper";X&&s.traverse(q=>{q.name==="glass-highlight"&&(q.visible=!1)}),L.scale.set(r*1.34,a*1.34+(r-a)*.35,1),L.position.y=e*.03,s.add(L);let W=null;t&&(W=lM(n,i),W.position.set(-.05*r,o*1.12,0),s.add(W));let U=new A,z={last:null,treads:[0,0],spin:0,active:!1,kind:"",phase:1,lift:0,tags:!0},H=h.spinning[0]?.radius??o*.2,Q=v?v.scale.clone():null;function ie(q,Z,j=0,de=""){u.rotation.y=q.cabin_yaw;for(let me of h.steering)me.rotation.y=Math.max(-.6,Math.min(.6,q.wheel_angle*Math.PI/9));z.active=Z,z.kind=de,z.phase=j,L.visible=Z&&z.tags;let Ge=q.loaded>0;if(v.visible=Ge||de==="dump"&&j<.52||de==="transfer"&&j<.52||de==="dig"&&j>.5,de==="receive"&&(v.visible=j>.62),Q&&n.type!==0){let me=de==="dump"?Math.max(1,q.previous_loaded??q.loaded):q.loaded,k=.55+.45*(1-Math.exp(-Math.max(me,1)/18));de==="dump"&&(k*=1-Xi(Mi(j,.22,.5)),v.visible=j<.5),v.scale.set(Q.x,Q.y*Math.max(k,.02),Q.z*(de==="dump"?.7+.3*k:1))}if(m){let me=de==="dig"?Ip.dig:de==="dump"||de==="transfer"?Ip.dump:null,k=me?uM(Pp,me,j):Pp.carry;m.rotation.z=k.boom,M.rotation.z=k.stick,b.rotation.z=k.pitch-m.rotation.z-M.rotation.z,s.updateWorldMatrix(!0,!0),C.update();for(let ce of P)ce.update()}if(T){let me=de==="dump"?j<.45?Xi(Mi(j,0,.45)):j<.7?1:1-Xi(Mi(j,.7,1)):0;T.rotation.z=.62*me}if(w){let me=q.shovel_lifted?.35:-.2,k=Ge?.22:0,ce=me,ae=k;de==="dig"?(ce=j<.35?Vt.lerp(-.2,-.3,Xi(Mi(j,0,.35))):j<.6?-.3:Vt.lerp(-.3,me,Yu(Mi(j,.6,1))),ae=j<.35?-.15*Xi(Mi(j,0,.35)):j<.6?Vt.lerp(-.15,.3,Xi(Mi(j,.35,.6))):Vt.lerp(.3,k,Xi(Mi(j,.6,1)))):de==="dump"?(ce=j<.6?Vt.lerp(.35,.45,Xi(Mi(j,0,.35))):Vt.lerp(.45,me,Xi(Mi(j,.6,1))),ae=j<.3?.22*(1-Mi(j,0,.3)):j<.65?-.8*Xi(Mi(j,.3,.5)):Vt.lerp(-.8,k,Xi(Mi(j,.65,1)))):de==="turn"&&(ce=me),w.rotation.z=ce,b.rotation.z=ae}}return{root:s,agent:n,bucketRig:C,hydraulics:P,suspension:d,ringColor:I,setPose:ie,setTags(q){z.tags=q,W&&(W.visible=q),L.visible=q&&z.active},drive(q,Z){let j=z.last;if(z.last={x:q.x,z:q.z,yaw:Z},!j)return;let de=q.x-j.x,Ge=q.z-j.z,me=Math.atan2(Math.sin(Z-j.yaw),Math.cos(Z-j.yaw));if(Math.hypot(de,Ge)>r*1.5||Math.abs(me)>1.2)return;let k=de*Math.cos(Z)-Ge*Math.sin(Z);z.speed=k;for(let[ce,ae]of h.tracks.entries())z.treads[ce]-=k+ae.side*a*.35*me,ae.update(z.treads[ce]);z.spin-=k/H;for(let ce of h.spinning)ce.spin.rotation.z=z.spin},tick(q,{move:Z=null,direction:j=1,reducedMotion:de=!1}={}){let Ge=z.active;if(X){p.emissiveIntensity=.15,L.material.opacity=Ge?.9:0,L.rotation.z=0,W&&(W.material.opacity=1,W.position.y=o*1.12),d.position.y=0,d.rotation.z=0;return}if(p.emissiveIntensity=Ge?.5+.9*Math.max(0,Math.sin(q*7))**3:.15,L.material.opacity=Ge?.75+.25*Math.sin(q*3.2):0,L.rotation.z=q*.25,W&&(W.material.opacity=Ge?1:.72,W.position.y=o*1.12+(Ge&&!de?Math.sin(q*3)*o*.03:0)),de){d.position.y=0,d.rotation.z=0;return}let me=Z===null?0:-Math.sin(Z*Math.PI*2)*.035*j;d.rotation.z=me,d.position.y=Ge?Math.sin(q*41)*o*.0015:0,Z!==null&&(d.position.y+=Math.abs(Math.sin(Z*Math.PI*3))*o*.008)},tip(){return v.getWorldPosition(U),U.clone()},exhaust(){return _?_.getWorldPosition(new A):s.position.clone()},bedLip(){return T?T.localToWorld(new A(-.02*r,.05*o,0)):this.tip()},dispose(){let q=new Set;s.traverse(Z=>{Z.geometry&&Z.geometry!==$u&&Z.geometry!==ks&&Z.geometry!==Zu&&![...kc.values()].includes(Z.geometry)&&q.add(Z.geometry),Z.isSprite&&(Z.material.map.dispose(),Z.material.dispose())});for(let Z of q)Z.dispose();L.material.map?.dispose(),L.material.dispose(),p.dispose();for(let Z of h.tracks)Z.shoes.dispose()}}}var Np="terra.viewer3d.v1",ju=["Excavator","Truck","Skid steer"],dM=["Forward","Backward","Turn clockwise","Turn anticlockwise","Cabin clockwise","Cabin anticlockwise","Work","Wait"],fM=["action","target","padding","dumpability"],Up=["dumpability_static","interaction","traversability","precision_required_band","fresh_dig_current","fresh_dig_swing","remaining_target","footprint","work_cone","pull_permission"],ps=n=>typeof n=="number"&&Number.isFinite(n),Fn=n=>Number.isSafeInteger(n);function Ct(n,e){if(!n)throw new Error(e)}function Qu(n){Ct(n&&typeof n=="object"&&n.schema===Np,`Expected a ${Np} replay.`),Ct(n.metadata&&typeof n.metadata.title=="string"&&typeof n.metadata.source=="string","Replay metadata must include title and source strings."),Ct(Array.isArray(n.frames)&&n.frames.length>0,"The replay contains no frames."),Ct(n.frames.length<=1e5,"This viewer supports at most 100,000 frames.");for(let[e,t]of n.frames.entries())ed(t,`Frame ${e}`);return n}function ed(n,e="Frame"){Ct(n&&typeof n=="object",`${e}: expected an object.`);let{grid:t,maps:i,agents:s}=n;Ct(t&&Fn(t.rows)&&Fn(t.cols)&&t.rows>0&&t.cols>0&&t.rows<=128&&t.cols<=128,`${e}: grid must be between 1 and 128 cells on each side.`),Ct(ps(t.tile_size_m)&&t.tile_size_m>0,`${e}: invalid tile size.`),Ct(i&&typeof i=="object",`${e}: missing maps.`);for(let a of[...fM,...Up]){let o=i[a];if(o==null&&Up.includes(a))continue;Ct(Array.isArray(o)&&o.length===t.rows,`${e}: ${a} has the wrong row count.`);let c=a==="action"||a==="target",l=a==="traversability"?[-1,0,1,!1,!0]:[0,1,!1,!0];for(let h of o)Ct(Array.isArray(h)&&h.length===t.cols&&h.every(d=>c?Fn(d):l.includes(d)),`${e}: ${a} has invalid cells or columns.`)}Ct(Fn(n.step)&&n.step>=0&&ps(n.reward),`${e}: invalid step or reward.`),Ct(n.action===null||Fn(n.action)&&n.action>=0&&n.action<=7,`${e}: invalid action.`),Ct(typeof n.done=="boolean"&&typeof n.task_done=="boolean",`${e}: invalid episode outcome.`),Ct(!n.task_done||n.done,`${e}: task_done requires done.`),Ct(Array.isArray(s)&&s.length>0&&s.length<=4,`${e}: expected 1\u20134 active agents.`);let r=new Set;for(let a of s)Ct(a&&Fn(a.id)&&a.id>=0&&a.id<=3&&!r.has(a.id),`${e}: agent IDs must be unique original slots from 0 to 3.`),r.add(a.id),Ct(Fn(a.type)&&a.type>=0&&a.type<=2&&(a.action_type===0||a.action_type===1),`${e}: unknown machine type.`),Ct(Array.isArray(a.position)&&a.position.length===2&&a.position.every(ps)&&a.position[0]>=0&&a.position[0]<t.rows&&a.position[1]>=0&&a.position[1]<t.cols,`${e}: agent position is outside the map.`),Ct(ps(a.base_yaw)&&ps(a.cabin_yaw)&&Fn(a.wheel_angle),`${e}: invalid machine angle.`),Ct(ps(a.width)&&ps(a.height)&&a.width>0&&a.height>0,`${e}: invalid machine footprint.`),Ct(Fn(a.loaded)&&a.loaded>=0&&(a.shovel_lifted===0||a.shovel_lifted===1),`${e}: invalid machine load or shovel state.`),Ct(Array.isArray(a.reach)&&a.reach.length===2&&a.reach.every(ps)&&a.reach[0]>=0&&a.reach[1]>=a.reach[0],`${e}: invalid machine reach.`);return Ct(r.has(n.current_agent),`${e}: the active agent does not exist.`),Ct(n.actor_id===null||r.has(n.actor_id),`${e}: the preceding actor does not exist.`),n}function td(n,e){if(n.action===null)return"Initial state";let t=(e||n).agents.find(i=>i.id===n.actor_id)||n.agents[0];return t.action_type===1&&(n.action===2||n.action===3)?n.action===2?"Steer left":"Steer right":n.action===6?t.type===2?"Shovel action":t.loaded>0?"Dump / transfer":t.type===1?"Dump":"Dig":dM[n.action]}function so(n){let{maps:e,grid:t}=n,i=(n.agents.find(o=>o.id===n.current_agent)?.loaded??0)>0,s=e.fresh_dig_current!=null&&e.fresh_dig_swing!=null,r={current:0,swing:0,blocked:0,remaining:0,precision:0},a=Array.from({length:t.rows},()=>Array(t.cols).fill(0));for(let o=0;o<t.rows;o++)for(let c=0;c<t.cols;c++)if(!(e.padding[o][c]||(e.precision_required_band?.[o][c]&&r.precision++,!(e.remaining_target!=null?!!e.remaining_target[o][c]:e.target[o][c]<0&&e.action[o][c]>e.target[o][c])))&&(r.remaining++,!!s)){if(i){a[o][c]=4;continue}e.fresh_dig_current[o][c]?(a[o][c]=1,r.current++):e.fresh_dig_swing[o][c]?(a[o][c]=2,r.swing++):(a[o][c]=3,r.blocked++)}return{available:s,loaded:i,cells:a,counts:r}}var Fp=["No remaining dig target","Dig now","Swing cabin only","Not diggable from this base","Unload first"];function Vc(n,e){if(!n||e.grid.rows!==n.grid.rows||e.grid.cols!==n.grid.cols)return{kind:"snapshot",changed:[],removed:0,placed:0,message:"Initial state"};let t=[],i=0,s=0;for(let d=0;d<e.grid.rows;d++)for(let u=0;u<e.grid.cols;u++){let f=e.maps.action[d][u]-n.maps.action[d][u];f&&(t.push({row:d,col:u,delta:f}),f<0?i-=f:s+=f)}let r=e.agents.find(d=>d.id===e.actor_id),a=n.agents.find(d=>d.id===e.actor_id),o=r&&a?r.loaded-a.loaded:0,c=e.agents.find(d=>d.id!==e.actor_id&&d.loaded>(n.agents.find(u=>u.id===d.id)?.loaded??d.loaded)),l="unchanged",h="No visible state change";return i>0&&o>0?(l="dig",h=`Picked up ${o} soil units \xB7 ${t.length} cells changed`):s>0&&o<0?(l="dump",h=`Placed ${-o} soil units \xB7 ${t.length} cells changed`):o<0&&c?(l="transfer",h=`Transferred soil to machine ${c.id+1}`):t.length?(l="terrain",h=`${t.length} terrain cells changed`):r&&a&&r.position.some((d,u)=>d!==a.position[u])?(l="move",h="Machine moved"):r&&a&&(r.base_yaw!==a.base_yaw||r.cabin_yaw!==a.cabin_yaw||r.wheel_angle!==a.wheel_angle||r.shovel_lifted!==a.shovel_lifted)&&(l="turn",h="Machine configuration changed"),{kind:l,changed:t,removed:i,placed:s,loadDelta:o,recipient:c,message:h}}function Op(n){let e=0,t=0,i=0;for(let s=0;s<n.grid.rows;s++)for(let r=0;r<n.grid.cols;r++){let a=n.maps.action[s][r];e+=Math.max(0,-a),t+=Math.max(0,a),i+=Math.max(0,-n.maps.target[s][r])}return{cut:e,fill:t,target:i,carried:n.agents.reduce((s,r)=>s+r.loaded,0)}}function id(n,e,t){let i=Math.atan2(Math.sin(e-n),Math.cos(e-n));return n+i*t}function Bp(n){if(n===0)return"0.00";let e=Math.abs(n)<.01?n.toPrecision(3):n.toFixed(2);return`${n>0?"+":""}${e}`}var ms={name:"diorama",sand:14069366,dug:[12749400,11366984,9855037,8343860],loose:11037754,strata:[13013084,11433289,13673068,10250821],rock:9340541,grass:[9224530,7646533],grassEdge:6263612,sky:[9225962,13624815,16246732]},Xc={diorama:ms,paper:{...ms,name:"paper",sand:14208441,dug:[12889485,11441525,9928288,8415310],loose:11242084,strata:[13153428,11705468,13878182,10718574],rock:10196622}},on={uTime:{value:0},uUnit:{value:.27},uTile:{value:.57},uFloor:{value:-1},uMotion:{value:1}},zp=`
varying vec3 vTWorld;
varying vec3 vTNormal;
float tHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float tNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
  return mix(mix(tHash(i), tHash(i + vec2(1, 0)), f.x), mix(tHash(i + vec2(0, 1)), tHash(i + vec2(1, 1)), f.x), f.y);
}
float tFbm(vec2 p) { return tNoise(p) * .55 + tNoise(p * 2.13 + 7.1) * .3 + tNoise(p * 4.7 + 3.3) * .15; }
`,kp=`
vec4 tWorld = vec4(transformed, 1.);
vec3 tNormal = objectNormal;
#ifdef USE_INSTANCING
tWorld = instanceMatrix * tWorld; tNormal = mat3(instanceMatrix) * tNormal;
#endif
tWorld = modelMatrix * tWorld; vTWorld = tWorld.xyz; vTNormal = normalize(mat3(modelMatrix) * tNormal);
`,Gc=n=>new Le(n),Wc=n=>`vec3(${n.r.toFixed(4)}, ${n.g.toFixed(4)}, ${n.b.toFixed(4)})`;function pM(n,e){let[t,i,s,r]=e.strata.map(a=>Wc(Gc(a)));return`
  {
    float depth = -vTWorld.y / uUnit;
    float along = vTWorld.x * .83 + vTWorld.z * 1.17;
    float wobble = (tNoise(vec2(along * 1.4, depth * .35)) - .5) * .32;
    float band = depth + wobble, index = mod(floor(band), 4.), phase = fract(band);
    vec3 stratum = index < 1. ? ${t} : index < 2. ? ${i} : index < 3. ? ${s} : ${r};
    stratum *= .93 + tNoise(vec2(along * 5.1, vTWorld.y * 9.)) * .12;
    stratum *= mix(.8, 1., smoothstep(0., .1, phase));
    stratum *= 1. - clamp(depth * .025, 0., .28);
    if (vTWorld.y < uFloor) stratum = ${Wc(Gc(e.rock))} * (.86 + tNoise(vec2(along * 2.3, vTWorld.y * 2.7)) * .2);
    ${n?`if (vTWorld.y > -uTile * .2) stratum = ${Wc(Gc(e.grassEdge))} * (.92 + tNoise(vec2(along * 3., 1.)) * .14);`:""}
    diffuseColor.rgb = stratum;
  }`}function On(n,e={},t=ms){let i=new Qe({roughness:1,metalness:0,...e}),[s,r]=t.grass.map(o=>Wc(Gc(o))),a=n==="island"?`float g = smoothstep(.3, .72, tFbm(vTWorld.xz * .28)); diffuseColor.rgb = mix(${s}, ${r}, g) * (.94 + tNoise(vTWorld.xz * 3.1) * .1);`:n==="pile"?"diffuseColor.rgb *= (.84 + .2 * smoothstep(0., 5., vTWorld.y / uUnit)) * (.92 + tFbm(vTWorld.xz * 2.6) * .16);":"diffuseColor.rgb *= .9 + tFbm(vTWorld.xz * 1.15) * .2;";return i.onBeforeCompile=o=>{Object.assign(o.uniforms,on),o.vertexShader=`varying vec3 vTWorld;
varying vec3 vTNormal;
${o.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
${kp}`),o.fragmentShader=`uniform float uUnit;
uniform float uTile;
uniform float uFloor;
${zp}
${o.fragmentShader}`.replace("#include <color_fragment>",`#include <color_fragment>
      {
        vec3 tn = normalize(vTNormal);
        if (tn.y > .5) { ${a} }
        else if (tn.y > -.5) { ${n==="pile"?a:pM(n==="island",t)} }
      }`)},i.customProgramCacheKey=()=>`terra-earth-${n}-${t.name}`,i}var mM={hatch:0,dots:1,solid:2,cross:3,stripes:4};function qc({color:n,opacity:e,pattern:t="solid",...i}){let s=new In({color:n,transparent:!0,opacity:e,depthWrite:!1,...i}),r=mM[t];return s.onBeforeCompile=a=>{Object.assign(a.uniforms,on),a.vertexShader=`varying vec3 vTWorld;
varying vec3 vTNormal;
${a.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
vec3 objectNormal = vec3(0., 1., 0.);
${kp}`),a.fragmentShader=`uniform float uTile;
uniform float uTime;
uniform float uMotion;
${zp}
${a.fragmentShader}`.replace("#include <color_fragment>",`#include <color_fragment>
      {
        vec2 p = vTWorld.xz / uTile;
        float a = 1.;
        ${r===0?"a = mix(.42, 1., step(.5, fract((p.x + p.y) * .7 - uTime * .12 * uMotion)));":""}
        ${r===1?"vec2 q = fract(p * 1.5) - .5; a = mix(.5, 1., 1. - smoothstep(.2, .26, length(q)));":""}
        ${r===3?"a = mix(.35, 1., max(step(.72, fract((p.x + p.y) * .7)), step(.72, fract((p.x - p.y) * .7))));":""}
        ${r===4?"a = mix(.3, 1., step(.62, fract((p.x - p.y) * .55)));":""}
        diffuseColor.a *= a;
      }`)},s.customProgramCacheKey=()=>`terra-zone-${r}`,s}function Hp(){let n=document.createElement("canvas");n.width=4,n.height=256;let e=n.getContext("2d"),t=e.createLinearGradient(0,0,0,256),[i,s,r]=ms.sky.map(o=>`#${o.toString(16).padStart(6,"0")}`);t.addColorStop(0,i),t.addColorStop(.58,s),t.addColorStop(1,r),e.fillStyle=t,e.fillRect(0,0,4,256);let a=new Nn(n);return a.colorSpace=Ft,a}var gM=[[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[0,2],[0,1]],_M=1.6;function Yc(n,e,t,i,s){if(e<0||t<0||e>=n.grid.rows||t>=n.grid.cols||n.maps.padding[e][t])return 0;let r=n.maps.action[e][t],a=i?i.maps.action[e][t]:r;return Math.max(0,a+(r-a)*s)}function xM(n,e,t,i,s){let r=e%2?[(e-1)/2]:[e/2-1,e/2],a=t%2?[(t-1)/2]:[t/2-1,t/2],o=1/0;for(let c of r)for(let l of a)o=Math.min(o,Yc(n,c,l,i,s));return o}function ro(n,e=null,t=1,i=null){let s=n.grid.rows*2+1,r=n.grid.cols*2+1,a=new Float64Array(s*r),o=_M/2;for(let c=0;c<s;c++)for(let l=0;l<r;l++)a[c*r+l]=xM(n,c,l,e,t);for(let c=0;c<s;c++){let l=c*r;for(let h=1;h<r;h++)a[l+h]=Math.min(a[l+h],a[l+h-1]+o);for(let h=r-2;h>=0;h--)a[l+h]=Math.min(a[l+h],a[l+h+1]+o)}for(let c=1;c<s;c++)for(let l=0;l<r;l++){let h=c*r+l;a[h]=Math.min(a[h],a[h-r]+o)}for(let c=s-2;c>=0;c--)for(let l=0;l<r;l++){let h=c*r+l;a[h]=Math.min(a[h],a[h+r]+o)}if(e&&t>0&&t<1){let c=i?.start??ro(n,e,0),l=i?.end??ro(n);for(let h=0;h<a.length;h++)a[h]=Math.min(a[h],c.heights[h]+(l.heights[h]-c.heights[h])*t)}return{rows:s,cols:r,heights:a}}function vM(n,e=null){let t=[],i=[],s=new Map,r=n.grid.cols*2+1,a=(o,c)=>{let l=o*r+c;return s.has(l)||(s.set(l,t.length),t.push([o,c])),s.get(l)};for(let o=0;o<n.grid.rows;o++)for(let c=0;c<n.grid.cols;c++){if(Yc(n,o,c,e,0)<=0&&Yc(n,o,c,e,1)<=0)continue;let l=a(o*2+1,c*2+1),h=gM.map(([u,f])=>a(o*2+u,c*2+f)),d=h.flatMap((u,f)=>[l,u,h[(f+1)%h.length]]);i.push({row:o,col:c,center:l,ring:h,triangles:d})}return{nodes:t,cells:i}}var $c=class extends et{constructor(e,{previous:t=null,unitHeight:i=e.grid.tile_size_m*.48,layerSettings:s={},visibility:r={},palette:a=ms}={}){super(),this.frame=e,this.previous=t,this.unitHeight=i,this.endHeights=ro(e),this.startHeights=t?ro(e,t,0):this.endHeights,this.endpoints={start:this.startHeights,end:this.endHeights},this.progress=1,this.topology=vM(e,t),this.activeCells=[];let{rows:o,cols:c,tile_size_m:l}=e.grid,h=new Float32Array(this.topology.nodes.length*3),d=new Float32Array(h.length),u=new Le;for(let x=0;x<this.topology.nodes.length;x++){let[p,m]=this.topology.nodes[x];h[x*3]=(m/2-c/2)*l,h[x*3+2]=(p/2-o/2)*l;let M=(p*37+m*61+p*m*7)%29/29;u.set(a.loose).multiplyScalar(.95+M*.1),u.toArray(d,x*3)}this.positions=new Yt(h,3).setUsage(Pr);let f=new mt;f.setAttribute("position",this.positions),f.setAttribute("color",new Yt(d,3)),this.surface=new tt(f,On("pile",{vertexColors:!0,flatShading:!0,polygonOffset:!0,polygonOffsetFactor:-1,polygonOffsetUnits:-2},a)),this.surface.name="connected-soil-piles",this.surface.castShadow=!0,this.surface.receiveShadow=!0,this.surface.userData.soilPiles=this,this.add(this.surface),this.layerSettings=s,this.overlays={};for(let[x,p]of Object.entries(s)){let m=new mt;m.setAttribute("position",this.positions);let M=qc({color:p.color,opacity:p.opacity*.7,pattern:p.pattern,polygonOffset:!0,polygonOffsetFactor:-2}),b=new tt(m,M);b.position.y=l*(.008+Object.keys(this.overlays).length*.003),b.renderOrder=3+Object.keys(this.overlays).length,b.visible=!!r[x],b.frustumCulled=!1,b.userData.skipAO=!0,this.overlays[x]=b,this.add(b)}let g=new mt;g.setAttribute("position",this.positions),this.gridLines=new Qn(g,new Ln({color:7426351,transparent:!0,opacity:.25,depthWrite:!1})),this.gridLines.position.y=l*.022,this.gridLines.renderOrder=12,this.gridLines.visible=!!r.grid,this.gridLines.frustumCulled=!1,this.add(this.gridLines),this.update(1)}nodeHeight(e,t){return(this.heights.heights[e*this.heights.cols+t]??0)*this.unitHeight}endpointHeight(e,t,i=!1){let s=i?this.startHeights:this.endHeights;return(s.heights[(e*2+1)*s.cols+t*2+1]??0)*this.unitHeight}update(e=1){this.progress=e,this.heights=e<=0?this.startHeights:e>=1?this.endHeights:ro(this.frame,this.previous,e,this.endpoints);let t=this.positions.array;for(let r=0;r<this.topology.nodes.length;r++){let[a,o]=this.topology.nodes[r];t[r*3+1]=this.nodeHeight(a,o)}this.positions.needsUpdate=!0;let i=this.topology.cells.filter(r=>Yc(this.frame,r.row,r.col,this.previous,e)>0),s=i.length!==this.activeCells.length||i.some((r,a)=>r!==this.activeCells[a]);if(this.activeCells=i,this.surface.visible=i.length>0,s||!this.surface.geometry.index){this.surface.geometry.setIndex(i.flatMap(r=>r.triangles));for(let[r,a]of Object.entries(this.overlays)){let o=this.layerSettings[r],c=this.frame.maps[o.map];a.geometry.setIndex(i.filter(l=>c!=null&&o.test(c[l.row][l.col])).flatMap(l=>l.triangles))}this.gridLines.geometry.setIndex(i.flatMap(r=>r.ring.flatMap((a,o)=>[a,r.ring[(o+1)%r.ring.length]])))}this.surface.geometry.computeVertexNormals(),this.surface.geometry.computeBoundingSphere()}setLayer(e,t){e==="grid"?this.gridLines.visible=t:this.overlays[e]&&(this.overlays[e].visible=t&&this.frame.maps[this.layerSettings[e].map]!=null)}cellForHit(e){let t=this.activeCells[Math.floor(e.faceIndex/8)];return t?{row:t.row,col:t.col}:null}dispose(){this.traverse(e=>{e.geometry?.dispose(),e.material&&e.material.dispose()}),this.clear()}};function nd(n,e=0){let t=(n^Math.imul(e+1,2654435761))>>>0;return t=Math.imul(t^t>>>16,2246822507),t=Math.imul(t^t>>>13,3266489909),(t^t>>>16)>>>0}var Hs=(n,e)=>nd(n,e)/4294967295,Vp=(n,e,t)=>Math.max(e,Math.min(t,n));function Gp(n){if(!Array.isArray(n)||!n.length||!Array.isArray(n[0])||!n[0].length)throw new Error("Obstacle padding must be a nonempty rectangular array.");let e=n.length,t=n[0].length;if(e>128||t>128||n.some(i=>!Array.isArray(i)||i.length!==t||i.some(s=>![0,1,!1,!0].includes(s))))throw new Error("Obstacle padding must contain aligned 0/1 cells, at most 128 \xD7 128.");return{rows:e,cols:t}}function yM(n,e,t){let i=new Uint8Array(e*t),s=[];for(let r=0;r<e;r++)for(let a=0;a<t;a++){let o=r*t+a;if(!n[r][a]||i[o])continue;let c=[[r,a]];i[o]=1;let l=r,h=r,d=a,u=a;for(let g=0;g<c.length;g++){let[x,p]=c[g];l=Math.min(l,x),h=Math.max(h,x),d=Math.min(d,p),u=Math.max(u,p);for(let[m,M]of[[x-1,p],[x,p-1],[x,p+1],[x+1,p]]){if(m<0||M<0||m>=e||M>=t)continue;let b=m*t+M;n[m][M]&&!i[b]&&(i[b]=1,c.push([m,M]))}}let f=2166136261;for(let[g,x]of[...c].sort((p,m)=>p[0]-m[0]||p[1]-m[1]))f=Math.imul(f^g*131+x,16777619)>>>0;s.push({cells:c,minRow:l,maxRow:h,minCol:d,maxCol:u,seed:f})}return s}function MM(n,e,t){let i=new Uint16Array(t),s=null,r=0;for(let a=0;a<e;a++){let o=[];for(let c=0;c<t;c++)i[c]=n[a*t+c]?i[c]+1:0;for(let c=0;c<=t;c++){let l=c<t?i[c]:0,h=c;for(;o.length&&o[o.length-1].height>l;){let d=o.pop(),u=d.height*(c-d.start),f={row:a-d.height+1,col:d.start,rows:d.height,cols:c-d.start};(u>r||u===r&&(f.row<s.row||f.row===s.row&&f.col<s.col))&&(s=f,r=u),h=d.start}l&&(!o.length||o[o.length-1].height<l)&&o.push({start:h,height:l})}}return s}function bM(n,e,t,i){let s={...n};function r(a){if(a.row<0||a.col<0||a.row+a.rows>t||a.col+a.cols>i)return!1;for(let o=a.row;o<a.row+a.rows;o++)for(let c=a.col;c<a.col+a.cols;c++)if(!e[o*i+c])return!1;return!0}for(;;){let a=s,o=[{...a,row:a.row-1,rows:a.rows+1},{...a,col:a.col-1,cols:a.cols+1},{...a,rows:a.rows+1},{...a,cols:a.cols+1}].filter(r).sort((c,l)=>l.rows*l.cols-c.rows*c.cols||c.row-l.row||c.col-l.col);if(!o.length)return s;s=o[0]}}function SM(n){let{rows:e,cols:t}=Gp(n),i=[],s=yM(n,e,t);for(let[r,a]of s.entries()){let o=a.maxRow-a.minRow+1,c=a.maxCol-a.minCol+1,l=new Uint8Array(o*c);for(let[f,g]of a.cells)l[(f-a.minRow)*c+g-a.minCol]=1;let h=l.slice(),d=a.cells.length,u=a.cells.length/(o*c);for(;d;){let f=bM(MM(h,o,c),l,o,c);for(let v=f.row;v<f.row+f.rows;v++)for(let T=f.col;T<f.col+f.cols;T++){let w=v*c+T;h[w]&&(h[w]=0,d--)}let g={row:f.row+a.minRow,col:f.col+a.minCol,rows:f.rows,cols:f.cols},x=nd(a.seed,g.row*131+g.col),p=Math.min(f.rows,f.cols),m=Math.max(f.rows,f.cols),M=(a.minRow+a.minCol+o+c)%3;if(u>=.9&&p>=3&&m>=6&&f.rows*f.cols>=24&&M!==0){let v=f.cols>=f.rows,T=Math.min(3,Math.floor(p/3),Math.max(1,Math.round(p/(m*.37))));for(let w=0;w<T;w++){let C=Math.floor(p*w/T),_=Math.floor(p*(w+1)/T),E={...g};v?(E.row+=C,E.rows=_-C):(E.col+=C,E.cols=_-C),i.push({...E,kind:"container",component:r,seed:nd(x,w)})}}else i.push({...g,kind:"boulder",component:r,seed:x})}}return i}function EM(n,e,t,i,s=!0){let a=[],o=[],c=[],l=new Le().setHex([10131340,10721928,9278606][i%3]),h=new Le(8824919),d=new A,u=new A,f=new A;for(let p=0;p<3;p++){let m=[];for(let M=0;M<9;M++){let b=(M+Hs(i,M)*.13)*Math.PI*2/9,v=p===2?.44+Hs(i,M+20)*.22:.85+Hs(i,M+p*9+40)*.14;m.push(new A(Math.cos(b)*n/2*v,p===0?0:t*(p===1?.38+Hs(i,M+70)*.1:.76+Hs(i,M+90)*.18),Math.sin(b)*e/2*v))}a.push(m)}function g(p,m,M){f.crossVectors(d.subVectors(m,p),u.subVectors(M,p)).normalize();let b=Math.abs(f.y),v=l.clone().multiplyScalar(.84+Hs(i,o.length)*.2+b*.12);s&&b>.72&&Hs(i,o.length+7)<.45&&v.lerp(h,.55);for(let T of[p,m,M])o.push(T.x,T.y,T.z),c.push(v.r,v.g,v.b)}for(let p=0;p<9;p++){let m=(p+1)%9;for(let M=0;M<2;M++)g(a[M][p],a[M+1][p],a[M+1][m]),g(a[M][p],a[M+1][m],a[M][m]);g(new A(0,0,0),a[0][p],a[0][m]),g(a[2][p],new A(0,t,0),a[2][m])}let x=new mt;return x.setAttribute("position",new nt(o,3)),x.setAttribute("color",new nt(c,3)),x.computeVertexNormals(),x.computeBoundingBox(),x}function Or(n,e,t,i,s,r,a,o,c){let l=new tt(e,t);return l.position.set(i,s,r),l.scale.set(a,o,c),l.castShadow=!0,l.receiveShadow=!0,n.add(l),l}function wM(n,e,t,i,s){let r=Math.max(e.rows,e.cols)*t,a=Math.min(e.rows,e.cols)*t,o=r*.94,c=Math.min(a*.88,o*.42),l=Vp(c*.94,t*.7,t*3.6),h=new et;h.rotation.y=e.rows>e.cols?Math.PI/2:0,n.add(h);let d=i+t*.12;s.box||(s.box=new Bt(1,1,1)),s.metal||(s.metal=new Qe({color:7831675,roughness:.64,metalness:.25})),s.foundation||(s.foundation=new Qe({color:9343364,roughness:1}));let u=new Qe({color:(s.paper?[9277839,8357252,10001045]:[10772291,5340795,6455185])[e.seed%3],roughness:.77,metalness:.15}),f=u.clone();f.color.multiplyScalar(1.12),Or(h,s.box,s.foundation,0,d/2,0,o+t*.06,d,c+t*.09),Or(h,s.box,u,0,d+l/2,0,o,l,c),Or(h,s.box,f,0,d+l+t*.025,0,o,t*.05,c);let g=Math.max(4,Math.round(o/(t*.45))),x=new $t(s.box,f,g*2),p=new pt;x.castShadow=!0,x.receiveShadow=!0;for(let m=0;m<2;m++)for(let M=0;M<g;M++)p.position.set(o*(-.46+.92*M/(g-1)),d+l/2,(m?1:-1)*(c/2+t*.013)),p.scale.set(t*.065,l*.94,t*.033),p.updateMatrix(),x.setMatrixAt(m*g+M,p.matrix);h.add(x);for(let m of[-1,1]){Or(h,s.box,f,o/2+t*.02,d+l*.5,m*c*.237,t*.04,l*.88,c*.45),Or(h,s.box,s.metal,o/2+t*.047,d+l*.5,m*c*.17,t*.028,l*.79,t*.038);for(let M of[-1,1])Or(h,s.box,s.metal,M*(o/2-t*.045),d+l/2,m*(c/2-t*.036),t*.09,l,t*.075)}}function Wp(n,e={}){typeof e=="number"&&(e={tile:e});let t=e.tile??n.grid.tile_size_m,i=e.unitHeight??t*.48;if(!Number.isFinite(t)||t<=0||!Number.isFinite(i)||i<=0)throw new Error("Obstacle display scale must be finite and positive.");let{rows:s,cols:r}=Gp(n.maps.padding);if(s!==n.grid.rows||r!==n.grid.cols||!Array.isArray(n.maps.action)||n.maps.action.length!==s||n.maps.action.some(l=>!Array.isArray(l)||l.length!==r||l.some(h=>!Number.isFinite(h))))throw new Error("Obstacle terrain must match the frame grid and contain finite heights.");let a=SM(n.maps.padding),o=new et,c={paper:e.style==="paper"};o.name="Terra obstacle props",o.userData.footprints=a;for(let l of a){let h=new et;h.name=`${l.kind}-${l.row}-${l.col}`,h.userData.footprint={...l};let d=1/0,u=-1/0;for(let f=l.row;f<l.row+l.rows;f++)for(let g=l.col;g<l.col+l.cols;g++){let x=n.maps.action[f][g]*i;d=Math.min(d,x),u=Math.max(u,x)}if(h.position.set((l.col+l.cols/2-r/2)*t,d+t*.008,(l.row+l.rows/2-s/2)*t),l.kind==="container")wM(h,l,t,u-d,c);else{c.stone||(c.stone=new Qe({vertexColors:!0,roughness:1,flatShading:!0}));let f=Vp(Math.min(l.rows,l.cols)*t*.62,t*.52,t*3.4)+u-d,g=new tt(EM(l.cols*t*.96,l.rows*t*.96,f,l.seed,!c.paper),c.stone);g.castShadow=!0,g.receiveShadow=!0,h.add(g)}o.add(h)}return o}function qp(n){let e=n>>>0||1;return()=>(e=Math.imul(e^e>>>15,739982445)+1831565813>>>0,e^=e>>>13,(e>>>0)/4294967295)}function Yp(n,e,t,i,s=!1){let r=Math.min(i,e*.98,t*.98),a=[],o=[[e-r,t-r,0],[-e+r,t-r,Math.PI/2],[-e+r,-t+r,Math.PI],[e-r,-t+r,Math.PI*1.5]];for(let[c,l,h]of o)for(let d=0;d<=6;d++){let u=h+d/6*Math.PI/2;a.push(new te(c+Math.cos(u)*r,l+Math.sin(u)*r))}return s&&a.reverse(),n?(n.setFromPoints(a),n):a}function TM(n,e,t,i,s){let r=qp(s),a=Yp(null,n,e,t),o=[a],c=[[.9,.34],[.66,.72],[.26,1]];for(let[m,M]of c)o.push(a.map(b=>{let v=.9+r()*.2;return new A(b.x*m*v,-i*M*(.85+r()*.3),b.y*m*v)}));o[0]=a.map(m=>new A(m.x,0,m.y));let l=[],h=[],d=new Le,u=[9206374,8219740,9864302,7299410],f=(m,M,b)=>{d.setHex(u[Math.floor(r()*u.length)]);for(let v of[m,M,b])l.push(v.x,v.y,v.z),h.push(d.r,d.g,d.b)};for(let m=0;m<o.length-1;m++)for(let M=0;M<a.length;M++){let b=(M+1)%a.length,v=o[m][M],T=o[m][b],w=o[m+1][M],C=o[m+1][b];f(v,w,T),f(T,w,C)}let g=new A(0,-i*1.25,0),x=o[o.length-1];for(let m=0;m<x.length;m++)f(x[m],g,x[(m+1)%x.length]);let p=new mt;return p.setAttribute("position",new nt(l,3)),p.setAttribute("color",new nt(h,3)),p.computeVertexNormals(),p}function AM(n,e=8){if(typeof document>"u")return null;let t=document.createElement("canvas");t.width=64,t.height=8;let i=t.getContext("2d");for(let r=0;r<e;r++){i.fillStyle=n[r%n.length],i.beginPath();let a=64/e;i.moveTo(r*a,0),i.lineTo(r*a+a,0),i.lineTo(r*a+a-4,8),i.lineTo(r*a-4,8),i.fill()}let s=new Nn(t);return s.colorSpace=Ft,s.wrapS=zi,s.anisotropy=4,s}function Xp(n){let e=new Qe({roughness:.9,flatShading:!0,...n});return e.onBeforeCompile=t=>{t.uniforms.uTime=on.uTime,t.vertexShader=`uniform float uTime;
${t.vertexShader}`.replace("#include <begin_vertex>",`#include <begin_vertex>
      #ifdef USE_INSTANCING
      float swayPhase = instanceMatrix[3].x * .37 + instanceMatrix[3].z * .23;
      float swayHeight = max(0., position.y + .5);
      transformed.x += sin(uTime * 1.3 + swayPhase) * .05 * swayHeight;
      transformed.z += cos(uTime * 1.1 + swayPhase) * .035 * swayHeight;
      #endif`)},e.customProgramCacheKey=()=>"terra-sway",e}var Bn=class{constructor(e,t,i,s){this.mesh=new $t(t,i,s),this.mesh.count=0,this.mesh.castShadow=!0,this.mesh.receiveShadow=!0,e.add(this.mesh),this.dummy=new pt,this.color=new Le}add(e,t,i,s,r,a,o,c=0,l=0){if(this.mesh.count>=this.mesh.instanceMatrix.count)return;let h=this.dummy;h.position.set(e,t,i),h.rotation.set(l,c,l*.6),h.scale.set(s,r,a),h.updateMatrix(),this.mesh.setMatrixAt(this.mesh.count,h.matrix),this.mesh.setColorAt(this.mesh.count,this.color.setHex(o)),this.mesh.count++}finish(){this.mesh.instanceMatrix.needsUpdate=!0,this.mesh.instanceColor&&(this.mesh.instanceColor.needsUpdate=!0),this.mesh.computeBoundingSphere()}};function Zt(n,e,t,i,s,r,a,o,c){let l=new tt(c,e);return l.position.set(t,i,s),l.scale.set(r,a,o),l.castShadow=!0,l.receiveShadow=!0,n.add(l),l}function RM(n,e,t,i,s){let r=new et;r.position.set(e,0,t),r.rotation.y=i,n.add(r);let{cube:a}=s,o=s.materials;Zt(r,o.concrete,0,.08,0,4.2,.16,2.5,a),Zt(r,o.office,0,1.4,0,4,2.5,2.3,a),Zt(r,o.trim,0,2.7,0,4.15,.14,2.45,a),Zt(r,o.trim,0,.22,0,4.1,.14,2.4,a);for(let l of[-1.2,.15])Zt(r,o.window,l,1.6,1.16,1,.75,.04,a),Zt(r,o.trim,l,1.18,1.19,1.1,.07,.08,a);Zt(r,o.door,1.35,1.15,1.16,.8,1.9,.05,a),Zt(r,o.concrete,1.35,.15,1.55,1.05,.3,.6,a),Zt(r,o.window,-2.005,1.6,0,.04,.7,1,a),Zt(r,o.metal,-1.2,2.95,-.4,.8,.36,.6,a),Zt(r,o.sign,.15,3.08,1.05,1.7,.46,.06,a);let c=new et;return c.position.set(1.3,0,-1.95),r.add(c),Zt(c,o.loo,0,1.12,0,1.05,2.24,1.05,a),Zt(c,o.looRoof,0,2.3,0,1.12,.12,1.12,a),Zt(c,o.trim,0,1.12,.53,.7,1.8,.03,a),r}function CM(n,e,t,i,s){let r=new et;r.position.set(e,0,t),r.rotation.y=i,n.add(r);let a=s.pipe;for(let[o,c]of[[-.55,.32],[0,.32],[.55,.32],[-.27,.8],[.27,.8],[0,1.27]]){let l=new tt(a,s.materials.pipe);l.rotation.x=Math.PI/2,l.position.set(o,c,0),l.scale.set(.3,3.2,.3),l.castShadow=!0,l.receiveShadow=!0,r.add(l);let h=new tt(s.ring,s.materials.pipeEnd);h.position.set(o,c,1.61),h.scale.setScalar(.3),r.add(h)}return Zt(r,s.materials.wood,0,.03,-1.1,1.8,.06,.2,s.cube),Zt(r,s.materials.wood,0,.03,1.1,1.8,.06,.2,s.cube),r}function PM(n,e,t,i,s,r){let a=new et;a.position.set(e,0,t),a.rotation.y=i,n.add(a),Zt(a,s.materials.wood,0,.07,0,1.2,.14,1,s.cube);let o=2+Math.floor(r()*3);for(let c=0;c<o;c++){let l=Zt(a,s.materials.bag,(c%2-.5)*.52,.28+Math.floor(c/2)*.26,0,.5,.24,.86,s.bagGeometry);l.rotation.y=(r()-.5)*.2}return a}function IM(n){let{rows:e,cols:t,tile_size_m:i}=n.grid,s=Math.max(e,t)*i,r=new et;r.name="Terra plinth";let a=new Bt(t*i,1,e*i);a.translate(0,-.5,0);let o=new tt(a,On("soil",{polygonOffset:!0,polygonOffsetFactor:1,polygonOffsetUnits:2},Xc.paper));return o.name="plinth",o.receiveShadow=!0,r.add(o),r.userData.extent={hx:t*i/2,hz:e*i/2},r.setFloor=c=>{let l=Math.min(-Math.max(s*.06,1.8),c-i*.8);o.position.y=c,o.scale.y=c-l,on.uFloor.value=c},r.update=()=>{},r.dispose=()=>{a.dispose(),o.material.dispose(),r.removeFromParent()},r}function $p(n,{style:e="diorama"}={}){if(e==="paper")return IM(n);let{rows:t,cols:i,tile_size_m:s}=n.grid,r=i*s/2,a=t*s/2,o=Math.max(t,i)*s,c=Vt.clamp(o*.2,6,22),l=new et;l.name="Terra surroundings";let h=qp(t*7919+i*104729+Math.round(s*1e3)),d=r+c,u=a+c,f=c*1.1,g={cube:new Bt(1,1,1),pipe:new tn(1,1,1,10,1,!0),ring:new wa(.72,1,10),bagGeometry:new Bt(1,1,1),materials:{concrete:new Qe({color:12170925,roughness:1}),office:new Qe({color:15986918,roughness:.8}),trim:new Qe({color:4157338,roughness:.7}),window:new Qe({color:10475238,roughness:.15,metalness:.1,emissive:1915460,emissiveIntensity:.3}),door:new Qe({color:3102072,roughness:.7}),metal:new Qe({color:13225680,roughness:.5}),sign:new Qe({color:15905329,roughness:.6}),loo:new Qe({color:3842264,roughness:.6}),looRoof:new Qe({color:15397621,roughness:.6}),pipe:new Qe({color:15040058,roughness:.7,side:xi}),pipeEnd:new Qe({color:12083499,roughness:.8,side:xi}),wood:new Qe({color:12159573,roughness:1}),bag:new Qe({color:15327433,roughness:1})}},x=Yp(new Di,d,u,f),p=new Ps;p.moveTo(-r,-a),p.lineTo(-r,a),p.lineTo(r,a),p.lineTo(r,-a),p.closePath(),x.holes.push(p);let m=new Hi(x,{depth:1,bevelEnabled:!1,curveSegments:6});m.rotateX(Math.PI/2);let M=new tt(m,On("island"));M.receiveShadow=!0,M.castShadow=!1,M.name="island-turf",l.add(M);let b=new tt(TM(d,u,f,Math.max(o*.16,4),t*31+i),new Qe({vertexColors:!0,flatShading:!0,roughness:1,side:xi}));b.name="island-underside",l.add(b);let v=Math.min(4.2,a*.5),T=new tt(new sn(1,1),On("soil",{color:13482134}));T.rotation.x=-Math.PI/2,T.scale.set(c,v,1),T.position.set(r+c/2+.01,.012,0),T.name="site-road",T.receiveShadow=!0,l.add(T);let w=v/2+.3,C=new Bn(l,g.cube,new Qe({color:15657696,roughness:.7}),400),_=new $t(g.cube,new Qe({map:AM(["#e8573a","#f7f2e8"]),color:typeof document>"u"?15226682:16777215,roughness:.7}),800);_.count=0,_.castShadow=!0,l.add(_);let E=new pt,P=.18,I=[[[-r-P,-a-P],[r+P,-a-P]],[[-r-P,a+P],[r+P,a+P]],[[-r-P,-a-P],[-r-P,a+P]],[[r+P,-a-P],[r+P,-w]],[[r+P,w],[r+P,a+P]]];for(let[[J,se],[fe,ge]]of I){let we=Math.hypot(fe-J,ge-se),Se=Math.max(1,Math.round(we/2.4)),D=Math.atan2(-(ge-se),fe-J);for(let Pe=0;Pe<=Se;Pe++){let Ze=Pe/Se;C.add(J+(fe-J)*Ze,.45,se+(ge-se)*Ze,.1,.9,.1,Pe%2?15657696:15226682)}for(let Pe=0;Pe<Se;Pe++)for(let Ze of[.42,.78]){let R=(Pe+.5)/Se;E.position.set(J+(fe-J)*R,Ze,se+(ge-se)*R),E.rotation.set(0,D,0),E.scale.set(we/Se,.09,.02),E.updateMatrix(),_.setMatrixAt(_.count++,E.matrix)}}C.finish(),_.instanceMatrix.needsUpdate=!0,_.computeBoundingSphere();let L=new yr(.2,.6,8);L.translate(0,.3,0);let X=new Bn(l,L,new Qe({color:16777215,roughness:.6,flatShading:!0}),40),W=new Bn(l,new tn(.115,.145,.1,8),new Qe({color:16777215,roughness:.4}),40);for(let J=0;J<Math.floor(c/2.2);J++)for(let se of[-1,1]){let fe=r+1+J*2.2,ge=se*(v/2+.35);X.add(fe,0,ge,1,1,1,15953706),W.add(fe,.33,ge,1,1,1,16250090)}X.finish(),W.finish();let U=[],z=(J,se,fe)=>{if(Math.abs(J)<r+1.2+fe&&Math.abs(se)<a+1.2+fe)return!1;let ge=Math.max(0,Math.abs(J)-(d-f)),we=Math.max(0,Math.abs(se)-(u-f));return Math.hypot(ge,we)>f-fe-.5||J>r&&Math.abs(se)<v/2+fe+.6?!1:U.every(([Se,D,Pe])=>Math.hypot(J-Se,se-D)>fe+Pe)},H=-(v/2+3.2),Q=r+Math.min(c*.55,6);c>=6&&z(Q,H,2.3)&&z(Q+1.3,H-1.95,.8)&&(RM(l,Q,H,0,g),U.push([Q,H,2.8],[Q+1.3,H-1.95,.9]));let ie=r+Math.min(c*.6,6.5),q=v/2+3;c>=6&&z(ie,q,1.7)&&(CM(l,ie,q,Math.PI/2+.1,g),U.push([ie,q,2]));for(let J=0;J<3;J++){let se=-r-c*(.35+h()*.3),fe=(h()-.5)*a*1.4;z(se,fe,.9)&&(PM(l,se,fe,h()*Math.PI,g,h),U.push([se,fe,.9]))}let Z=new tn(.5,.7,1,6);Z.translate(0,.5,0);let j=new yr(1,1,7);j.translate(0,.5,0);let de=new nn(1,0),Ge=new Bn(l,Z,new Qe({color:16777215,roughness:1,flatShading:!0}),400),me=new Bn(l,j,Xp({color:16777215}),900),k=new Bn(l,de,Xp({color:16777215}),700),ce=new Bn(l,new ga(1,0),new Qe({color:16777215,roughness:1,flatShading:!0}),200),ae=[5214042,6069343,4620114],Te=[8238678,9224541,6989903,10930522],Ue=[15905628,15306091],Oe=4*(d*u-r*a),st=Math.min(2600,Math.round(Oe/2.2));for(let J=0;J<st;J++){let se=h(),fe=Math.floor(h()*4),ge=Math.pow(h(),.8),we,Se;fe<2?(we=(se*2-1)*d,Se=(fe?1:-1)*(a+1.8+ge*(c-1.8))):(Se=(se*2-1)*u,we=(fe===3?1:-1)*(r+1.8+ge*(c-1.8)));let D=h(),Pe=.62+h()*.45,Ze=D<.45?1.1*Pe:D<.8?1.3*Pe:.7*Pe,R=Math.min(Math.abs(Math.abs(we)-r),Math.abs(Math.abs(Se)-a));if(h()>.25+Math.min(1,R/c)*.9||!z(we,Se,Ze))continue;U.push([we,Se,Ze]);let y=h()*Math.PI*2,F=(h()-.5)*.08;if(D<.45){let B=(3.4+h()*1.8)*Pe;Ge.add(we,0,Se,.18*Pe,B*.3,.18*Pe,9067835,y);for(let Y=0;Y<3;Y++)me.add(we,B*(.22+Y*.22),Se,(1.25-Y*.3)*Pe,B*.42,(1.25-Y*.3)*Pe,ae[(J+Y)%3],y+Y,F)}else if(D<.8){let B=(2.2+h()*1.4)*Pe;Ge.add(we,0,Se,.16*Pe,B*.55,.16*Pe,9725247,y),k.add(we,B*.75,Se,1.25*Pe,1.05*Pe,1.2*Pe,h()<.08?Ue[J%2]:Te[J%4],y,F),h()<.6&&k.add(we+.45*Pe,B*.98,Se-.2*Pe,.8*Pe,.7*Pe,.8*Pe,Te[(J+1)%4],y+1)}else D<.93?k.add(we,.35*Pe,Se,.7*Pe,.5*Pe,.7*Pe,Te[(J+2)%4],y):ce.add(we,.12*Pe,Se,.55*Pe,.38*Pe,.5*Pe,[10130828,9078399,10985879][J%3],y,F*3)}for(let J of[Ge,me,k,ce])J.finish();let He=new et,oe=new Qe({color:16777215,roughness:1,flatShading:!0,emissive:16777215,emissiveIntensity:.25}),ee=new nn(1,1),le=Math.hypot(d,u);for(let J=0;J<6;J++){let se=new et,fe=J/6*Math.PI*2+h()*.6,ge=le*(1.25+h()*.45),we=o*(.035+h()*.025);se.position.set(Math.cos(fe)*ge,o*(.12+h()*.22),Math.sin(fe)*ge);for(let Se=0;Se<5;Se++){let D=new tt(ee,oe);D.position.set((Se-2)*we*.9,(1-Math.abs(Se-2)*.45)*we*.35,(h()-.5)*we*.7),D.scale.setScalar(we*(1.1-Math.abs(Se-2)*.22)),se.add(D)}se.userData={angle:fe,distance:ge,height:se.position.y,speed:.006+h()*.006},He.add(se)}return l.add(He),l.userData.extent={hx:d,hz:u},l.setFloor=J=>{let se=Math.min(-Math.max(o*.07,2.2),J-s*.8);M.scale.y=-se,b.position.y=se,on.uFloor.value=J},l.update=J=>{for(let se of He.children){let{angle:fe,distance:ge,height:we,speed:Se}=se.userData,D=fe+J*Se;se.position.set(Math.cos(D)*ge,we+Math.sin(J*.3+fe*3)*o*.006,Math.sin(D)*ge)}},l.dispose=()=>{let J=new Set,se=new Set;l.traverse(fe=>{fe.geometry&&J.add(fe.geometry),fe.material&&se.add(fe.material)});for(let fe of J)fe.dispose();for(let fe of se)fe.map?.dispose(),fe.dispose();l.removeFromParent()},l}var Zp=9.81,Zc=[11039039,12157001,9723951,12881752],DM=new A(0,1,0),Jc=class extends et{constructor({groundHeight:e=()=>0}={}){super(),this.name="Terra effects",this.groundHeight=e,this.dummy=new pt,this.color=new Le;let t=new nn(1,0);this.clodMesh=new $t(t,new Qe({roughness:1,flatShading:!0}),320),this.clodMesh.castShadow=!0,this.clodMesh.frustumCulled=!1,this.clodMesh.count=0,this.add(this.clodMesh),this.puffMesh=new $t(new nn(1,1),new Qe({roughness:1,flatShading:!0}),260),this.puffMesh.frustumCulled=!1,this.puffMesh.count=0,this.add(this.puffMesh);for(let i of[this.clodMesh,this.puffMesh])i.setColorAt(0,this.color.setHex(16777215)),i.userData.skipAO=!0;this.clods=[],this.puffs=[],this.puffsEnabled=!0}throwClods(e,t,{count:i=10,flight:s=.45,spread:r=.3,size:a=.09,settle:o=!0,jitter:c=.08}={}){for(let l=0;l<i&&this.clods.length<320;l++){let h=e.clone().add(new A((Math.random()-.5)*c*2,(Math.random()-.5)*c,(Math.random()-.5)*c*2)),d=t.clone().add(new A((Math.random()-.5)*r*2,0,(Math.random()-.5)*r*2)),u=s*(.8+Math.random()*.4),f=Math.random()*s*.6,g=d.sub(h).multiplyScalar(1/u);g.y+=.5*Zp*u,this.clods.push({position:h,velocity:g,spin:new A(Math.random()*9,Math.random()*9,Math.random()*9),rotation:new Ri(Math.random()*6,Math.random()*6,0),size:a*(.6+Math.random()*.8),age:-f,life:u+(o?1.4:0),arrive:u,settle:o,bounced:!1,color:Zc[l%Zc.length]})}}burst(e,{count:t=12,speed:i=2.2,size:s=.07}={}){for(let r=0;r<t&&this.clods.length<320;r++){let a=Math.random()*Math.PI*2,o=.55+Math.random()*.5,c=new A(Math.cos(a)*(1-o),o*1.4,Math.sin(a)*(1-o)).multiplyScalar(i*(.6+Math.random()*.6));this.clods.push({position:e.clone(),velocity:c,spin:new A(Math.random()*12,Math.random()*12,0),rotation:new Ri,size:s*(.6+Math.random()*.8),age:-Math.random()*.08,life:2.2,arrive:1/0,settle:!0,bounced:!1,color:Zc[r%Zc.length]})}}puff(e,{count:t=6,size:i=.25,spread:s=.35,rise:r=.6,life:a=.9,color:o=15326402,drift:c=null}={}){if(this.puffsEnabled)for(let l=0;l<t&&this.puffs.length<260;l++){let h=new A((Math.random()-.5)*s*2,Math.random()*s*.5,(Math.random()-.5)*s*2),d=h.clone().multiplyScalar(1.4).add(DM.clone().multiplyScalar(r*(.6+Math.random()*.6)));c&&d.add(c),this.puffs.push({position:e.clone().add(h),velocity:d,size:i*(.6+Math.random()*.7),age:-Math.random()*.12,life:a*(.75+Math.random()*.5),color:o})}}update(e){e=Math.min(e,.05);let t=this.dummy;this.clods=this.clods.filter(s=>{if(s.age+=e,s.age<0)return!0;if(s.age>s.life)return!1;if((s.age<s.arrive||s.settle)&&!s.resting){s.velocity.y-=Zp*e,s.position.addScaledVector(s.velocity,e),s.rotation.x+=s.spin.x*e,s.rotation.y+=s.spin.y*e,s.rotation.z+=s.spin.z*e;let r=this.groundHeight(s.position.x,s.position.z)+s.size*.5;s.position.y<r&&s.velocity.y<0&&(s.position.y=r,!s.bounced&&s.velocity.y<-1.2?(s.velocity.y*=-.28,s.velocity.x*=.45,s.velocity.z*=.45,s.bounced=!0):(s.resting=!0,s.restAge=s.age))}return!(s.age>=s.arrive&&!s.settle)});let i=this.clods.length;for(let s=0;s<i;s++){let r=this.clods[s],a=r.age>=0,o=r.resting?Math.max(0,1-(r.age-r.restAge)/.9):1;t.position.copy(r.position),t.rotation.copy(r.rotation),t.scale.setScalar(a?r.size*o:0),t.updateMatrix(),this.clodMesh.setMatrixAt(s,t.matrix),this.clodMesh.setColorAt(s,this.color.setHex(r.color))}this.clodMesh.count=i,this.clodMesh.instanceMatrix.needsUpdate=!0,this.clodMesh.instanceColor&&(this.clodMesh.instanceColor.needsUpdate=!0),this.puffs=this.puffs.filter(s=>(s.age+=e,s.age<s.life));for(let s=0;s<this.puffs.length;s++){let r=this.puffs[s];r.age>=0&&(r.position.addScaledVector(r.velocity,e),r.velocity.multiplyScalar(Math.exp(-e*2.2)));let a=Math.max(0,r.age)/r.life,o=r.age<0?0:Math.sin(Math.min(1,a*1.25)*Math.PI)*(1+a*.6);t.position.copy(r.position),t.rotation.set(r.age*.7,r.age,0),t.scale.setScalar(r.size*o),t.updateMatrix(),this.puffMesh.setMatrixAt(s,t.matrix),this.puffMesh.setColorAt(s,this.color.setHex(r.color))}this.puffMesh.count=this.puffs.length,this.puffMesh.instanceMatrix.needsUpdate=!0,this.puffMesh.instanceColor&&(this.puffMesh.instanceColor.needsUpdate=!0)}clear(){this.clods=[],this.puffs=[],this.clodMesh.count=0,this.puffMesh.count=0}dispose(){this.clear();for(let e of[this.clodMesh,this.puffMesh])e.geometry.dispose(),e.material.dispose(),e.dispose();this.removeFromParent()}};var Br={name:"CopyShader",uniforms:{tDiffuse:{value:null},opacity:{value:1}},vertexShader:`

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


		}`};var Ui=class{constructor(){this.isPass=!0,this.enabled=!0,this.needsSwap=!0,this.clear=!1,this.renderToScreen=!1}setSize(){}render(){console.error("THREE.Pass: .render() must be implemented in derived pass.")}dispose(){}},LM=new ns(-1,1,1,-1,0,1),sd=class extends mt{constructor(){super(),this.setAttribute("position",new nt([-1,3,0,-1,-1,0,3,-1,0],3)),this.setAttribute("uv",new nt([0,2,0,0,2,0],2))}},NM=new sd,gs=class{constructor(e){this._mesh=new tt(NM,e)}dispose(){this._mesh.geometry.dispose()}render(e){e.render(this._mesh,LM)}get material(){return this._mesh.material}set material(e){this._mesh.material=e}};var _s=class extends Ui{constructor(e,t="tDiffuse"){super(),this.textureID=t,this.uniforms=null,this.material=null,e instanceof Rt?(this.uniforms=e.uniforms,this.material=e):e&&(this.uniforms=fi.clone(e.uniforms),this.material=new Rt({name:e.name!==void 0?e.name:"unspecified",defines:Object.assign({},e.defines),uniforms:this.uniforms,vertexShader:e.vertexShader,fragmentShader:e.fragmentShader})),this._fsQuad=new gs(this.material)}render(e,t,i){this.uniforms[this.textureID]&&(this.uniforms[this.textureID].value=i.texture),this._fsQuad.material=this.material,this.renderToScreen?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(t),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this._fsQuad.render(e))}dispose(){this.material.dispose(),this._fsQuad.dispose()}};var ao=class extends Ui{constructor(e,t){super(),this.scene=e,this.camera=t,this.clear=!0,this.needsSwap=!1,this.inverse=!1}render(e,t,i){let s=e.getContext(),r=e.state;r.buffers.color.setMask(!1),r.buffers.depth.setMask(!1),r.buffers.color.setLocked(!0),r.buffers.depth.setLocked(!0);let a,o;this.inverse?(a=0,o=1):(a=1,o=0),r.buffers.stencil.setTest(!0),r.buffers.stencil.setOp(s.REPLACE,s.REPLACE,s.REPLACE),r.buffers.stencil.setFunc(s.ALWAYS,a,4294967295),r.buffers.stencil.setClear(o),r.buffers.stencil.setLocked(!0),e.setRenderTarget(i),this.clear&&e.clear(),e.render(this.scene,this.camera),e.setRenderTarget(t),this.clear&&e.clear(),e.render(this.scene,this.camera),r.buffers.color.setLocked(!1),r.buffers.depth.setLocked(!1),r.buffers.color.setMask(!0),r.buffers.depth.setMask(!0),r.buffers.stencil.setLocked(!1),r.buffers.stencil.setFunc(s.EQUAL,1,4294967295),r.buffers.stencil.setOp(s.KEEP,s.KEEP,s.KEEP),r.buffers.stencil.setLocked(!0)}},Kc=class extends Ui{constructor(){super(),this.needsSwap=!1}render(e){e.state.buffers.stencil.setLocked(!1),e.state.buffers.stencil.setTest(!1)}};var jc=class{constructor(e,t){if(this.renderer=e,this._pixelRatio=e.getPixelRatio(),t===void 0){let i=e.getSize(new te);this._width=i.width,this._height=i.height,t=new Ht(this._width*this._pixelRatio,this._height*this._pixelRatio,{type:ei}),t.texture.name="EffectComposer.rt1"}else this._width=t.width,this._height=t.height;this.renderTarget1=t,this.renderTarget2=t.clone(),this.renderTarget2.texture.name="EffectComposer.rt2",this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2,this.renderToScreen=!0,this.passes=[],this.copyPass=new _s(Br),this.copyPass.material.blending=zt,this.timer=new Na}swapBuffers(){let e=this.readBuffer;this.readBuffer=this.writeBuffer,this.writeBuffer=e}addPass(e){this.passes.push(e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}insertPass(e,t){this.passes.splice(t,0,e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}removePass(e){let t=this.passes.indexOf(e);t!==-1&&this.passes.splice(t,1)}isLastEnabledPass(e){for(let t=e+1;t<this.passes.length;t++)if(this.passes[t].enabled)return!1;return!0}render(e){this.timer.update(),e===void 0&&(e=this.timer.getDelta());let t=this.renderer.getRenderTarget(),i=!1;for(let s=0,r=this.passes.length;s<r;s++){let a=this.passes[s];if(a.enabled!==!1){if(a.renderToScreen=this.renderToScreen&&this.isLastEnabledPass(s),a.render(this.renderer,this.writeBuffer,this.readBuffer,e,i),a.needsSwap){if(i){let o=this.renderer.getContext(),c=this.renderer.state.buffers.stencil;c.setFunc(o.NOTEQUAL,1,4294967295),this.copyPass.render(this.renderer,this.writeBuffer,this.readBuffer,e),c.setFunc(o.EQUAL,1,4294967295)}this.swapBuffers()}ao!==void 0&&(a instanceof ao?i=!0:a instanceof Kc&&(i=!1))}}this.renderer.setRenderTarget(t)}reset(e){if(e===void 0){let t=this.renderer.getSize(new te);this._pixelRatio=this.renderer.getPixelRatio(),this._width=t.width,this._height=t.height,e=this.renderTarget1.clone(),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.renderTarget1=e,this.renderTarget2=e.clone(),this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2}setSize(e,t){this._width=e,this._height=t;let i=this._width*this._pixelRatio,s=this._height*this._pixelRatio;this.renderTarget1.setSize(i,s),this.renderTarget2.setSize(i,s);for(let r=0;r<this.passes.length;r++)this.passes[r].setSize(i,s)}setPixelRatio(e){this._pixelRatio=e,this.setSize(this._width,this._height)}dispose(){this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.copyPass.dispose()}};var Qc=class extends Ui{constructor(e,t,i=null,s=null,r=null){super(),this.scene=e,this.camera=t,this.overrideMaterial=i,this.clearColor=s,this.clearAlpha=r,this.clear=!0,this.clearDepth=!1,this.needsSwap=!1,this.isRenderPass=!0,this._oldClearColor=new Le}render(e,t,i){let s=e.autoClear;e.autoClear=!1;let r,a;this.overrideMaterial!==null&&(a=this.scene.overrideMaterial,this.scene.overrideMaterial=this.overrideMaterial),this.clearColor!==null&&(e.getClearColor(this._oldClearColor),e.setClearColor(this.clearColor,e.getClearAlpha())),this.clearAlpha!==null&&(r=e.getClearAlpha(),e.setClearAlpha(this.clearAlpha)),this.clearDepth==!0&&e.clearDepth(),e.setRenderTarget(this.renderToScreen?null:i),this.clear===!0&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),e.render(this.scene,this.camera),this.clearColor!==null&&e.setClearColor(this._oldClearColor),this.clearAlpha!==null&&e.setClearAlpha(r),this.overrideMaterial!==null&&(this.scene.overrideMaterial=a),e.autoClear=s}};var oo={name:"GTAOShader",defines:{PERSPECTIVE_CAMERA:1,SAMPLES:16,NORMAL_VECTOR_TYPE:1,DEPTH_SWIZZLING:"x",SCREEN_SPACE_RADIUS:0,SCREEN_SPACE_RADIUS_SCALE:100,SCENE_CLIP_BOX:0},uniforms:{tNormal:{value:null},tDepth:{value:null},tNoise:{value:null},resolution:{value:new te},cameraNear:{value:null},cameraFar:{value:null},cameraProjectionMatrix:{value:new rt},cameraProjectionMatrixInverse:{value:new rt},cameraWorldMatrix:{value:new rt},radius:{value:.25},distanceExponent:{value:1},thickness:{value:1},distanceFallOff:{value:1},scale:{value:1},sceneBoxMin:{value:new A(-1,-1,-1)},sceneBoxMax:{value:new A(1,1,1)}},vertexShader:`

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
		}`},lo={name:"GTAODepthShader",defines:{PERSPECTIVE_CAMERA:1},uniforms:{tDepth:{value:null},cameraNear:{value:null},cameraFar:{value:null}},vertexShader:`
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

		}`},eh={name:"GTAOBlendShader",uniforms:{tDiffuse:{value:null},intensity:{value:1}},vertexShader:`
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
		}`};function Jp(n=5){let e=Math.floor(n)%2===0?Math.floor(n)+1:Math.floor(n),t=UM(e),i=t.length,s=new Uint8Array(i*4);for(let a=0;a<i;++a){let o=t[a],c=2*Math.PI*o/i,l=new A(Math.cos(c),Math.sin(c),0).normalize();s[a*4]=(l.x*.5+.5)*255,s[a*4+1]=(l.y*.5+.5)*255,s[a*4+2]=127,s[a*4+3]=255}let r=new Dn(s,e,e);return r.wrapS=zi,r.wrapT=zi,r.needsUpdate=!0,r}function UM(n){let e=Math.floor(n)%2===0?Math.floor(n)+1:Math.floor(n),t=e*e,i=Array(t).fill(0),s=Math.floor(e/2),r=e-1;for(let a=1;a<=t;){if(s===-1&&r===e?(r=e-2,s=0):(r===e&&(r=0),s<0&&(s=e-1)),i[s*e+r]!==0){r-=2,s++;continue}else i[s*e+r]=a++;r++,s--}return i}var co={name:"PoissonDenoiseShader",defines:{SAMPLES:16,SAMPLE_VECTORS:rd(16,2,1),NORMAL_VECTOR_TYPE:1,DEPTH_VALUE_SOURCE:0},uniforms:{tDiffuse:{value:null},tNormal:{value:null},tDepth:{value:null},tNoise:{value:null},resolution:{value:new te},cameraProjectionMatrixInverse:{value:new rt},lumaPhi:{value:5},depthPhi:{value:5},normalPhi:{value:5},radius:{value:4},index:{value:0}},vertexShader:`

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
		}`};function rd(n,e,t){let i=FM(n,e,t),s="vec3[SAMPLES](";for(let r=0;r<n;r++){let a=i[r];s+=`vec3(${a.x}, ${a.y}, ${a.z})${r<n-1?",":")"}`}return s}function FM(n,e,t){let i=[];for(let s=0;s<n;s++){let r=2*Math.PI*e*s/n,a=Math.pow(s/(n-1),t);i.push(new A(Math.cos(r),Math.sin(r),a))}return i}var th=class{constructor(e=Math){this.grad3=[[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]],this.grad4=[[0,1,1,1],[0,1,1,-1],[0,1,-1,1],[0,1,-1,-1],[0,-1,1,1],[0,-1,1,-1],[0,-1,-1,1],[0,-1,-1,-1],[1,0,1,1],[1,0,1,-1],[1,0,-1,1],[1,0,-1,-1],[-1,0,1,1],[-1,0,1,-1],[-1,0,-1,1],[-1,0,-1,-1],[1,1,0,1],[1,1,0,-1],[1,-1,0,1],[1,-1,0,-1],[-1,1,0,1],[-1,1,0,-1],[-1,-1,0,1],[-1,-1,0,-1],[1,1,1,0],[1,1,-1,0],[1,-1,1,0],[1,-1,-1,0],[-1,1,1,0],[-1,1,-1,0],[-1,-1,1,0],[-1,-1,-1,0]],this.p=[];for(let t=0;t<256;t++)this.p[t]=Math.floor(e.random()*256);this.perm=[];for(let t=0;t<512;t++)this.perm[t]=this.p[t&255];this.simplex=[[0,1,2,3],[0,1,3,2],[0,0,0,0],[0,2,3,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,3,0],[0,2,1,3],[0,0,0,0],[0,3,1,2],[0,3,2,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,3,2,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,0,3],[0,0,0,0],[1,3,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,3,0,1],[2,3,1,0],[1,0,2,3],[1,0,3,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,3,1],[0,0,0,0],[2,1,3,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,1,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,0,1,2],[3,0,2,1],[0,0,0,0],[3,1,2,0],[2,1,0,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,1,0,2],[0,0,0,0],[3,2,0,1],[3,2,1,0]]}noise(e,t){let i,s,r,a=.5*(Math.sqrt(3)-1),o=(e+t)*a,c=Math.floor(e+o),l=Math.floor(t+o),h=(3-Math.sqrt(3))/6,d=(c+l)*h,u=c-d,f=l-d,g=e-u,x=t-f,p,m;g>x?(p=1,m=0):(p=0,m=1);let M=g-p+h,b=x-m+h,v=g-1+2*h,T=x-1+2*h,w=c&255,C=l&255,_=this.perm[w+this.perm[C]]%12,E=this.perm[w+p+this.perm[C+m]]%12,P=this.perm[w+1+this.perm[C+1]]%12,I=.5-g*g-x*x;I<0?i=0:(I*=I,i=I*I*this._dot(this.grad3[_],g,x));let L=.5-M*M-b*b;L<0?s=0:(L*=L,s=L*L*this._dot(this.grad3[E],M,b));let X=.5-v*v-T*T;return X<0?r=0:(X*=X,r=X*X*this._dot(this.grad3[P],v,T)),70*(i+s+r)}noise3d(e,t,i){let s,r,a,o,l=(e+t+i)*.3333333333333333,h=Math.floor(e+l),d=Math.floor(t+l),u=Math.floor(i+l),f=1/6,g=(h+d+u)*f,x=h-g,p=d-g,m=u-g,M=e-x,b=t-p,v=i-m,T,w,C,_,E,P;M>=b?b>=v?(T=1,w=0,C=0,_=1,E=1,P=0):M>=v?(T=1,w=0,C=0,_=1,E=0,P=1):(T=0,w=0,C=1,_=1,E=0,P=1):b<v?(T=0,w=0,C=1,_=0,E=1,P=1):M<v?(T=0,w=1,C=0,_=0,E=1,P=1):(T=0,w=1,C=0,_=1,E=1,P=0);let I=M-T+f,L=b-w+f,X=v-C+f,W=M-_+2*f,U=b-E+2*f,z=v-P+2*f,H=M-1+3*f,Q=b-1+3*f,ie=v-1+3*f,q=h&255,Z=d&255,j=u&255,de=this.perm[q+this.perm[Z+this.perm[j]]]%12,Ge=this.perm[q+T+this.perm[Z+w+this.perm[j+C]]]%12,me=this.perm[q+_+this.perm[Z+E+this.perm[j+P]]]%12,k=this.perm[q+1+this.perm[Z+1+this.perm[j+1]]]%12,ce=.6-M*M-b*b-v*v;ce<0?s=0:(ce*=ce,s=ce*ce*this._dot3(this.grad3[de],M,b,v));let ae=.6-I*I-L*L-X*X;ae<0?r=0:(ae*=ae,r=ae*ae*this._dot3(this.grad3[Ge],I,L,X));let Te=.6-W*W-U*U-z*z;Te<0?a=0:(Te*=Te,a=Te*Te*this._dot3(this.grad3[me],W,U,z));let Ue=.6-H*H-Q*Q-ie*ie;return Ue<0?o=0:(Ue*=Ue,o=Ue*Ue*this._dot3(this.grad3[k],H,Q,ie)),32*(s+r+a+o)}noise4d(e,t,i,s){let r=this.grad4,a=this.simplex,o=this.perm,c=(Math.sqrt(5)-1)/4,l=(5-Math.sqrt(5))/20,h,d,u,f,g,x=(e+t+i+s)*c,p=Math.floor(e+x),m=Math.floor(t+x),M=Math.floor(i+x),b=Math.floor(s+x),v=(p+m+M+b)*l,T=p-v,w=m-v,C=M-v,_=b-v,E=e-T,P=t-w,I=i-C,L=s-_,X=E>P?32:0,W=E>I?16:0,U=P>I?8:0,z=E>L?4:0,H=P>L?2:0,Q=I>L?1:0,ie=X+W+U+z+H+Q,q=a[ie][0]>=3?1:0,Z=a[ie][1]>=3?1:0,j=a[ie][2]>=3?1:0,de=a[ie][3]>=3?1:0,Ge=a[ie][0]>=2?1:0,me=a[ie][1]>=2?1:0,k=a[ie][2]>=2?1:0,ce=a[ie][3]>=2?1:0,ae=a[ie][0]>=1?1:0,Te=a[ie][1]>=1?1:0,Ue=a[ie][2]>=1?1:0,Oe=a[ie][3]>=1?1:0,st=E-q+l,He=P-Z+l,oe=I-j+l,ee=L-de+l,le=E-Ge+2*l,J=P-me+2*l,se=I-k+2*l,fe=L-ce+2*l,ge=E-ae+3*l,we=P-Te+3*l,Se=I-Ue+3*l,D=L-Oe+3*l,Pe=E-1+4*l,Ze=P-1+4*l,R=I-1+4*l,y=L-1+4*l,F=p&255,B=m&255,Y=M&255,pe=b&255,_e=o[F+o[B+o[Y+o[pe]]]]%32,K=o[F+q+o[B+Z+o[Y+j+o[pe+de]]]]%32,ne=o[F+Ge+o[B+me+o[Y+k+o[pe+ce]]]]%32,Me=o[F+ae+o[B+Te+o[Y+Ue+o[pe+Oe]]]]%32,ke=o[F+1+o[B+1+o[Y+1+o[pe+1]]]]%32,ve=.6-E*E-P*P-I*I-L*L;ve<0?h=0:(ve*=ve,h=ve*ve*this._dot4(r[_e],E,P,I,L));let xe=.6-st*st-He*He-oe*oe-ee*ee;xe<0?d=0:(xe*=xe,d=xe*xe*this._dot4(r[K],st,He,oe,ee));let Be=.6-le*le-J*J-se*se-fe*fe;Be<0?u=0:(Be*=Be,u=Be*Be*this._dot4(r[ne],le,J,se,fe));let Xe=.6-ge*ge-we*we-Se*Se-D*D;Xe<0?f=0:(Xe*=Xe,f=Xe*Xe*this._dot4(r[Me],ge,we,Se,D));let Je=.6-Pe*Pe-Ze*Ze-R*R-y*y;return Je<0?g=0:(Je*=Je,g=Je*Je*this._dot4(r[ke],Pe,Ze,R,y)),27*(h+d+u+f+g)}_dot(e,t,i){return e[0]*t+e[1]*i}_dot3(e,t,i,s){return e[0]*t+e[1]*i+e[2]*s}_dot4(e,t,i,s,r){return e[0]*t+e[1]*i+e[2]*s+e[3]*r}};var ho=class n extends Ui{constructor(e,t,i=512,s=512,r,a,o){super(),this.width=i,this.height=s,this.clear=!0,this.camera=t,this.scene=e,this.output=0,this._renderGBuffer=!0,this._visibilityCache=[],this.blendIntensity=1,this.pdRings=2,this.pdRadiusExponent=2,this.pdSamples=16,this.gtaoNoiseTexture=Jp(),this.pdNoiseTexture=this._generateNoise(),this.gtaoRenderTarget=new Ht(this.width,this.height,{type:ei}),this.pdRenderTarget=this.gtaoRenderTarget.clone(),this.gtaoMaterial=new Rt({defines:Object.assign({},oo.defines),uniforms:fi.clone(oo.uniforms),vertexShader:oo.vertexShader,fragmentShader:oo.fragmentShader,blending:zt,depthTest:!1,depthWrite:!1}),this.gtaoMaterial.defines.PERSPECTIVE_CAMERA=this.camera.isPerspectiveCamera?1:0,this.gtaoMaterial.uniforms.tNoise.value=this.gtaoNoiseTexture,this.gtaoMaterial.uniforms.resolution.value.set(this.width,this.height),this.gtaoMaterial.uniforms.cameraNear.value=this.camera.near,this.gtaoMaterial.uniforms.cameraFar.value=this.camera.far,this.normalMaterial=new Aa,this.normalMaterial.blending=zt,this.pdMaterial=new Rt({defines:Object.assign({},co.defines),uniforms:fi.clone(co.uniforms),vertexShader:co.vertexShader,fragmentShader:co.fragmentShader,depthTest:!1,depthWrite:!1}),this.pdMaterial.uniforms.tDiffuse.value=this.gtaoRenderTarget.texture,this.pdMaterial.uniforms.tNoise.value=this.pdNoiseTexture,this.pdMaterial.uniforms.resolution.value.set(this.width,this.height),this.pdMaterial.uniforms.lumaPhi.value=10,this.pdMaterial.uniforms.depthPhi.value=2,this.pdMaterial.uniforms.normalPhi.value=3,this.pdMaterial.uniforms.radius.value=8,this.depthRenderMaterial=new Rt({defines:Object.assign({},lo.defines),uniforms:fi.clone(lo.uniforms),vertexShader:lo.vertexShader,fragmentShader:lo.fragmentShader,blending:zt}),this.depthRenderMaterial.uniforms.cameraNear.value=this.camera.near,this.depthRenderMaterial.uniforms.cameraFar.value=this.camera.far,this.copyMaterial=new Rt({uniforms:fi.clone(Br.uniforms),vertexShader:Br.vertexShader,fragmentShader:Br.fragmentShader,transparent:!0,depthTest:!1,depthWrite:!1,blendSrc:za,blendDst:Ls,blendEquation:Ti,blendSrcAlpha:Ba,blendDstAlpha:Ls,blendEquationAlpha:Ti}),this.blendMaterial=new Rt({uniforms:fi.clone(eh.uniforms),vertexShader:eh.vertexShader,fragmentShader:eh.fragmentShader,transparent:!0,depthTest:!1,depthWrite:!1,blending:Fl,blendSrc:za,blendDst:Ls,blendEquation:Ti,blendSrcAlpha:Ba,blendDstAlpha:Ls,blendEquationAlpha:Ti}),this._fsQuad=new gs(null),this._originalClearColor=new Le,this.setGBuffer(r?r.depthTexture:void 0,r?r.normalTexture:void 0),a!==void 0&&this.updateGtaoMaterial(a),o!==void 0&&this.updatePdMaterial(o)}setSize(e,t){this.width=e,this.height=t,this.gtaoRenderTarget.setSize(e,t),this.normalRenderTarget.setSize(e,t),this.pdRenderTarget.setSize(e,t),this.gtaoMaterial.uniforms.resolution.value.set(e,t),this.gtaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.gtaoMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this.pdMaterial.uniforms.resolution.value.set(e,t),this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse)}dispose(){this.gtaoNoiseTexture.dispose(),this.pdNoiseTexture.dispose(),this.normalRenderTarget.dispose(),this.gtaoRenderTarget.dispose(),this.pdRenderTarget.dispose(),this.normalMaterial.dispose(),this.pdMaterial.dispose(),this.copyMaterial.dispose(),this.depthRenderMaterial.dispose(),this._fsQuad.dispose()}get gtaoMap(){return this.pdRenderTarget.texture}setGBuffer(e,t){e!==void 0?(this.depthTexture=e,this.normalTexture=t,this._renderGBuffer=!1):(this.depthTexture=new en,this.depthTexture.format=xn,this.depthTexture.type=hs,this.normalRenderTarget=new Ht(this.width,this.height,{minFilter:Ot,magFilter:Ot,type:ei,depthTexture:this.depthTexture}),this.normalTexture=this.normalRenderTarget.texture,this._renderGBuffer=!0);let i=this.normalTexture?1:0,s=this.depthTexture===this.normalTexture?"w":"x";this.gtaoMaterial.defines.NORMAL_VECTOR_TYPE=i,this.gtaoMaterial.defines.DEPTH_SWIZZLING=s,this.gtaoMaterial.uniforms.tNormal.value=this.normalTexture,this.gtaoMaterial.uniforms.tDepth.value=this.depthTexture,this.pdMaterial.defines.NORMAL_VECTOR_TYPE=i,this.pdMaterial.defines.DEPTH_SWIZZLING=s,this.pdMaterial.uniforms.tNormal.value=this.normalTexture,this.pdMaterial.uniforms.tDepth.value=this.depthTexture,this.depthRenderMaterial.uniforms.tDepth.value=this.normalRenderTarget.depthTexture}setSceneClipBox(e){e?(this.gtaoMaterial.needsUpdate=this.gtaoMaterial.defines.SCENE_CLIP_BOX!==1,this.gtaoMaterial.defines.SCENE_CLIP_BOX=1,this.gtaoMaterial.uniforms.sceneBoxMin.value.copy(e.min),this.gtaoMaterial.uniforms.sceneBoxMax.value.copy(e.max)):(this.gtaoMaterial.needsUpdate=this.gtaoMaterial.defines.SCENE_CLIP_BOX===0,this.gtaoMaterial.defines.SCENE_CLIP_BOX=0)}updateGtaoMaterial(e){e.radius!==void 0&&(this.gtaoMaterial.uniforms.radius.value=e.radius),e.distanceExponent!==void 0&&(this.gtaoMaterial.uniforms.distanceExponent.value=e.distanceExponent),e.thickness!==void 0&&(this.gtaoMaterial.uniforms.thickness.value=e.thickness),e.distanceFallOff!==void 0&&(this.gtaoMaterial.uniforms.distanceFallOff.value=e.distanceFallOff,this.gtaoMaterial.needsUpdate=!0),e.scale!==void 0&&(this.gtaoMaterial.uniforms.scale.value=e.scale),e.samples!==void 0&&e.samples!==this.gtaoMaterial.defines.SAMPLES&&(this.gtaoMaterial.defines.SAMPLES=e.samples,this.gtaoMaterial.needsUpdate=!0),e.screenSpaceRadius!==void 0&&(e.screenSpaceRadius?1:0)!==this.gtaoMaterial.defines.SCREEN_SPACE_RADIUS&&(this.gtaoMaterial.defines.SCREEN_SPACE_RADIUS=e.screenSpaceRadius?1:0,this.gtaoMaterial.needsUpdate=!0)}updatePdMaterial(e){let t=!1;e.lumaPhi!==void 0&&(this.pdMaterial.uniforms.lumaPhi.value=e.lumaPhi),e.depthPhi!==void 0&&(this.pdMaterial.uniforms.depthPhi.value=e.depthPhi),e.normalPhi!==void 0&&(this.pdMaterial.uniforms.normalPhi.value=e.normalPhi),e.radius!==void 0&&e.radius!==this.radius&&(this.pdMaterial.uniforms.radius.value=e.radius),e.radiusExponent!==void 0&&e.radiusExponent!==this.pdRadiusExponent&&(this.pdRadiusExponent=e.radiusExponent,t=!0),e.rings!==void 0&&e.rings!==this.pdRings&&(this.pdRings=e.rings,t=!0),e.samples!==void 0&&e.samples!==this.pdSamples&&(this.pdSamples=e.samples,t=!0),t&&(this.pdMaterial.defines.SAMPLES=this.pdSamples,this.pdMaterial.defines.SAMPLE_VECTORS=rd(this.pdSamples,this.pdRings,this.pdRadiusExponent),this.pdMaterial.needsUpdate=!0)}render(e,t,i){switch(this._renderGBuffer&&(this._overrideVisibility(),this._renderOverride(e,this.normalMaterial,this.normalRenderTarget,7829503,1),this._restoreVisibility()),this.gtaoMaterial.uniforms.cameraNear.value=this.camera.near,this.gtaoMaterial.uniforms.cameraFar.value=this.camera.far,this.gtaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.gtaoMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this.gtaoMaterial.uniforms.cameraWorldMatrix.value.copy(this.camera.matrixWorld),this._renderPass(e,this.gtaoMaterial,this.gtaoRenderTarget,16777215,1),this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse),this._renderPass(e,this.pdMaterial,this.pdRenderTarget,16777215,1),this.output){case n.OUTPUT.Off:break;case n.OUTPUT.Diffuse:this.copyMaterial.uniforms.tDiffuse.value=i.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.AO:this.copyMaterial.uniforms.tDiffuse.value=this.gtaoRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Denoise:this.copyMaterial.uniforms.tDiffuse.value=this.pdRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Depth:this.depthRenderMaterial.uniforms.cameraNear.value=this.camera.near,this.depthRenderMaterial.uniforms.cameraFar.value=this.camera.far,this._renderPass(e,this.depthRenderMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Normal:this.copyMaterial.uniforms.tDiffuse.value=this.normalRenderTarget.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t);break;case n.OUTPUT.Default:this.copyMaterial.uniforms.tDiffuse.value=i.texture,this.copyMaterial.blending=zt,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:t),this.blendMaterial.uniforms.intensity.value=this.blendIntensity,this.blendMaterial.uniforms.tDiffuse.value=this.pdRenderTarget.texture,this._renderPass(e,this.blendMaterial,this.renderToScreen?null:t);break;default:console.warn("THREE.GTAOPass: Unknown output type.")}}_renderPass(e,t,i,s,r){e.getClearColor(this._originalClearColor);let a=e.getClearAlpha(),o=e.autoClear;e.setRenderTarget(i),e.autoClear=!1,s!=null&&(e.setClearColor(s),e.setClearAlpha(r||0),e.clear()),this._fsQuad.material=t,this._fsQuad.render(e),e.autoClear=o,e.setClearColor(this._originalClearColor),e.setClearAlpha(a)}_renderOverride(e,t,i,s,r){e.getClearColor(this._originalClearColor);let a=e.getClearAlpha(),o=e.autoClear;e.setRenderTarget(i),e.autoClear=!1,s=t.clearColor||s,r=t.clearAlpha||r,s!=null&&(e.setClearColor(s),e.setClearAlpha(r||0),e.clear()),this.scene.overrideMaterial=t,e.render(this.scene,this.camera),this.scene.overrideMaterial=null,e.autoClear=o,e.setClearColor(this._originalClearColor),e.setClearAlpha(a)}_overrideVisibility(){let e=this.scene,t=this._visibilityCache;e.traverse(function(i){(i.isPoints||i.isLine||i.isLine2)&&i.visible&&(i.visible=!1,t.push(i))})}_restoreVisibility(){let e=this._visibilityCache;for(let t=0;t<e.length;t++)e[t].visible=!0;e.length=0}_generateNoise(e=64){let t=new th,i=e*e*4,s=new Uint8Array(i);for(let a=0;a<e;a++)for(let o=0;o<e;o++){let c=a,l=o;s[(a*e+o)*4]=(t.noise(c,l)*.5+.5)*255,s[(a*e+o)*4+1]=(t.noise(c+e,l)*.5+.5)*255,s[(a*e+o)*4+2]=(t.noise(c,l+e)*.5+.5)*255,s[(a*e+o)*4+3]=(t.noise(c+e,l+e)*.5+.5)*255}let r=new Dn(s,e,e,vi,li);return r.wrapS=zi,r.wrapT=zi,r.needsUpdate=!0,r}};ho.OUTPUT={Off:-1,Default:0,Diffuse:1,Depth:2,Normal:3,AO:4,Denoise:5};var uo={name:"OutputShader",uniforms:{tDiffuse:{value:null},toneMappingExposure:{value:1}},vertexShader:`
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

		}`};var ih=class extends Ui{constructor(){super(),this.isOutputPass=!0,this.uniforms=fi.clone(uo.uniforms),this.material=new Sr({name:uo.name,uniforms:this.uniforms,vertexShader:uo.vertexShader,fragmentShader:uo.fragmentShader}),this._fsQuad=new gs(this.material),this._outputColorSpace=null,this._toneMapping=null}render(e,t,i){this.uniforms.tDiffuse.value=i.texture,this.uniforms.toneMappingExposure.value=e.toneMappingExposure,(this._outputColorSpace!==e.outputColorSpace||this._toneMapping!==e.toneMapping)&&(this._outputColorSpace=e.outputColorSpace,this._toneMapping=e.toneMapping,this.material.defines={},ht.getTransfer(this._outputColorSpace)===ft&&(this.material.defines.SRGB_TRANSFER=""),this._toneMapping===ka?this.material.defines.LINEAR_TONE_MAPPING="":this._toneMapping===Ha?this.material.defines.REINHARD_TONE_MAPPING="":this._toneMapping===Va?this.material.defines.CINEON_TONE_MAPPING="":this._toneMapping===os?this.material.defines.ACES_FILMIC_TONE_MAPPING="":this._toneMapping===Wa?this.material.defines.AGX_TONE_MAPPING="":this._toneMapping===Ns?this.material.defines.NEUTRAL_TONE_MAPPING="":this._toneMapping===Ga&&(this.material.defines.CUSTOM_TONE_MAPPING=""),this.material.needsUpdate=!0),this.renderToScreen===!0?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(t),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this._fsQuad.render(e))}dispose(){this.material.dispose(),this._fsQuad.dispose()}};var Kp={name:"FXAAShader",uniforms:{tDiffuse:{value:null},resolution:{value:new te(1/1024,1/512)}},vertexShader:`

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

		}`};var nh=class extends _s{constructor(){super(Kp)}setSize(e,t){this.material.uniforms.resolution.value.set(1/e,1/t)}};var ad=class extends ho{_overrideVisibility(){super._overrideVisibility();let e=this._visibilityCache;this.scene.traverse(t=>{(t.isSprite||t.isLineSegments2||t.userData.skipAO)&&t.visible&&(t.visible=!1,e.push(t))})}_renderOverride(e,...t){let i=this.scene.background;this.scene.background=null,super._renderOverride(e,...t),this.scene.background=i}},OM={uniforms:{tDiffuse:{value:null},tDepth:{value:null},tNormal:{value:null},resolution:{value:new te(1,1)},cameraNear:{value:.1},cameraFar:{value:1e3},thickness:{value:1},strength:{value:.92},vignette:{value:.16}},vertexShader:"varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }",fragmentShader:`
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
    }`},sh=class{constructor(e,t,i){this.renderer=e,this.scene=t,this.camera=i;let s=e.getDrawingBufferSize(new te),r=new Ht(s.x,s.y,{type:ei,samples:4});this.composer=new jc(e,r),this.composer.addPass(new Qc(t,i)),this.ao=new ad(t,i,s.x,s.y),this.ao.blendIntensity=.9,this.composer.addPass(this.ao),this.ink=new _s(OM),this.ink.uniforms.tDepth.value=this.ao.depthTexture,this.ink.uniforms.tNormal.value=this.ao.normalTexture,this.composer.addPass(this.ink),this.composer.addPass(new ih),this.composer.addPass(new nh)}configure({span:e,tile:t}){this.ao.updateGtaoMaterial({radius:Math.max(t*1.6,e*.018),distanceExponent:1.4,thickness:1.2,scale:1.05,samples:16}),this.ao.updatePdMaterial({lumaPhi:10,depthPhi:2,normalPhi:3,radius:6,rings:2,samples:12})}setSize(e,t,i){this.composer.setPixelRatio(i),this.composer.setSize(e,t);let s=this.renderer.getDrawingBufferSize(new te);this.ink.uniforms.resolution.value.copy(s),this.ink.uniforms.thickness.value=1.35*i}setLook({vignette:e=0}={}){this.ink.uniforms.vignette.value=e}setSamples(e){for(let t of[this.composer.renderTarget1,this.composer.renderTarget2])t.samples!==e&&(t.samples=e,t.dispose())}render(){this.ink.uniforms.cameraNear.value=this.camera.near,this.ink.uniforms.cameraFar.value=this.camera.far,this.composer.render()}dispose(){this.ao.dispose(),this.composer.dispose()}};var Vs={dig:{color:15113984,opacity:.55,map:"target",pattern:"hatch",test:n=>n<0},dump:{color:40563,opacity:.5,map:"target",pattern:"dots",test:n=>n>0},restricted:{color:13983232,opacity:.42,map:"dumpability_static",pattern:"cross",test:n=>!n},dumpability:{color:29362,opacity:.28,map:"dumpability",pattern:"solid",test:n=>!!n},interaction:{color:5682409,opacity:.2,map:"_workspace",pattern:"solid",test:n=>!!n},footprint:{color:2575449,opacity:.12,map:"footprint",pattern:"solid",test:n=>!!n},precision:{color:7689141,opacity:.08,map:"precision_required_band",pattern:"solid",test:n=>!!n},eligibleNow:{color:39784,opacity:.8,map:"_digging",pattern:"solid",test:n=>n===1},eligibleSwing:{color:14983959,opacity:.82,map:"_digging",pattern:"dots",test:n=>n===2},eligibleBlocked:{color:13061445,opacity:.58,map:"_digging",pattern:"hatch",test:n=>n===3},eligibleLoaded:{color:9608096,opacity:.85,map:"_digging",pattern:"solid",test:n=>n===4}},BM=["eligibleNow","eligibleSwing","eligibleBlocked","eligibleLoaded"],dw=new rt,zn=new pt,jp=new Le,zM=n=>n*n*(3-2*n),kM=n=>n<.5?4*n*n*n:1-(-2*n+2)**3/2,HM=n=>1+(1.4+1)*(n-1)**3+1.4*(n-1)**2,rh=(n,e,t)=>n+(e-n)*t,kn=(n,e,t)=>Math.min(t,Math.max(e,n)),VM=["dig","dump","transfer"],Qp="terra-viewer3d-quality",em="terra-viewer3d-presentation";function tm(n){if(!n)return;let e=new Set,t=new Set;n.traverse(i=>{if(i.geometry&&e.add(i.geometry),i.material)for(let s of Array.isArray(i.material)?i.material:[i.material])t.add(s)});for(let i of e)i.dispose();for(let i of t)i.dispose();n.removeFromParent(),n.clear()}function im(n){try{return localStorage.getItem(n)}catch{return null}}function nm(n,e){try{localStorage.setItem(n,e)}catch{}}var ah=class{constructor(e,{onPick:t,onCameraChange:i,onError:s,onQualityChange:r}={}){this.element=e,this.onPick=t,this.onCameraChange=i,this.onError=s,this.onQualityChange=r,this.scene=new Cs,this.renderer=new Cc({antialias:!0,alpha:!1,preserveDrawingBuffer:!0,powerPreference:"high-performance"}),this.pixelRatio=Math.min(window.devicePixelRatio||1,2),this.renderer.setPixelRatio(this.pixelRatio),this.renderer.shadowMap.enabled=!0,this.renderer.shadowMap.type=Ds,this.renderer.toneMapping=os,this.renderer.toneMappingExposure=1,this.renderer.outputColorSpace=Ft,e.appendChild(this.renderer.domElement),this.renderer.domElement.addEventListener("webglcontextlost",l=>{l.preventDefault(),this.onError?.(new Error("The graphics context was lost. Reload the viewer to reconnect to the scene. Your live episode remains on the server."))});let a=new Lr(this.renderer);this.scene.environment=a.fromScene(new Nc,.04).texture,this.scene.environmentIntensity=.28,a.dispose(),this.camera=new Kt(32,1,.1,2e3),this.controls=new Lc(this.camera,this.renderer.domElement),this.controls.enableDamping=!0,this.controls.dampingFactor=.085,this.controls.maxPolarAngle=Math.PI*.47,this.controls.minPolarAngle=.001,this.controls.screenSpacePanning=!0,this.controls.addEventListener("start",()=>{this.tween=null,this.follow&&(this.follow=!1,this.onCameraChange?.({follow:!1}))}),this.hemi=new Pa(13624575,10122832,.8),this.scene.add(this.hemi),this.sun=new wr(16771529,2.7),this.sun.castShadow=!0,this.sun.shadow.mapSize.set(2048,2048),this.sun.shadow.bias=-4e-4,this.sun.shadow.radius=3,this.scene.add(this.sun),this.scene.add(this.sun.target),this.fill=new wr(11127295,.35),this.scene.add(this.fill),this.world=new et,this.scene.add(this.world),this.machines=new Map,this.heightScale=1,this.visibility={dig:!0,dump:!0,restricted:!1,dumpability:!1,interaction:!0,footprint:!0,precision:!0,eligibility:!0,eligibleNow:!0,eligibleSwing:!0,eligibleBlocked:!0,eligibleLoaded:!0,grid:!1,tags:!0},this.raycaster=new Ua,this.pointer=new te,this.selected=null,this.reducedMotion=window.matchMedia("(prefers-reduced-motion: reduce)").matches,this.effects=new Jc({groundHeight:(l,h)=>this.groundAt(l,h)}),this.scene.add(this.effects);let o=im(Qp);this.quality=o==="fast"?"fast":"high",this.perf=o?null:{frames:0,elapsed:0},this.presentation=im(em)==="diorama"?"diorama":"paper";try{this.post=new sh(this.renderer,this.scene,this.camera)}catch(l){console.warn("Post-processing unavailable",l),this.post=null,this.quality="fast"}this.lineMaterials=new Set,this.applyLook();let c=null;e.addEventListener("pointerdown",l=>{c={x:l.clientX,y:l.clientY,button:l.button}}),e.addEventListener("pointerup",l=>{c?.button===0&&Math.hypot(l.clientX-c.x,l.clientY-c.y)<5&&this.pick(l),c=null}),this.resizeObserver=new ResizeObserver(()=>this.resize()),this.resizeObserver.observe(e),this.resize(),this.clock={last:performance.now(),idle:0},this.renderer.setAnimationLoop(l=>{this.update(l),this.controls.update(),this.render()})}render(){this.quality==="high"&&this.post?this.post.render():this.renderer.render(this.scene,this.camera)}measure(e){if(!this.perf||!this.frame||this.quality!=="high"||document.hidden||(++this.perf.frames>20&&(this.perf.elapsed+=e),this.perf.frames<110))return;let t=this.perf.elapsed/(this.perf.frames-20);this.perf=null,t>1/24&&(this.quality="fast",this.onQualityChange?.("fast"))}setQuality(e){return this.quality=e==="fast"||!this.post?"fast":"high",nm(Qp,this.quality),this.quality}applyLook(){let e=this.presentation==="paper";this.palette=Xc[this.presentation],this.scene.background=e?new Le(12,12,12):this.sky||(this.sky=Hp()),this.scene.fog=!e&&this.span?new oa(new Le(ms.sky[1]),this.span*3.2,this.span*7.5):null,this.renderer.toneMapping=e?Ns:os,this.hemi.color.set(e?16777215:13624575),this.hemi.groundColor.set(e?9275520:10122832),this.hemi.intensity=e?.9:.8,this.sun.color.set(e?16777215:16771529),this.sun.intensity=e?2.3:2.7,this.fill.color.set(e?16777215:11127295),this.fill.intensity=e?.45:.35,this.effects.puffsEnabled=!e,on.uMotion.value=e?0:1,this.post?.setLook({vignette:e?0:.16}),this.element.ownerDocument?.body&&(this.element.ownerDocument.body.dataset.presentation=this.presentation)}setPresentation(e){if(this.presentation=e==="diorama"?"diorama":"paper",nm(em,this.presentation),this.applyLook(),this.frame){let t=this.camera.position.clone(),i=this.controls.target.clone();this.setFrame(this.frame,{reset:!0}),this.tween=null,this.camera.position.copy(t),this.controls.target.copy(i),this.controls.update()}return this.presentation}resize(){let e=this.element.clientWidth||1,t=this.element.clientHeight||1;this.renderer.setSize(e,t,!1),this.camera.aspect=e/t,this.camera.updateProjectionMatrix(),this.post?.setSize(e,t,this.pixelRatio);let i=this.renderer.getDrawingBufferSize(new te);for(let s of this.lineMaterials??[])s.resolution.copy(i)}point(e,t,i=0){let{rows:s,cols:r,tile_size_m:a}=this.frame.grid;return new A((t+.5-r/2)*a,i*this.unitHeight,(e+.5-s/2)*a)}heightAt(e,t,i=this.frame){e=kn(Math.round(e),0,i.grid.rows-1),t=kn(Math.round(t),0,i.grid.cols-1);let s=i.maps.action[e][t];return s>0&&!i.maps.padding[e][t]?this.piles.endpointHeight(e,t,i===this.piles.previous):s*this.unitHeight}displayHeightAt(e,t){e=kn(Math.round(e),0,this.frame.grid.rows-1),t=kn(Math.round(t),0,this.frame.grid.cols-1);let i=this.displayHeights?.[e]?.[t]??this.frame.maps.action[e][t];return i>0&&!this.frame.maps.padding[e][t]?this.piles.nodeHeight(e*2+1,t*2+1):i*this.unitHeight}surfacePoint(e,t){let i=this.point(e,t);return i.y=this.displayHeightAt(e,t),i}groundAt(e,t){if(!this.frame)return 0;let{rows:i,cols:s,tile_size_m:r}=this.frame.grid,a=Math.floor(e/r+s/2),o=Math.floor(t/r+i/2);return o<0||a<0||o>=i||a>=s?0:this.frame.maps.padding[o][a]?this.obstacleTop??0:this.displayHeightAt(o,a)}buildWorld(e){this.terrain&&this.disposeWorld();let{rows:t,cols:i,tile_size_m:s}=e.grid,r=t*i;this.span=Math.max(t,i)*s,on.uTile.value=s,this.camera.near=Math.max(s*.025,this.span/200),this.camera.far=this.span*20,this.camera.updateProjectionMatrix(),this.controls.minDistance=Math.max(s*2,this.span*.08),this.controls.maxDistance=this.span*4.5,this.applyLook();let a=this.span*.5+Vt.clamp(this.span*.2,6,22)+2;this.sun.position.set(-this.span*.75,this.span*1.35,-this.span*.45),Object.assign(this.sun.shadow.camera,{left:-a,right:a,top:a,bottom:-a,near:.1,far:this.span*4}),this.sun.shadow.camera.updateProjectionMatrix(),this.sun.shadow.normalBias=s*.04,this.fill.position.set(this.span*.8,this.span*.6,this.span*.9),this.post?.configure({span:this.span,tile:s});let o=On("soil",{polygonOffset:!0,polygonOffsetFactor:1,polygonOffsetUnits:2},this.palette),c=On("soil",{color:16777215,polygonOffset:!0,polygonOffsetFactor:-1,polygonOffsetUnits:-2},this.palette),l=new Qe({color:9204051,roughness:1});for(let h of[o,c,l])h.shadowSide=ji;this.terrain=new $t(new Bt(1,1,1),[o,o,c,l,o,o],r),this.terrain.instanceMatrix.setUsage(Pr),this.terrain.castShadow=!0,this.terrain.receiveShadow=!0,this.world.add(this.terrain),this.environment=$p(e,{style:this.presentation}),this.world.add(this.environment),this.layers={},this.boundaries={},this.boundaryEntries=new Map;for(let[h,d]of Object.entries(Vs)){let u=new sn(1,1);u.rotateX(-Math.PI/2);let f=qc({color:d.color,opacity:d.opacity,pattern:d.pattern,polygonOffset:!0,polygonOffsetFactor:-2}),g=new $t(u,f,r);if(g.instanceMatrix.setUsage(Pr),g.visible=this.visibility[h],g.renderOrder=3+Object.keys(this.layers).length,g.frustumCulled=!1,g.userData.skipAO=!0,this.layers[h]=g,this.world.add(g),h==="dig"||h==="dump"||h==="interaction"||h==="precision"||h==="footprint"){let x=new Fr({color:new Le(d.color).multiplyScalar(h==="precision"?1:.82),linewidth:h==="precision"?3.5:2.6,transparent:!0,opacity:.95,depthWrite:!1});x.resolution.copy(this.renderer.getDrawingBufferSize(new te)),this.lineMaterials.add(x);let p=new Bc(new Bs,x);p.renderOrder=h==="precision"?18:14,p.visible=this.visibility[h],p.frustumCulled=!1,this.boundaries[h]=p,this.world.add(p)}}this.gridLines=new Qn(new mt,new Ln({color:7033138,transparent:!0,opacity:.22,depthWrite:!1})),this.gridLines.renderOrder=10,this.gridLines.visible=this.visibility.grid,this.gridLines.frustumCulled=!1,this.world.add(this.gridLines),this.selection=new Qn(new _a(new Bt(s*.99,s*.04,s*.99)),new Ln({color:16776160,depthTest:!1})),this.selection.renderOrder=20,this.selection.visible=!1,this.world.add(this.selection)}disposeWorld(){this.clearMotion(),this.effects.clear();for(let i of this.machines.values())this.scene.remove(i.root),i.dispose();this.machines.clear();let e=new Set,t=new Set;this.world.traverse(i=>{if(i.geometry&&e.add(i.geometry),i.material)for(let s of Array.isArray(i.material)?i.material:[i.material])t.add(s)});for(let i of e)i.dispose();for(let i of t)i.map?.dispose(),i.dispose();this.world.clear(),this.selected=null,this.piles=null,this.obstacleProps=null,this.environment=null,this.lineMaterials.clear()}setFrame(e,{animate:t=!1,duration:i=650,reset:s=!1}={}){let r=this.frame,a=!r||r.grid.rows!==e.grid.rows||r.grid.cols!==e.grid.cols||r.grid.tile_size_m!==e.grid.tile_size_m;this.clearMotion(),this.frame=e,this.unitHeight=e.grid.tile_size_m*.48*this.heightScale,on.uUnit.value=this.unitHeight;let o=so(e);this.layerMaps={...e.maps,_workspace:e.maps.work_cone??e.maps.interaction,_digging:o.available?o.cells:null},(a||s)&&this.buildWorld(e);let c=Vc(r,e),l=t&&!s&&!a&&!this.reducedMotion&&r&&!r.done&&e.step===r.step+1,h=0;for(let u of e.maps.action)for(let f of u)h=Math.min(h,f);if(this.finalFloor=h*this.unitHeight-e.grid.tile_size_m*.85,l)for(let u of r.maps.action)for(let f of u)h=Math.min(h,f);this.setFloor(h*this.unitHeight-e.grid.tile_size_m*.85),tm(this.obstacleProps),this.obstacleProps=Wp(e,{unitHeight:this.unitHeight,style:this.presentation}),this.world.add(this.obstacleProps),tm(this.piles),this.piles=new $c({...e,maps:this.layerMaps},{previous:l?r:null,unitHeight:this.unitHeight,layerSettings:Vs,visibility:this.visibility,palette:this.palette}),this.world.add(this.piles),this.piles.update(l?0:1),this.populate(e);let d=new Set(e.agents.map(u=>u.id));for(let[u,f]of this.machines)d.has(u)||(this.scene.remove(f.root),f.dispose(),this.machines.delete(u));for(let u of e.agents){let f=this.machines.get(u.id);f&&(f.agent.type!==u.type||f.agent.action_type!==u.action_type||f.agent.width!==u.width||f.agent.height!==u.height||f.agent.reach.some((g,x)=>g!==u.reach[x]))&&(this.scene.remove(f.root),f.dispose(),this.machines.delete(u.id),f=null),f||(f=Lp(u,e.grid.tile_size_m,{style:this.presentation}),f.setTags(this.visibility.tags),this.machines.set(u.id,f),this.scene.add(f.root)),l||(f.lastMove=null),this.poseMachine(f,u,e,1)}if(l){let u=VM.includes(c.kind)?i*1.3:i;this.motion={previous:r,frame:e,facts:c,start:performance.now(),duration:kn(u,100,900),events:this.planEvents(c,r,e),fired:new Set};for(let f of c.changed)this.updateCell(f.row,f.col,r.maps.action[f.row][f.col]);this.dirtyInstances(),this.update(performance.now())}this.selected&&this.highlight(this.selected.row,this.selected.col),(a||s)&&this.home({instant:!0})}populate(e){let{rows:t,cols:i,tile_size_m:s}=e.grid,r=[];this.boundaryEntries.clear(),this.gridEntries=new Map,this.displayHeights=e.maps.action.map(l=>[...l]);let a=this.palette.dug.map(l=>new Le(l)),o=new Le(this.palette.sand),c=new Le(this.palette.loose);for(let l=0;l<t;l++)for(let h=0;h<i;h++){let d=l*i+h,u=e.maps.action[l][h];this.updateCell(l,h,u);let f=(l*71+h*29+l*h%47)%31/31;jp.copy(u<0?a[Math.min(a.length-1,-u-1)]:u>0?c:o).multiplyScalar(.98+f*.04),this.terrain.setColorAt(d,jp),this.visibility.grid&&(this.gridEntries.set(d,r.length),r.push(...this.flatGridCell(l,h,u)))}this.dirtyInstances(),this.terrain.instanceColor.needsUpdate=!0,this.terrain.computeBoundingSphere(),this.gridLines.geometry.dispose(),this.gridLines.geometry=new mt,this.gridLines.geometry.setAttribute("position",new nt(r,3));for(let[l,h]of Object.entries(this.layers))h.visible=this.visibility[l]&&this.layerMaps[Vs[l].map]!=null;for(let[l,h]of Object.entries(this.boundaries)){let d=[],u=Vs[l].test,f=this.layerMaps[Vs[l].map],g=(x,p)=>f!=null&&x>=0&&x<t&&p>=0&&p<i&&!e.maps.padding[x][p]&&u(f[x][p]);for(let x=0;x<t;x++)for(let p=0;p<i;p++)if(g(x,p)){let m=x*i+p,M=[[x-1,p,[[0,0],[0,1],[0,2]]],[x+1,p,[[2,0],[2,1],[2,2]]],[x,p-1,[[0,0],[1,0],[2,0]]],[x,p+1,[[0,2],[1,2],[2,2]]]];for(let[b,v,T]of M)if(!g(b,v)){this.boundaryEntries.has(m)||this.boundaryEntries.set(m,[]);for(let w of[T[0],T[1],T[1],T[2]]){let C=x*2+w[0],_=p*2+w[1];this.boundaryEntries.get(m).push({name:l,y:d.length+1,row2:C,col2:_}),d.push((_/2-i/2)*s,this.boundaryHeight(x,p,C,_),(C/2-t/2)*s)}}}h.geometry.dispose(),h.geometry=new Bs,d.length&&h.geometry.setPositions(d),h.visible=this.visibility[l]&&d.length>0}}flatGridCell(e,t,i){let s=this.frame.grid.tile_size_m,r=this.point(e,t,i),a=r.y+s*.022,o=i>0&&!this.frame.maps.padding[e][t]?0:s/2,c=r.x,l=r.z;return[c-o,a,l-o,c+o,a,l-o,c+o,a,l-o,c+o,a,l+o,c+o,a,l+o,c-o,a,l+o,c-o,a,l+o,c-o,a,l-o]}setFloor(e){this.floor=e,this.environment?.setFloor(e)}boundaryHeight(e,t,i,s){let r=this.displayHeights[e][t];return(r>0?this.piles.nodeHeight(i,s):r*this.unitHeight)+this.frame.grid.tile_size_m*.06}updateCell(e,t,i){let s=this.frame,r=s.grid.tile_size_m,a=e*s.grid.cols+t,o=i>0&&!s.maps.padding[e][t],c=this.point(e,t,i);this.displayHeights[e][t]=i;let l=Math.max(r*.02,(o?0:c.y)-this.floor);zn.rotation.set(0,0,0),zn.position.set(c.x,this.floor+l/2,c.z),zn.scale.set(r,l,r),zn.updateMatrix(),this.terrain.setMatrixAt(a,zn.matrix);let h=0;for(let[d,u]of Object.entries(this.layers)){let f=Vs[d],g=this.layerMaps[f.map],x=!o&&g!=null&&f.test(g[e][t])&&!s.maps.padding[e][t];zn.position.set(c.x,c.y+r*(.008+h*.003),c.z),zn.scale.set(x?r:0,1,x?r:0),zn.updateMatrix(),u.setMatrixAt(a,zn.matrix),h++}this.gridEntries.has(a)&&this.gridLines.geometry.attributes.position.array.set(this.flatGridCell(e,t,i),this.gridEntries.get(a))}dirtyInstances(){this.terrain.instanceMatrix.needsUpdate=!0;for(let t of Object.values(this.layers))t.instanceMatrix.needsUpdate=!0;let e=this.frame.grid.cols;for(let[t,i]of this.boundaryEntries)for(let s of i){let r=this.boundaries[s.name].geometry.attributes.instanceStart?.data.array;r&&(r[s.y]=this.boundaryHeight(Math.floor(t/e),t%e,s.row2,s.col2))}for(let t of Object.values(this.boundaries)){let i=t.geometry.attributes.instanceStart?.data;i&&(i.needsUpdate=!0)}this.gridLines.geometry.attributes.position&&(this.gridLines.geometry.attributes.position.needsUpdate=!0)}poseMachine(e,t,i,s,r,a,o=""){let c=r||t,l=o==="turn"?HM(s):s,h=t.position.map((u,f)=>rh(c.position[f],u,s)),d=this.point(h[0],h[1]);d.y=rh(this.displayHeightAt(...c.position),this.displayHeightAt(...t.position),s),e.root.position.copy(d),e.root.rotation.y=id(c.base_yaw,t.base_yaw,l),e.setPose({...t,previous_loaded:c.loaded,cabin_yaw:id(c.cabin_yaw,t.cabin_yaw,l),wheel_angle:rh(c.wheel_angle,t.wheel_angle,s)},t.id===i.current_agent,s,o),e.drive?.(e.root.position,e.root.rotation.y)}planEvents(e,t,i){let s=[],r=i.agents.find(l=>l.id===i.actor_id),a=t.agents.find(l=>l.id===i.actor_id);if(!r||!a)return s;let o=r.position.some((l,h)=>l!==a.position[h]);s.push({at:0,once:"exhaust-start"}),o&&s.push({from:.05,to:.9,stream:"tracks",rate:16});let c=l=>e.changed.filter(h=>l==="dig"?h.delta<0:h.delta>0);if(e.kind==="dig"){let l=r.type===2?.38:.32;s.push({at:l,once:"bite",cells:c("dig")}),s.push({from:l,to:l+.22,stream:"scoop",cells:c("dig"),rate:70})}else if(e.kind==="dump"){let[l,h]=r.type===1?[.32,.72]:r.type===2?[.34,.62]:[.44,.74];s.push({from:l,to:h,stream:r.type===1?"bed":"pour",cells:c("dump"),rate:60}),s.push({at:(l+h)/2+.1,once:"landing",cells:c("dump")})}else e.kind==="transfer"&&s.push({from:.44,to:.72,stream:"transfer",rate:55});return s}centroid(e){let t=new A;if(!e?.length)return null;for(let i of e)t.add(this.surfacePoint(i.row,i.col));return t.multiplyScalar(1/e.length)}runEvents(e,t,i){let s=this.machines.get(e.frame.actor_id);if(!s)return;s.root.updateMatrixWorld(!0);let r=e.frame.grid.tile_size_m,a=this.effects;for(let[o,c]of e.events.entries())if(c.once){if(e.fired.has(o)||t<c.at)continue;if(e.fired.add(o),c.once==="exhaust-start")a.puff(s.exhaust(),{count:4,size:r*.28,rise:1.4,spread:r*.15,color:6185835,life:1.1});else if(c.once==="bite"){let l=this.centroid(c.cells)??s.tip();a.burst(l,{count:14,speed:2.4,size:r*.09}),a.puff(l,{count:7,size:r*.38,spread:r*.6,rise:.5})}else if(c.once==="landing"){let l=this.centroid(c.cells);l&&a.puff(l,{count:8,size:r*.42,spread:r*.7,rise:.4})}}else if(t>=c.from&&t<=c.to){c.carry=(c.carry??0)+c.rate*i;let l=Math.floor(c.carry);for(c.carry-=l;l-- >0;)this.emitStream(c,s,e,r)}}emitStream(e,t,i,s){let r=this.effects,a=o=>o?.length?this.surfacePoint(...Object.values(o[Math.floor(Math.random()*o.length)]).slice(0,2)):null;if(e.stream==="tracks"){let o=i.frame.agents.find(l=>l.id===t.agent.id),c=new A(-t.agent.height*s*.45,0,(Math.random()<.5?-1:1)*t.agent.width*s*.35).applyAxisAngle(new A(0,1,0),t.root.rotation.y).add(t.root.position);o&&Math.random()<.5&&r.puff(c,{count:1,size:s*.3,spread:s*.2,rise:.35,life:.8}),Math.random()<.25&&r.puff(t.exhaust(),{count:1,size:s*.2,rise:1.3,spread:s*.08,color:6975351,life:1})}else if(e.stream==="scoop"){let o=a(e.cells);o&&r.throwClods(o,t.tip(),{count:1,flight:.22,spread:0,size:s*.08,settle:!1,jitter:s*.3})}else if(e.stream==="pour"||e.stream==="bed"){let o=a(e.cells);if(!o)return;let c=e.stream==="bed"?t.bedLip():t.tip();r.throwClods(c,o,{count:1,flight:.34,spread:s*.45,size:s*.095,jitter:s*.12})}else if(e.stream==="transfer"){let o=this.machines.get(i.facts.recipient?.id);if(!o)return;o.root.updateMatrixWorld(!0),r.throwClods(t.tip(),o.tip(),{count:1,flight:.3,spread:s*.2,size:s*.09,settle:!1,jitter:s*.1})}}update(e){let t=Math.min(.1,Math.max(0,(e-(this.clock?.last??e))/1e3));if(this.clock&&(this.clock.last=e),on.uTime.value=e/1e3,!this.frame)return;this.measure(t),this.environment?.update(e/1e3);let i=new Map;if(this.motion){let{previous:r,frame:a,facts:o,start:c,duration:l}=this.motion,h=kn((e-c)/l,0,1),d=zM(h);o.changed.length&&this.piles.update(d);for(let u of o.changed)this.updateCell(u.row,u.col,rh(r.maps.action[u.row][u.col],a.maps.action[u.row][u.col],d));for(let u of a.agents){let f=r.agents.find(x=>x.id===u.id),g=u.id===a.actor_id?o.kind:u.id===o.recipient?.id&&o.kind==="transfer"?"receive":"";if(this.poseMachine(this.machines.get(u.id),u,a,d,f,r,g==="turn"||g==="move"?f&&u.position.some((x,p)=>x!==f.position[p])?"move":"turn":g),f&&u.position.some((x,p)=>x!==f.position[p])){let x=new te(Math.cos(u.base_yaw),Math.sin(u.base_yaw)),p=new te(u.position[1]-f.position[1],u.position[0]-f.position[0]);i.set(u.id,{move:h,direction:Math.sign(x.x*p.x-x.y*p.y)||1})}}o.changed.length&&(this.dirtyInstances(),this.selected&&this.highlight(this.selected.row,this.selected.col)),this.reducedMotion||this.runEvents(this.motion,h,t),h>=1&&(this.clearMotion(),this.floor!==this.finalFloor&&(this.setFloor(this.finalFloor),this.populate(this.frame)))}let s=e/1e3;for(let r of this.machines.values())r.tick?.(s,{...i.get(r.agent.id)||{},reducedMotion:this.reducedMotion});if(this.clock.idle-=t,!this.reducedMotion&&this.clock.idle<=0){this.clock.idle=1.3+Math.random()*.8;let r=this.machines.get(this.frame.current_agent),a=this.frame.grid.tile_size_m;r&&!this.frame.done&&(r.root.updateMatrixWorld(!0),this.effects.puff(r.exhaust(),{count:1,size:a*.18,rise:1.1,spread:a*.05,color:7764867,life:1.2}))}if(this.effects.update(t),this.tween){let{from:r,to:a,start:o,duration:c}=this.tween,l=kM(kn((e-o)/c,0,1));this.camera.position.lerpVectors(r.position,a.position,l),this.controls.target.lerpVectors(r.target,a.target,l),l>=1&&(this.tween=null)}if(this.follow){let r=this.machines.get(this.frame.current_agent);if(r){let a=r.root.position.clone();a.y+=this.frame.grid.tile_size_m;let o=a.sub(this.controls.target).multiplyScalar(.055);this.camera.position.add(o),this.controls.target.add(o)}}}clearMotion(){if(this.motion)for(let e of this.motion.frame.agents){let t=this.machines.get(e.id);t&&t.tick?.(performance.now()/1e3,{reducedMotion:this.reducedMotion})}this.motion=null}setLayer(e,t){if(this.visibility[e]=t,e==="eligibility"){for(let i of BM)this.setLayer(i,t);return}if(e==="tags"){for(let i of this.machines.values())i.setTags(t);return}this.frame&&(this.piles?.setLayer(e,t),e==="grid"?(this.gridLines.visible=t,this.populate(this.frame)):this.layers[e]&&(this.layers[e].visible=t&&this.layerMaps[Vs[e].map]!=null),this.boundaries[e]&&(this.boundaries[e].visible=t&&!!this.boundaries[e].geometry.attributes.instanceStart))}setHeight(e){this.heightScale=e,this.frame&&this.setFrame(this.frame)}flyTo(e,t,{instant:i=!1}={}){if(i||this.reducedMotion){this.tween=null,this.camera.position.copy(e),this.controls.target.copy(t),this.controls.update();return}this.tween={from:{position:this.camera.position.clone(),target:this.controls.target.clone()},to:{position:e,target:t},start:performance.now(),duration:750}}home({instant:e=!1}={}){if(!this.frame)return;this.follow=!1;let t=this.camera.aspect,i=this.presentation==="paper"?1.45:1.75,s=this.span*(t<1?i/t:i);this.flyTo(new A(s*.72,s*.66,s*.84),new A(0,-this.span*.04,0),{instant:e}),this.onCameraChange?.({view:"home",follow:!1})}top({instant:e=!1}={}){if(!this.frame)return;this.follow=!1,this.camera.up.set(0,1,0);let t=this.span*.5/Math.tan(Vt.degToRad(this.camera.fov)/2)/Math.min(this.camera.aspect,1)*1.3;this.flyTo(new A(0,t,this.span*.001),new A(0,0,0),{instant:e}),this.onCameraChange?.({view:"top",follow:!1})}setFollow(e){if(this.follow=e,this.tween=null,e&&this.frame){let t=this.machines.get(this.frame.current_agent);if(t){let i=this.frame.agents.find(d=>d.id===this.frame.current_agent),s=this.frame.grid.tile_size_m,r=t.root.position.clone();r.y+=s;let a=Vt.degToRad(this.camera.fov),o=2*Math.atan(Math.tan(a/2)*this.camera.aspect),c=Math.max(i.reach[1],Math.hypot(i.width,i.height)*.65)*s,l=Math.max(this.span*.5,c*1.15/Math.sin(Math.min(a,o)/2)),h=this.camera.position.clone().sub(this.controls.target).normalize().multiplyScalar(l);this.flyTo(r.clone().add(h),r)}}this.onCameraChange?.({follow:e})}pick(e){if(!this.terrain)return;let t=this.renderer.domElement.getBoundingClientRect();this.pointer.set((e.clientX-t.left)/t.width*2-1,-(e.clientY-t.top)/t.height*2+1),this.camera.updateMatrixWorld(),this.world.updateMatrixWorld(!0),this.raycaster.setFromCamera(this.pointer,this.camera);let i=[this.terrain,this.obstacleProps];this.piles.surface.visible&&i.push(this.piles.surface);let s=this.raycaster.intersectObjects(i.filter(Boolean),!0)[0],r;if(s?.object===this.piles.surface)r=this.piles.cellForHit(s);else if(s?.object===this.terrain&&s.instanceId!==void 0)r={row:Math.floor(s.instanceId/this.frame.grid.cols),col:s.instanceId%this.frame.grid.cols};else if(s){let{rows:a,cols:o,tile_size_m:c}=this.frame.grid;r={row:kn(Math.floor(s.point.z/c+a/2),0,a-1),col:kn(Math.floor(s.point.x/c+o/2),0,o-1)}}r&&(this.highlight(r.row,r.col),this.onPick?.(r))}highlight(e,t){if(e>=this.frame.grid.rows||t>=this.frame.grid.cols){this.selected=null,this.selection.visible=!1;return}this.selected={row:e,col:t},this.selection.position.copy(this.surfacePoint(e,t)),this.selection.position.y+=this.frame.grid.tile_size_m*.03,this.selection.visible=!0}capture({scale:e=2}={}){let t=this.element.clientWidth||1,i=this.element.clientHeight||1,s=Math.min(Math.max(e,this.pixelRatio),this.renderer.capabilities.maxTextureSize/Math.max(t,i));this.renderer.setPixelRatio(s),this.renderer.setSize(t,i,!1),this.post?.setSamples(0),this.post?.setSize(t,i,s);let r=this.renderer.getDrawingBufferSize(new te);for(let a of this.lineMaterials)a.resolution.copy(r);try{return this.render(),{url:this.renderer.domElement.toDataURL("image/png"),width:r.x,height:r.y}}finally{this.renderer.setPixelRatio(this.pixelRatio),this.post?.setSamples(4),this.resize()}}};function sm({replay:n,index:e=0,mode:t,imported:i=!1,busy:s=!1,playing:r=!1,session:a={}}){let o=n?.frames[e],c=t==="manual"&&!i,l=!!o&&e===n.frames.length-1;return{action:c&&l&&!s&&!r&&(!o.done||!!a.exploring),reset:c&&!s,undo:c&&l&&!s&&!r&&!!a.can_undo,continueVisible:c&&l&&!!a.cases?.length&&!!o.done&&!o.task_done&&!a.exploring,continueEnabled:!s&&!r}}function rm(n){if(!n?.loaded)return"Empty bucket. Cabin turns keep the base in place.";if(n.accepted_unload_now)return"Press Space to unload into the accepted dump area.";let e=n.accepted_unload_by_cabin_offset;if(Array.isArray(e)&&e.length){let t=e.flatMap((i,s)=>!i||s===0?[]:[{key:s<=e.length/2?"Q":"E",turns:Math.min(s,e.length-s)}]);if(t.sort((i,s)=>i.turns-s.turns),t.length)return`${t[0].key} \xD7 ${t[0].turns}, then Space to unload. Base stays in place.`}return n.accepted_unload_any?"Swing the cabin to an accepted dump direction, then press Space.":n.dump_status==="off_zone_only"?"Only off-target unloading is available at this base. It will not count as accepted disposal.":"No unloading direction at this base. Undo is available; the loaded base cannot move."}/*!
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
 */var ue=n=>document.getElementById(n),ln=n=>new Intl.NumberFormat(void 0,{maximumFractionDigits:2}).format(n),ut=(n,e)=>{ue(n).textContent=e},wt,ot,At=0,qi="replay",dm=null,bi=!1,xt=!1,En=!1,fo=null,od,oh=0,Jt={},fm=ue("terra-replay"),pm=!!fm?.textContent.trim();function Vn(n){ue("loading").hidden=!0,ut("error-message",n?.message||String(n)),ue("error").hidden=!1}function cn(n,e=3100){clearTimeout(od),ut("event",n),ue("event").classList.add("visible"),od=setTimeout(()=>ue("event").classList.remove("visible"),e)}function xs(){return ot?.frames[At]}function mm(){return sm({replay:ot,index:At,mode:qi,imported:bi,busy:xt,playing:En,session:Jt})}function GM(){return mm().action}function _i(){let n=mm(),e=n.action,t=xs();document.querySelectorAll("[data-action]").forEach(s=>{s.disabled=!e}),ue("reset").disabled=!n.reset,ue("undo").disabled=!n.undo;for(let s of["case-select","precision-select","start-select"])ue(s).disabled=xt||qi!=="manual"||bi;ue("continue").hidden=!n.continueVisible,ue("continue").disabled=!n.continueEnabled,ue("export").disabled=!ot||xt,ue("screenshot").disabled=!wt||!ot,ue("open-file").disabled=xt,ue("previous").disabled=!ot||At<=0||xt,ue("next").disabled=!ot||At>=ot.frames.length-1||xt,ue("play").disabled=!ot||ot.frames.length<2||xt,ue("seek").disabled=!ot||ot.frames.length<2||xt,ue("play").textContent=En?"\u2161":"\u25B6",ue("play").setAttribute("aria-label",En?"Pause replay":"Play replay"),ue("resume-live").hidden=!dm||pm||!bi&&!(qi==="manual"&&ot&&At<ot.frames.length-1),ue("resume-live").disabled=xt;let i=ot&&At<ot.frames.length-1;ut("manual-status",xt?"Working\u2026":bi||qi!=="manual"?"Replay":i?"History":Jt.exploring?"Exploring":t?.done?"Ended":En?"Playback":"Live"),ut("session-mode",bi?"Imported replay":qi==="manual"?i?"Manual \xB7 history":"Manual session":"Replay session")}function ud(){if(!fo||!xs())return;let{row:n,col:e}=fo,t=xs(),{maps:i}=t;if(n>=t.grid.rows||e>=t.grid.cols){fo=null;return}let s=ue("cell-inspector");s.replaceChildren();let r=document.createElement("span");r.className="eyebrow",r.textContent="CELL INSPECTOR",s.append(r);let a=document.createElement("div");a.className="cell-heading",a.textContent=`ROW ${n}  \xB7  COL ${e}`,s.append(a);let o=document.createElement("div");o.className="cell-data",s.append(o);let c=(u,f,g)=>i[u]==null?"Unavailable":i[u][n][e]?f:g,l=i.target[n][e],h=so(t),d=[["Raw soil height",`${i.action[n][e]} units`],["Target",l<0?`Dig ${-l}`:l>0?`Dump ${l}`:"Neutral"],["Obstacle",i.padding[n][e]?"Yes":"No"],["Static dumping",c("dumpability_static","Allowed","Prohibited")],["Dumpable now",c("dumpability","Yes","No")],["Workspace preview",c(i.work_cone!=null?"work_cone":"interaction","Inside","Outside")],...i.footprint!=null?[["Machine footprint",c("footprint","Inside","Outside")]]:[],...i.precision_required_band!=null?[["Precise edge",c("precision_required_band","Required","Not required")]]:[],...h.available?[["Fresh excavation",Fp[h.cells[n][e]]]]:[],["Traversability feature",i.traversability==null?"Unavailable":{"-1":"Occupied (\u22121)",0:"Clear (0)",1:"Blocked (1)"}[Number(i.traversability[n][e])]]];for(let[u,f]of d){let g=document.createElement("span"),x=document.createElement("strong");g.textContent=u,x.textContent=f,o.append(g,x)}}function WM(){let n=xs(),e=n.agents.find(d=>d.id===n.current_agent),t=Op(n);ut("title",ot.metadata.title),ut("source",ot.metadata.source),ue("source").title=ot.metadata.source,ut("grid-spec",`${n.grid.rows} \xD7 ${n.grid.cols} \xB7 ${ln(n.grid.tile_size_m)} m / cell`),ut("agent-count",`${n.agents.length} machine${n.agents.length===1?"":"s"}`),ut("agent-id",String(e.id+1).padStart(2,"0")),ut("machine-name",ju[e.type]),ut("embodiment",e.action_type===1?"Wheeled":"Tracked"),ue("load").replaceChildren(document.createTextNode(ln(e.loaded)));let i=document.createElement("small");i.textContent=" units",ue("load").append(i),ut("reward",Bp(n.reward)),ut("outcome",n.task_done?"Task complete":n.done?"Episode ended \xB7 task incomplete":`Ready \xB7 machine ${e.id+1} acts next`),ue("outcome").classList.toggle("done",n.done);let s=n.diagnostics,r=so(n),a=ot.metadata.selected_case||(bi?null:Jt.selected_case);if(ue("rule-badge").hidden=!a,a&&ut("rule-badge",a.precision?"Precise edges":"Bulk excavation"),ue("native-status").hidden=!s,ue("dig-legend").hidden=!r.available,ue("scene-legend").hidden=r.available,s){let d=s.step_budget??450,u=!!s.exploring||!bi&&At===ot.frames.length-1&&!!Jt.exploring;ut("step-budget",`${n.step} / ${d}`),ut("budget-note",u?"Exploration \xB7 outside the episode budget":n.task_done?"Completed within the episode":n.done?"Budget reached \xB7 episode frozen":`${Math.max(0,s.remaining_steps??d-n.step)} actions left`),ue("native-status").classList.toggle("exploring",u),ut("native-action-message",s.message||"Ready for a native simulator action."),ut("do-status",r.loaded?s.accepted_unload_now?"Unload allowed now":s.accepted_unload_any?"Turn cabin to unload":"No accepted unload here":r.counts.current?`Dig ${r.counts.current} fresh cells now`:s.do_kind==="relift"?"Work picks up loose soil":"No fresh dig at this heading"),ut("dump-status",rm(s))}r.available&&(ut("dig-current-count",r.counts.current),ut("dig-swing-count",r.counts.swing),ut("dig-blocked-count",r.counts.blocked),ue("dig-legend").classList.toggle("loaded",r.loaded),ue("eligibility-keys").hidden=r.loaded,ue("unload-first").hidden=!r.loaded,ut("eligibility-note",r.loaded?"Dig colors are paused while carrying soil. The precision outline stays visible.":`${r.counts.remaining} target cells remain \xB7 colors apply to this base position`),ue("precision-key").hidden=r.counts.precision===0);let o=ue("agent-list");o.replaceChildren(),o.hidden=n.agents.length<=1;for(let d of n.agents){let u=document.createElement("span");u.className=`agent-tag${d.id===n.current_agent?" active":""}`,u.textContent=`${String(d.id+1).padStart(2,"0")} ${ju[d.type]} \xB7 ${d.loaded}`,o.append(u)}let c=s?.metrics;ut("cut-label",c?"Excavated target":"Excavated"),ut("fill-label",c?"Accepted disposal":"Placed soil"),ut("cut-units",c?`${ln(c.dug)} / ${ln(c.required)} units`:`${ln(t.cut)} units`),ut("fill-units",c?`${ln(c.disposed)} / ${ln(c.required)} units`:`${ln(t.fill)} units`),ut("scene-caption",`${ln(n.grid.cols*n.grid.tile_size_m)} \xD7 ${ln(n.grid.rows*n.grid.tile_size_m)} m worksite \xB7 illustrative soil mounds`),ut("step",n.step),ut("frame-count",`${At+1} / ${ot.frames.length}`),ut("action-label",td(n,ot.frames[At-1])),ue("seek").max=String(ot.frames.length-1),ue("seek").value=String(At),ue("seek").setAttribute("aria-valuetext",`Snapshot ${At+1} of ${ot.frames.length}, step ${n.step}`);let l=ue("left-action"),h=ue("right-action");l.dataset.action=e.action_type===1?"2":"3",h.dataset.action=e.action_type===1?"3":"2",l.querySelector(".turn-label").textContent=e.action_type===1?"Steer left":"Turn left",h.querySelector(".turn-label").textContent=e.action_type===1?"Steer right":"Turn right",l.title=`${e.action_type===1?"Steer left":"Turn anticlockwise"} \xB7 Left or A`,h.title=`${e.action_type===1?"Steer right":"Turn clockwise"} \xB7 Right or D`,ut("work-label",e.type===2?e.shovel_lifted?"Lower shovel / dump":"Lift shovel":e.loaded>0?"Dump / transfer soil":e.type===1?"Dump (empty)":"Dig soil");for(let[d,u]of[["interaction",n.maps.work_cone!=null?"work_cone":"interaction"],["footprint","footprint"],["restricted","dumpability_static"],["dumpability","dumpability"],["precision","precision_required_band"],["eligibility","fresh_dig_current"]]){let f=document.querySelector(`[data-layer="${d}"]`);f.disabled=n.maps[u]==null,f.closest("label").title=n.maps[u]==null?"This diagnostic layer is unavailable in the recording.":""}hd(),ud(),_i()}function Hn(n,{animate:e=!1,reset:t=!1,announce:i=!1}={}){if(!ot)return;let s=At;At=Math.max(0,Math.min(ot.frames.length-1,n));let r=xs();if(wt.setFrame(r,{animate:e&&At===s+1,reset:t,duration:Math.min(650,800/Number(ue("speed").value))}),WM(),i&&At>0){let a=ot.frames[At-1];r.step>a.step?cn(r.diagnostics?.message||`${td(r,a)} \xB7 ${Vc(a,r).message}`):cn("Episode boundary \xB7 initial snapshot")}else t&&(clearTimeout(od),ue("event").classList.remove("visible"))}function Sn(n){En=n,oh=performance.now(),_i()}function am(){!ot||ot.frames.length<2||xt||(!En&&At===ot.frames.length-1&&Hn(0),Sn(!En))}function gm(n){En&&!document.hidden&&n-oh>=900/Number(ue("speed").value)&&(oh=n,At<ot.frames.length-1&&Hn(At+1,{animate:!0,announce:!0}),At>=ot.frames.length-1&&Sn(!1)),requestAnimationFrame(gm)}async function kr(n,e){let t=await fetch(n,e===void 0?{cache:"no-store"}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(e)}),i=await t.json().catch(()=>{throw new Error(`The server returned an unreadable response (${t.status}).`)});if(!t.ok)throw new Error(i.error||`Request failed (${t.status}).`);return i}function zr(n,{local:e=!1}={}){if(Qu(n.replay),!["manual","replay"].includes(n.mode))throw new Error("Unknown viewer session mode.");ot=n.replay,qi=n.mode,bi=e,At=0,fo=null,En=!1,e||(dm=n.mode,Jt=n),XM(),ue("cell-inspector").replaceChildren();let t=document.createElement("span");t.className="eyebrow",t.textContent="CELL INSPECTOR";let i=document.createElement("span");i.className="cell-hint",i.textContent="Click the terrain to inspect a cell",ue("cell-inspector").append(t,i),Hn(qi==="manual"&&!e?ot.frames.length-1:0,{reset:!0}),!e&&Jt.cases?.length&&wt.top({instant:!0}),ue("loading").hidden=!0,ue("error").hidden=!0}async function om(n){if(GM()){xt=!0,_i();try{let e=await kr("/api/action",{action:n}),t=e.frame;ed(t),Jt={...Jt,can_undo:e.can_undo??Jt.can_undo,exploring:e.exploring??Jt.exploring},ot.frames.push(t),Hn(ot.frames.length-1,{animate:!0,announce:!0})}catch(e){Vn(e)}finally{xt=!1,_i()}}}async function lm(){if(xt||qi!=="manual"||bi)return;xt=!0,Sn(!1),_i();let n=Jt.cases?.length?{case_id:ue("case-select").value,precision:ue("precision-select").value==="precision",start:Number(ue("start-select").value)}:{};try{zr(await kr("/api/reset",n)),cn("Episode reset \xB7 native initial state restored")}catch(e){Vn(e)}finally{xt=!1,_i()}}function XM(){let n=Jt.cases||[];if(ue("case-controls").hidden=bi||qi!=="manual"||!n.length,bi||!n.length)return;let e=ue("case-select");e.replaceChildren();for(let t of n){let i=document.createElement("option");i.value=t.id,i.textContent=t.title||`Map ${t.source_slot}`,e.append(i)}e.value=Jt.selected_case?.case_id??n[0].id,_m(!0)}function _m(n=!1){let e=Jt.cases?.find(a=>String(a.id)===ue("case-select").value);if(!e)return;let t=n?Jt.selected_case?.precision?"precision":"bulk":ue("precision-select").value,i=ue("precision-select");i.replaceChildren();for(let a of e.modes||["bulk","precision"]){let o=document.createElement("option");o.value=a,o.textContent=a==="precision"?"Precise edges":"Bulk excavation",i.append(o)}[...i.options].some(a=>a.value===t)&&(i.value=t);let s=n?Jt.selected_case?.start??0:Number(ue("start-select").value),r=ue("start-select");r.replaceChildren();for(let a of e.starts||[0]){let o=document.createElement("option");o.value=a,o.textContent=`Start ${Number(a)+1}`,r.append(o)}[...r.options].some(a=>Number(a.value)===s)&&(r.value=s),ue("start-label").hidden=r.options.length<=1,ld()}function ld(){let n=Jt.selected_case,e=n&&(String(n.case_id)!==ue("case-select").value||!!n.precision!=(ue("precision-select").value==="precision")||Number(n.start??0)!==Number(ue("start-select").value));ut("case-note",e?"Selection changed. Reset to apply.":"Reset restores the saved initial state."),ue("case-note").classList.toggle("pending",!!e)}async function cm(n,e){if(!(xt||bi||qi!=="manual")){xt=!0,Sn(!1),_i();try{zr(await kr(n,{})),cn(e)}catch(t){Vn(t)}finally{xt=!1,_i()}}}function xm(n,e,t=!1){let i=document.createElement("a");i.href=n,i.download=e,document.body.append(i),i.click(),i.remove(),t&&setTimeout(()=>URL.revokeObjectURL(n),1e3)}async function qM(){if(!(!ot||xt)){xt=!0,_i();try{let n=ot;if(qi==="manual"&&!bi&&Jt.cases?.length){let t=await kr("/api/export",{});t.replay&&(n=Qu(t.replay))}let e=new Blob([JSON.stringify(n)],{type:"application/json"});xm(URL.createObjectURL(e),"terra-replay.json",!0),cn(`Exported ${n.frames.length} snapshots with recorded overlays`)}catch(n){Vn(n)}finally{xt=!1,_i()}}}function YM(){document.querySelectorAll("[data-action]").forEach(n=>n.addEventListener("click",e=>{om(Number(n.dataset.action)),e.detail>0&&ue("viewport").focus({preventScroll:!0})})),ue("reset").addEventListener("click",n=>{lm(),n.detail>0&&ue("viewport").focus({preventScroll:!0})}),ue("undo").addEventListener("click",()=>cm("/api/undo","Undo \xB7 previous native state restored")),ue("continue").addEventListener("click",()=>cm("/api/continue","Exploration enabled \xB7 actions are outside the 450-step episode")),ue("case-select").addEventListener("change",()=>_m()),ue("precision-select").addEventListener("change",ld),ue("start-select").addEventListener("change",ld),ue("previous").addEventListener("click",()=>{Sn(!1),Hn(At-1)}),ue("next").addEventListener("click",()=>{Sn(!1),Hn(At+1,{animate:!0,announce:!0})}),ue("play").addEventListener("click",am),ue("seek").addEventListener("input",()=>{Sn(!1),Hn(Number(ue("seek").value))}),ue("speed").addEventListener("change",()=>{oh=performance.now()}),ue("camera-home").addEventListener("click",()=>wt?.home()),ue("brand-home").addEventListener("click",n=>{n.preventDefault(),wt?.home()}),ue("camera-top").addEventListener("click",()=>wt?.top()),ue("camera-follow").addEventListener("click",()=>wt?.setFollow(!wt.follow)),ue("quality").addEventListener("click",um),ue("presentation").addEventListener("click",hm),ue("height-scale").addEventListener("input",()=>{let n=Number(ue("height-scale").value);ut("height-value",`${ln(n)}\xD7`),wt?.setHeight(n)}),document.querySelectorAll("[data-layer]").forEach(n=>n.addEventListener("change",()=>{wt?.setLayer(n.dataset.layer,n.checked),hd()})),ue("layers-toggle").addEventListener("click",()=>{let n=[...document.querySelectorAll("[data-layer]")].filter(t=>!t.disabled),e=!n.some(t=>t.checked);for(let t of n)t.checked=e,wt?.setLayer(t.dataset.layer,e);hd()}),ue("export").addEventListener("click",qM),ue("screenshot").addEventListener("click",()=>{try{let n=wt.capture();xm(n.url,`terra-step-${xs().step}.png`),cn(`Scene captured \xB7 ${n.width} \xD7 ${n.height} PNG`)}catch(n){Vn(n)}}),ue("open-file").addEventListener("click",()=>ue("replay-file").click()),ue("replay-file").addEventListener("change",async n=>{let e=n.target.files[0];if(e){if(xt){n.target.value="",cn("Wait for the current action before opening a replay.");return}xt=!0,Sn(!1),_i();try{if(e.size>256*1024*1024)throw new Error("Please use a JSON recording smaller than 256 MB. Large recordings can be opened through Python with --replay.");let t=JSON.parse(await e.text());zr({mode:"replay",replay:t},{local:!0}),cn(`Opened ${e.name}`)}catch(t){Vn(t)}finally{n.target.value="",xt=!1,_i()}}}),ue("resume-live").addEventListener("click",async()=>{if(!xt){xt=!0,Sn(!1),_i();try{zr(await kr("/api/session"))}catch(n){Vn(n)}finally{xt=!1,_i()}}}),ue("dismiss-error").addEventListener("click",()=>{ue("error").hidden=!0}),document.addEventListener("keydown",n=>{if(n.ctrlKey||n.metaKey||n.altKey||n.repeat||["INPUT","SELECT","TEXTAREA","BUTTON"].includes(n.target.tagName)||n.target.isContentEditable||!ue("error").hidden)return;let e=n.key.toLowerCase();if(e==="g"){n.preventDefault(),um();return}if(e==="p"){n.preventDefault(),hm();return}if(e==="h"){n.preventDefault(),wt?.home();return}if(e==="t"){n.preventDefault(),wt?.top();return}if(e==="f"){n.preventDefault(),wt?.setFollow(!wt.follow);return}if(!ot||xt)return;if(qi!=="manual"||bi||At<ot.frames.length-1||En){e===" "&&(n.preventDefault(),am()),(e==="arrowleft"||e==="arrowright")&&(n.preventDefault(),Sn(!1),Hn(At+(e==="arrowright"?1:-1)));return}if(e==="r"){n.preventDefault(),lm();return}let t=xs().agents.find(a=>a.id===xs().current_agent),i=t.action_type===1?2:3,s=t.action_type===1?3:2,r={arrowup:0,w:0,arrowdown:1,s:1,arrowleft:i,a:i,arrowright:s,d:s,q:5,e:4," ":6,n:7};e in r&&(n.preventDefault(),om(r[e]))})}function cd(n){ue("quality").setAttribute("aria-pressed",String(n==="high"))}function vm(n){ue("presentation").querySelector("span").textContent=n==="paper"?"Paper":"Diorama"}function hm(){if(!wt)return;let n=wt.setPresentation(wt.presentation==="paper"?"diorama":"paper");vm(n),ud(),cn(n==="paper"?"Paper style \xB7 plain figure look":"Diorama style \xB7 stylized island")}function um(){if(!wt)return;let n=wt.setQuality(wt.quality==="high"?"fast":"high");cd(n),cn(n==="high"?"Rich lighting on \xB7 ambient occlusion and outlines":"Fast graphics \xB7 plain lighting")}function hd(){ut("layers-toggle",[...document.querySelectorAll("[data-layer]")].some(n=>n.checked&&!n.disabled)?"Hide all":"Show all")}async function $M(){YM(),_i();try{wt=new ah(ue("viewport"),{onPick:n=>{fo=n,ud()},onCameraChange:({view:n,follow:e})=>{n&&(ue("camera-home").classList.toggle("selected",n==="home"),ue("camera-top").classList.toggle("selected",n==="top")),e!==void 0&&ue("camera-follow").setAttribute("aria-pressed",String(e))},onError:Vn,onQualityChange:n=>{cd(n),cn("Switched to fast graphics for smoother motion \xB7 press G to restore")}}),cd(wt.quality),vm(wt.presentation),window.terraViewer={scene:wt,show:(n,e)=>Hn(n,e)},pm?zr({mode:"replay",replay:JSON.parse(fm.textContent)},{local:!0}):zr(await kr("/api/session")),requestAnimationFrame(gm)}catch(n){Vn(n),ut("session-mode","Unavailable"),ut("title","Open a Terra worksite"),ut("source","Check the error message to continue.")}}$M();})();
