/*! For license information please see main.35445d24.chunk.js.LICENSE.txt */
(this.webpackJsonpstreamlit_wallet_connect=this.webpackJsonpstreamlit_wallet_connect||[]).push([[0],{215:function(e,t,r){e.exports=r(332)},224:function(e,t){},267:function(e,t){},270:function(e,t){},299:function(e,t){},303:function(e,t){},331:function(e,t){function r(e){var t=new Error("Cannot find module '"+e+"'");throw t.code="MODULE_NOT_FOUND",t}r.keys=function(){return[]},r.resolve=r,e.exports=r,r.id=331},332:function(e,t,r){"use strict";r.r(t);var n=r(42),o=r.n(n),a=r(191),c=r.n(a),s=r(0),i=r(1),u=r(2),l=r(3),p=r(10),d=r(68),h=r(71),f=r(55),g=r(127),m=r(52),y=r(116),v=r(31),w=r(17),b=r(4),x=r(121),E=r(124),k=r.n(E),C=r(202),S=r.n(C),A=r(144),O=r(203);function I(){I=function(){return e};var e={},t=Object.prototype,r=t.hasOwnProperty,n="function"==typeof Symbol?Symbol:{},o=n.iterator||"@@iterator",a=n.asyncIterator||"@@asyncIterator",c=n.toStringTag||"@@toStringTag";function s(e,t,r){return Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}),e[t]}try{s({},"")}catch(A){s=function(e,t,r){return e[t]=r}}function i(e,t,r,n){var o=t&&t.prototype instanceof p?t:p,a=Object.create(o.prototype),c=new k(n||[]);return a._invoke=function(e,t,r){var n="suspendedStart";return function(o,a){if("executing"===n)throw new Error("Generator is already running");if("completed"===n){if("throw"===o)throw a;return S()}for(r.method=o,r.arg=a;;){var c=r.delegate;if(c){var s=b(c,r);if(s){if(s===l)continue;return s}}if("next"===r.method)r.sent=r._sent=r.arg;else if("throw"===r.method){if("suspendedStart"===n)throw n="completed",r.arg;r.dispatchException(r.arg)}else"return"===r.method&&r.abrupt("return",r.arg);n="executing";var i=u(e,t,r);if("normal"===i.type){if(n=r.done?"completed":"suspendedYield",i.arg===l)continue;return{value:i.arg,done:r.done}}"throw"===i.type&&(n="completed",r.method="throw",r.arg=i.arg)}}}(e,r,c),a}function u(e,t,r){try{return{type:"normal",arg:e.call(t,r)}}catch(A){return{type:"throw",arg:A}}}e.wrap=i;var l={};function p(){}function d(){}function h(){}var f={};s(f,o,(function(){return this}));var g=Object.getPrototypeOf,m=g&&g(g(C([])));m&&m!==t&&r.call(m,o)&&(f=m);var y=h.prototype=p.prototype=Object.create(f);function v(e){["next","throw","return"].forEach((function(t){s(e,t,(function(e){return this._invoke(t,e)}))}))}function w(e,t){var n;this._invoke=function(o,a){function c(){return new t((function(n,c){!function n(o,a,c,s){var i=u(e[o],e,a);if("throw"!==i.type){var l=i.arg,p=l.value;return p&&"object"==typeof p&&r.call(p,"__await")?t.resolve(p.__await).then((function(e){n("next",e,c,s)}),(function(e){n("throw",e,c,s)})):t.resolve(p).then((function(e){l.value=e,c(l)}),(function(e){return n("throw",e,c,s)}))}s(i.arg)}(o,a,n,c)}))}return n=n?n.then(c,c):c()}}function b(e,t){var r=e.iterator[t.method];if(void 0===r){if(t.delegate=null,"throw"===t.method){if(e.iterator.return&&(t.method="return",t.arg=void 0,b(e,t),"throw"===t.method))return l;t.method="throw",t.arg=new TypeError("The iterator does not provide a 'throw' method")}return l}var n=u(r,e.iterator,t.arg);if("throw"===n.type)return t.method="throw",t.arg=n.arg,t.delegate=null,l;var o=n.arg;return o?o.done?(t[e.resultName]=o.value,t.next=e.nextLoc,"return"!==t.method&&(t.method="next",t.arg=void 0),t.delegate=null,l):o:(t.method="throw",t.arg=new TypeError("iterator result is not an object"),t.delegate=null,l)}function x(e){var t={tryLoc:e[0]};1 in e&&(t.catchLoc=e[1]),2 in e&&(t.finallyLoc=e[2],t.afterLoc=e[3]),this.tryEntries.push(t)}function E(e){var t=e.completion||{};t.type="normal",delete t.arg,e.completion=t}function k(e){this.tryEntries=[{tryLoc:"root"}],e.forEach(x,this),this.reset(!0)}function C(e){if(e){var t=e[o];if(t)return t.call(e);if("function"==typeof e.next)return e;if(!isNaN(e.length)){var n=-1,a=function t(){for(;++n<e.length;)if(r.call(e,n))return t.value=e[n],t.done=!1,t;return t.value=void 0,t.done=!0,t};return a.next=a}}return{next:S}}function S(){return{value:void 0,done:!0}}return d.prototype=h,s(y,"constructor",h),s(h,"constructor",d),d.displayName=s(h,c,"GeneratorFunction"),e.isGeneratorFunction=function(e){var t="function"==typeof e&&e.constructor;return!!t&&(t===d||"GeneratorFunction"===(t.displayName||t.name))},e.mark=function(e){return Object.setPrototypeOf?Object.setPrototypeOf(e,h):(e.__proto__=h,s(e,c,"GeneratorFunction")),e.prototype=Object.create(y),e},e.awrap=function(e){return{__await:e}},v(w.prototype),s(w.prototype,a,(function(){return this})),e.AsyncIterator=w,e.async=function(t,r,n,o,a){void 0===a&&(a=Promise);var c=new w(i(t,r,n,o),a);return e.isGeneratorFunction(r)?c:c.next().then((function(e){return e.done?e.value:c.next()}))},v(y),s(y,c,"Generator"),s(y,o,(function(){return this})),s(y,"toString",(function(){return"[object Generator]"})),e.keys=function(e){var t=[];for(var r in e)t.push(r);return t.reverse(),function r(){for(;t.length;){var n=t.pop();if(n in e)return r.value=n,r.done=!1,r}return r.done=!0,r}},e.values=C,k.prototype={constructor:k,reset:function(e){if(this.prev=0,this.next=0,this.sent=this._sent=void 0,this.done=!1,this.delegate=null,this.method="next",this.arg=void 0,this.tryEntries.forEach(E),!e)for(var t in this)"t"===t.charAt(0)&&r.call(this,t)&&!isNaN(+t.slice(1))&&(this[t]=void 0)},stop:function(){this.done=!0;var e=this.tryEntries[0].completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(e){if(this.done)throw e;var t=this;function n(r,n){return c.type="throw",c.arg=e,t.next=r,n&&(t.method="next",t.arg=void 0),!!n}for(var o=this.tryEntries.length-1;o>=0;--o){var a=this.tryEntries[o],c=a.completion;if("root"===a.tryLoc)return n("end");if(a.tryLoc<=this.prev){var s=r.call(a,"catchLoc"),i=r.call(a,"finallyLoc");if(s&&i){if(this.prev<a.catchLoc)return n(a.catchLoc,!0);if(this.prev<a.finallyLoc)return n(a.finallyLoc)}else if(s){if(this.prev<a.catchLoc)return n(a.catchLoc,!0)}else{if(!i)throw new Error("try statement without catch or finally");if(this.prev<a.finallyLoc)return n(a.finallyLoc)}}}},abrupt:function(e,t){for(var n=this.tryEntries.length-1;n>=0;--n){var o=this.tryEntries[n];if(o.tryLoc<=this.prev&&r.call(o,"finallyLoc")&&this.prev<o.finallyLoc){var a=o;break}}a&&("break"===e||"continue"===e)&&a.tryLoc<=t&&t<=a.finallyLoc&&(a=null);var c=a?a.completion:{};return c.type=e,c.arg=t,a?(this.method="next",this.next=a.finallyLoc,l):this.complete(c)},complete:function(e,t){if("throw"===e.type)throw e.arg;return"break"===e.type||"continue"===e.type?this.next=e.arg:"return"===e.type?(this.rval=this.arg=e.arg,this.method="return",this.next="end"):"normal"===e.type&&t&&(this.next=t),l},finish:function(e){for(var t=this.tryEntries.length-1;t>=0;--t){var r=this.tryEntries[t];if(r.finallyLoc===e)return this.complete(r.completion,r.afterLoc),E(r),l}},catch:function(e){for(var t=this.tryEntries.length-1;t>=0;--t){var r=this.tryEntries[t];if(r.tryLoc===e){var n=r.completion;if("throw"===n.type){var o=n.arg;E(r)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(e,t,r){return this.delegate={iterator:C(e),resultName:t,nextLoc:r},"next"===this.method&&(this.arg=void 0),l}},e}var _=r(300),L={ethereum:{contractAddress:"0xA54F7579fFb3F98bd8649fF02813F575f9b3d353",chainId:1,name:"Ethereum",symbol:"ETH",decimals:18,type:"ERC1155",rpcUrls:["https://eth-mainnet.alchemyapi.io/v2/EuGnkVlzVoEkzdg0lpCarhm8YHOxWVxE"],blockExplorerUrls:["https://etherscan.io"],vmType:"EVM"},polygon:{contractAddress:"0x7C7757a9675f06F3BE4618bB68732c4aB25D2e88",chainId:137,name:"Polygon",symbol:"MATIC",decimals:18,rpcUrls:["https://polygon-rpc.com"],blockExplorerUrls:["https://explorer.matic.network"],type:"ERC1155",vmType:"EVM"},fantom:{contractAddress:"0x5bD3Fe8Ab542f0AaBF7552FAAf376Fd8Aa9b3869",chainId:250,name:"Fantom",symbol:"FTM",decimals:18,rpcUrls:["https://rpcapi.fantom.network"],blockExplorerUrls:["https://ftmscan.com"],type:"ERC1155",vmType:"EVM"},xdai:{contractAddress:"0xDFc2Fd83dFfD0Dafb216F412aB3B18f2777406aF",chainId:100,name:"xDai",symbol:"xDai",decimals:18,rpcUrls:["https://rpc.gnosischain.com"],blockExplorerUrls:[" https://blockscout.com/xdai/mainnet"],type:"ERC1155",vmType:"EVM"},bsc:{contractAddress:"0xc716950e5DEae248160109F562e1C9bF8E0CA25B",chainId:56,name:"Binance Smart Chain",symbol:"BNB",decimals:18,rpcUrls:["https://bsc-dataseed.binance.org/"],blockExplorerUrls:[" https://bscscan.com/"],type:"ERC1155",vmType:"EVM"},arbitrum:{contractAddress:"0xc716950e5DEae248160109F562e1C9bF8E0CA25B",chainId:42161,name:"Arbitrum",symbol:"AETH",decimals:18,type:"ERC1155",rpcUrls:["https://arb1.arbitrum.io/rpc"],blockExplorerUrls:["https://arbiscan.io/"],vmType:"EVM"},avalanche:{contractAddress:"0xBB118507E802D17ECDD4343797066dDc13Cde7C6",chainId:43114,name:"Avalanche",symbol:"AVAX",decimals:18,type:"ERC1155",rpcUrls:["https://api.avax.network/ext/bc/C/rpc"],blockExplorerUrls:["https://snowtrace.io/"],vmType:"EVM"},fuji:{contractAddress:"0xc716950e5DEae248160109F562e1C9bF8E0CA25B",chainId:43113,name:"Avalanche FUJI Testnet",symbol:"AVAX",decimals:18,type:"ERC1155",rpcUrls:["https://api.avax-test.network/ext/bc/C/rpc"],blockExplorerUrls:["https://testnet.snowtrace.io/"],vmType:"EVM"},harmony:{contractAddress:"0xBB118507E802D17ECDD4343797066dDc13Cde7C6",chainId:16666e5,name:"Harmony",symbol:"ONE",decimals:18,type:"ERC1155",rpcUrls:["https://api.harmony.one"],blockExplorerUrls:["https://explorer.harmony.one/"],vmType:"EVM"},kovan:{contractAddress:"0x9dB60Db3Dd9311861D87D33B0463AaD9fB4bb0E6",chainId:42,name:"Kovan",symbol:"ETH",decimals:18,rpcUrls:["https://kovan.infura.io/v3/ddf1ca3700f34497bca2bf03607fde38"],blockExplorerUrls:["https://kovan.etherscan.io"],type:"ERC1155",vmType:"EVM"},mumbai:{contractAddress:"0xc716950e5DEae248160109F562e1C9bF8E0CA25B",chainId:80001,name:"Mumbai",symbol:"MATIC",decimals:18,rpcUrls:["https://rpc-mumbai.maticvigil.com/v1/96bf5fa6e03d272fbd09de48d03927b95633726c"],blockExplorerUrls:["https://mumbai.polygonscan.com"],type:"ERC1155",vmType:"EVM"},goerli:{contractAddress:"0xc716950e5DEae248160109F562e1C9bF8E0CA25B",chainId:5,name:"Goerli",symbol:"ETH",decimals:18,rpcUrls:["https://goerli.infura.io/v3/96dffb3d8c084dec952c61bd6230af34"],blockExplorerUrls:["https://goerli.etherscan.io"],type:"ERC1155",vmType:"EVM"},ropsten:{contractAddress:"0x61544f0AE85f8fed6Eb315c406644eb58e15A1E7",chainId:3,name:"Ropsten",symbol:"ETH",decimals:18,rpcUrls:["https://ropsten.infura.io/v3/96dffb3d8c084dec952c61bd6230af34"],blockExplorerUrls:["https://ropsten.etherscan.io"],type:"ERC1155",vmType:"EVM"},rinkeby:{contractAddress:"0xc716950e5deae248160109f562e1c9bf8e0ca25b",chainId:4,name:"Rinkeby",symbol:"ETH",decimals:18,rpcUrls:["https://rinkeby.infura.io/v3/96dffb3d8c084dec952c61bd6230af34"],blockExplorerUrls:["https://rinkeby.etherscan.io"],type:"ERC1155",vmType:"EVM"},cronos:{contractAddress:"0xc716950e5DEae248160109F562e1C9bF8E0CA25B",chainId:25,name:"Cronos",symbol:"CRO",decimals:18,rpcUrls:["https://evm-cronos.org"],blockExplorerUrls:["https://cronos.org/explorer/"],type:"ERC1155",vmType:"EVM"},optimism:{contractAddress:"0xbF68B4c9aCbed79278465007f20a08Fa045281E0",chainId:10,name:"Optimism",symbol:"ETH",decimals:18,rpcUrls:["https://mainnet.optimism.io"],blockExplorerUrls:["https://optimistic.etherscan.io"],type:"ERC1155",vmType:"EVM"},celo:{contractAddress:"0xBB118507E802D17ECDD4343797066dDc13Cde7C6",chainId:42220,name:"Celo",symbol:"CELO",decimals:18,rpcUrls:["https://forno.celo.org"],blockExplorerUrls:["https://explorer.celo.org"],type:"ERC1155",vmType:"EVM"},aurora:{contractAddress:null,chainId:1313161554,name:"Aurora",symbol:"ETH",decimals:18,rpcUrls:["https://mainnet.aurora.dev"],blockExplorerUrls:["https://aurorascan.dev"],type:null,vmType:"EVM"},eluvio:{contractAddress:null,chainId:955305,name:"Eluvio",symbol:"ELV",decimals:18,rpcUrls:["https://host-76-74-28-226.contentfabric.io/eth"],blockExplorerUrls:["https://explorer.eluv.io"],type:null,vmType:"EVM"},alfajores:{contractAddress:null,chainId:44787,name:"Alfajores",symbol:"CELO",decimals:18,rpcUrls:["https://alfajores-forno.celo-testnet.org"],blockExplorerUrls:["https://alfajores-blockscout.celo-testnet.org"],type:null,vmType:"EVM"},xdc:{contractAddress:null,chainId:50,name:"XDC Blockchain",symbol:"XDC",decimals:18,rpcUrls:["https://rpc.xinfin.network"],blockExplorerUrls:["https://explorer.xinfin.network"],type:null,vmType:"EVM"},evmos:{contractAddress:null,chainId:9001,name:"EVMOS",symbol:"EVMOS",decimals:18,rpcUrls:["https://eth.bd.evmos.org:8545"],blockExplorerUrls:["https://evm.evmos.org"],type:null,vmType:"EVM"},evmosTestnet:{contractAddress:null,chainId:9e3,name:"EVMOS Testnet",symbol:"EVMOS",decimals:18,rpcUrls:["https://eth.bd.evmos.dev:8545"],blockExplorerUrls:["https://evm.evmos.dev"],type:null,vmType:"EVM"}};function T(){return j.apply(this,arguments)}function j(){return(j=Object(p.a)(I().mark((function e(){var t,r,n,o,a,c,s,i,u,l,p,d,h=arguments;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:for(t=h.length>0&&void 0!==h[0]?h[0]:{},r=t.chainId,n=void 0===r?1:r,o={},a=0;a<Object.keys(L).length;a++)c=Object.keys(L)[a],s=L[c].chainId,i=L[c].rpcUrls[0],o[s]=i;return{walletconnect:{package:g.default,options:{infuraId:"cd614bfa5c2f4703b7ab0ec0547d9f81",rpc:o,chainId:n}}},console.log("getting provider via lit connect modal"),u=new A.a.providers.Web3Provider(window.ethereum,"any"),console.log("got provider",u),l=u,e.next=10,u.send("eth_requestAccounts",[]);case 10:return console.log("listing accounts"),e.next=13,l.listAccounts();case 13:return p=e.sent,console.log("accounts",p),d=p[0].toLowerCase(),e.abrupt("return",{web3:l,account:d});case 17:case"end":return e.stop()}}),e)})))).apply(this,arguments)}var F=function(){var e=Object(p.a)(I().mark((function e(t,r,n){var o,a,c;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(o=Object(v.f)(n),!(t instanceof m.b)){e.next=21;break}return e.prev=2,console.log("Signing with personal_sign"),e.next=6,t.provider.send("personal_sign",[Object(b.hexlify)(o),r.toLowerCase()]);case 6:return a=e.sent,e.abrupt("return",a);case 10:if(e.prev=10,e.t0=e.catch(2),console.log("Signing with personal_sign failed, trying signMessage as a fallback"),e.t0 instanceof Error&&(c=e.t0.message),!c.includes("personal_sign")){e.next=18;break}return e.next=17,t.signMessage(o);case 17:return e.abrupt("return",e.sent);case 18:throw e.t0;case 19:e.next=25;break;case 21:return console.log("signing with signMessage"),e.next=24,t.signMessage(o);case 24:return e.abrupt("return",e.sent);case 25:case"end":return e.stop()}}),e,null,[[2,10]])})));return function(t,r,n){return e.apply(this,arguments)}}();function U(e){return M.apply(this,arguments)}function M(){return(M=Object(p.a)(I().mark((function e(t){var r,n,o,a,c,s,i;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(r=t.body,n=t.web3,o=t.account,n&&o){e.next=7;break}return e.next=4,T();case 4:a=e.sent,n=a.web3,o=a.account;case 7:return console.log("pausing..."),e.next=10,new Promise((function(e){return setTimeout(e,500)}));case 10:return console.log("signing with ",o),e.next=13,F(n.getSigner(),o,r);case 13:if(c=e.sent,s=Object(y.verifyMessage)(r,c).toLowerCase(),console.log("Signature: ",c),console.log("recovered address: ",s),s===o){e.next=22;break}throw i="ruh roh, the user signed with a different address (".concat(s,") then they're using with web3 (").concat(o,").  this will lead to confusion."),console.error(i),alert("something seems to be wrong with your wallets message signing.  maybe restart your browser or your wallet.  your recovered sig address does not match your web3 account address"),new Error(i);case 22:return e.abrupt("return",{signature:c,address:s});case 23:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function N(e){return B.apply(this,arguments)}function B(){return(B=Object(p.a)(I().mark((function e(t){var r,n,o,a,c,s,i,u,l,p;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=t.web3,n=t.account,o=t.chainId,a=t.resources,c={domain:globalThis.location.host,address:Object(w.a)(n),uri:globalThis.location.origin,version:"1",chainId:o},a&&a.length>0&&(c.resources=a),s=new x.SiweMessage(c),i=s.prepareMessage(),e.next=7,U({body:i,web3:r,account:n});case 7:return u=e.sent,l={sig:u.signature,derivedVia:"web3.eth.personal.sign",signedMessage:i,address:u.address},localStorage.setItem("lit-auth-signature",JSON.stringify(l)),p=S.a.box.keyPair(),localStorage.setItem("lit-comms-keypair",JSON.stringify({publicKey:k.a.encodeBase64(p.publicKey),secretKey:k.a.encodeBase64(p.secretKey)})),console.log("generated and saved lit-comms-keypair"),e.abrupt("return",l);case 14:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function V(e){return D.apply(this,arguments)}function D(){return(D=Object(p.a)(I().mark((function e(t){var r,n,o,a,c,s,i,u,l,p,d,h,f,m;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=t.chain,n=t.resources,o=t.switchChain,a=L[r],e.next=4,T({chainId:a.chainId});case 4:return c=e.sent,s=c.web3,i=c.account,console.log("got web3 and account: ".concat(i)),e.prev=8,e.next=11,s.getNetwork();case 11:l=e.sent,u=l.chainId,e.next=18;break;case 15:e.prev=15,e.t0=e.catch(8),console.log("getNetwork threw an exception",e.t0);case 18:if(p="0x"+a.chainId.toString("16"),console.log("chainId from web3",u),console.log("checkAndSignAuthMessage with chainId ".concat(u," and chain set to ").concat(r," and selectedChain is "),a),u===a.chainId||!o){e.next=43;break}if(!(s.provider instanceof g.default)){e.next=24;break}return e.abrupt("return");case 24:return e.prev=24,console.log("trying to switch to chainId",p),e.next=28,s.provider.request({method:"wallet_switchEthereumChain",params:[{chainId:p}]});case 28:e.next=42;break;case 30:return e.prev=30,e.t1=e.catch(24),console.log("error switching to chainId",e.t1),e.prev=33,d=[{chainId:p,chainName:a.name,nativeCurrency:{name:a.name,symbol:a.symbol,decimals:a.decimals},rpcUrls:a.rpcUrls,blockExplorerUrls:a.blockExplorerUrls}],e.next=37,s.provider.request({method:"wallet_addEthereumChain",params:d});case 37:e.next=42;break;case 39:throw e.prev=39,e.t2=e.catch(33),e.t2;case 42:u=a.chainId;case 43:if(console.log("checking if sig is in local storage"),h=localStorage.getItem("lit-auth-signature")){e.next=50;break}return console.log("signing auth message because sig is not in local storage"),e.next=49,N({web3:s,account:i,chainId:u,resources:n});case 49:h=localStorage.getItem("lit-auth-signature");case 50:if(h=JSON.parse(h),i===h.address){e.next=59;break}return console.log("signing auth message because account is not the same as the address in the auth sig"),e.next=55,N({web3:s,account:i,chainId:a.chainId,resources:n});case 55:h=localStorage.getItem("lit-auth-signature"),h=JSON.parse(h),e.next=66;break;case 59:f=!1;try{m=new x.SiweMessage(h.signedMessage),console.log("parsedSiwe.resources",m.resources),JSON.stringify(m.resources)!==JSON.stringify(n)?(console.log("signing auth message because resources differ from the resources in the auth sig"),f=!0):m.address!==Object(w.a)(m.address)&&(console.log("signing auth message because parsedSig.address is not equal to the same address but checksummed.  This usually means the user had a non-checksummed address saved and so they need to re-sign."),f=!0)}catch(y){console.log("error parsing siwe sig.  making the user sign again: ",y),f=!0}if(!f){e.next=66;break}return e.next=64,N({web3:s,account:i,chainId:a.chainId,resources:n});case 64:h=localStorage.getItem("lit-auth-signature"),h=JSON.parse(h);case 66:return console.log("got auth sig",h),e.abrupt("return",h);case 68:case"end":return e.stop()}}),e,null,[[8,15],[24,30],[33,39]])})))).apply(this,arguments)}function R(e){return P.apply(this,arguments)}function P(){return(P=Object(p.a)(I().mark((function e(t){var r,n,o,a;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=t.chain,n=t.resources,o=t.switchChain,a=void 0===o||o,e.abrupt("return",V({chain:r,resources:n,switchChain:a}));case 2:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function G(e){return H.apply(this,arguments)}function H(){return(H=Object(p.a)(I().mark((function e(t){var r;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,R({chain:t});case 2:return r=e.sent,window.authSig=r,e.abrupt("return",r);case 5:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function K(){return q.apply(this,arguments)}function q(){return(q=Object(p.a)(I().mark((function e(){var t;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=new _.LitNodeClient,e.next=3,t.connect();case 3:return window.litNodeClient=t,e.abrupt("return",t);case 5:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function J(e,t){return W.apply(this,arguments)}function W(){return(W=Object(p.a)(I().mark((function e(t,r){var n,o,a,c,s,i,u,l;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,K();case 2:return e.sent,n=[{contractAddress:"0x68085453B798adf9C09AD8861e0F0da96B908d81",standardContractType:"ERC1155",chain:r,method:"balanceOf",parameters:[":userAddress","0","1","2","3","4","5"],returnValueTest:{comparator:">",value:"0"}}],console.log("getting authSig"),e.next=7,G(r);case 7:return o=e.sent,console.log("got authSig ",o),a=r,e.next=12,_.encryptString(t);case 12:return c=e.sent,s=c.encryptedString,i=c.symmetricKey,e.next=17,window.litNodeClient.saveEncryptionKey({accessControlConditions:n,symmetricKey:i,authSig:o,chain:a});case 17:return u=e.sent,window.encryptedString=s,e.next=21,_.blobToBase64String(s);case 21:return l=e.sent,e.abrupt("return",{encryptedRealString:l,encryptedSymmetricKey:_.uint8arrayToString(u,"base16")});case 23:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function Y(e,t,r){return X.apply(this,arguments)}function X(){return(X=Object(p.a)(I().mark((function e(t,r,n){var o,a,c,s,i,u,l;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return console.log("Decrypting..."),o=_.base64StringToBlob(t),e.next=4,K();case 4:return a=e.sent,e.next=7,G(n);case 7:return c=e.sent,s=n,window.accessControlConditions=[{contractAddress:"0x68085453B798adf9C09AD8861e0F0da96B908d81",standardContractType:"ERC1155",chain:n,method:"balanceOf",parameters:[":userAddress","0","1","2","3","4","5"],returnValueTest:{comparator:">",value:"0"}}],i=window.accessControlConditions,e.next=13,a.getEncryptionKey({accessControlConditions:i,toDecrypt:r,chain:s,authSig:c});case 13:return u=e.sent,e.next=16,_.decryptString(o,u);case 16:return l=e.sent,e.abrupt("return",{decryptedString:l});case 18:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function z(){return Q.apply(this,arguments)}function Q(){return(Q=Object(p.a)(I().mark((function e(){var t,r,n,o=arguments;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=o.length>0&&void 0!==o[0]?o[0]:"ERC1155",window.accessControlConditions=[{contractAddress:_.LIT_CHAINS[window.chain].contractAddress,standardContractType:t,chain:window.chain,method:"balanceOf",parameters:[":userAddress",window.tokenId.toString()],returnValueTest:{comparator:">",value:"0"}}],r="/"+Math.random().toString(36).substring(2,15)+Math.random().toString(36).substring(2,15),window.resourceId={baseUrl:"my-dynamic-content-server.com",path:r,orgId:"",role:"",extraData:""},n=new _.LitNodeClient,e.next=7,n.connect();case 7:return window.litNodeClient=n,console.log("Lit client connected",n),console.log("Window.litNodeClient",window.litNodeClient),e.next=12,n.saveSigningCondition({accessControlConditions:window.accessControlConditions,chain:window.chain,authSig:window.authSig,resourceId:window.resourceId});case 12:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function Z(e,t){return $.apply(this,arguments)}function $(){return($=Object(p.a)(I().mark((function e(t,r){var n,o,a,c,s=arguments;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return n=s.length>2&&void 0!==s[2]?s[2]:"ERC1155",o=s.length>3&&void 0!==s[3]?s[3]:"0",window.accessControlConditions=[{contractAddress:t,standardContractType:n,chain:r,method:"balanceOf",parameters:[":userAddress","0","1","2","3","4","5"],returnValueTest:{comparator:">",value:o}}],a="/"+Math.random().toString(36).substring(2,15)+Math.random().toString(36).substring(2,15),window.resourceId={baseUrl:"lit-estuary-storage.herokuapp.com/",path:a,orgId:"",role:"",extraData:""},c=new _.LitNodeClient,e.next=8,c.connect();case 8:return window.litNodeClient=c,console.log("Lit client connected",c),console.log("Window.litNodeClient",window.litNodeClient),e.next=13,c.saveSigningCondition({accessControlConditions:window.accessControlConditions,chain:r,authSig:window.authSig,resourceId:window.resourceId});case 13:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function ee(e){return te.apply(this,arguments)}function te(){return(te=Object(p.a)(I().mark((function e(t){var r;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=new _.LitNodeClient,e.next=3,r.connect();case 3:return window.litNodeClient=r,console.log("Lit client connected",r),console.log("Window.litNodeClient",window.litNodeClient),e.next=8,r.getSignedToken({accessControlConditions:window.accessControlConditions,chain:t,authSig:window.authSig,resourceId:window.resourceId});case 8:window.jwt=e.sent;case 9:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function re(e){return ne.apply(this,arguments)}function ne(){return(ne=Object(p.a)(I().mark((function e(t){var r,n,o,a,c,s,i,u,l,p,d;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=t.chain,n=t.quantity,console.log("minting ".concat(n," tokens on ").concat(r)),e.prev=2,e.next=5,V({chain:r,switchChain:!0});case 5:if(!(o=e.sent).errorCode){e.next=8;break}return e.abrupt("return",o);case 8:return e.next=10,T();case 10:if(a=e.sent,c=a.web3,s=a.account,i=L[r].contractAddress){e.next=17;break}return console.log("No token address for this chain.  It's not supported via MintLIT."),e.abrupt("return");case 17:return u=new f.Contract(i,O.abi,c.getSigner()),console.log("sending to chain..."),e.next=21,u.mint(n);case 21:return l=e.sent,console.log("sent to chain.  waiting to be mined..."),e.next=25,l.wait();case 25:return p=e.sent,console.log("txReceipt: ",p),d=p.events[0].args[3].toNumber(),e.abrupt("return",{txHash:p.transactionHash,tokenId:d,tokenAddress:i,mintingAddress:s,authSig:o});case 31:return e.prev=31,e.t0=e.catch(2),console.log(e.t0),e.abrupt("return",{errorCode:"unknown_error"});case 35:case"end":return e.stop()}}),e,null,[[2,31]])})))).apply(this,arguments)}function oe(e){return ae.apply(this,arguments)}function ae(){return(ae=Object(p.a)(I().mark((function e(t){var r,n,o,a,c;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return console.log("Minting NFT, please wait for the tx to confirm..."),window.chain=t,e.next=4,re({chain:window.chain,quantity:1});case 4:return r=e.sent,n=r.txHash,o=r.tokenId,a=r.tokenAddress,r.mintingAddress,c=r.authSig,window.tokenId=o,window.tokenAddress=a,window.authSig=c,e.abrupt("return",n);case 14:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function ce(e,t){return se.apply(this,arguments)}function se(){return(se=Object(p.a)(I().mark((function e(t,r){var n,o,a=arguments;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return n=a.length>2&&void 0!==a[2]?a[2]:"ERC1155",o=a.length>3&&void 0!==a[3]?a[3]:"0",e.prev=2,e.next=5,G(r);case 5:return e.next=7,Z(t,r,n,o);case 7:return e.next=9,ee(r);case 9:return console.log("You're logged in!"),console.log("window.jwt",window.jwt),e.abrupt("return",!0);case 14:return e.prev=14,e.t0=e.catch(2),console.log("Error",e.t0),e.abrupt("return",!1);case 18:case"end":return e.stop()}}),e,null,[[2,14]])})))).apply(this,arguments)}function ie(e){return ue.apply(this,arguments)}function ue(){return(ue=Object(p.a)(I().mark((function e(t){var r,n,o=arguments;return I().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=o.length>1&&void 0!==o[1]?o[1]:"ERC1155",e.prev=1,e.next=4,G(t);case 4:return e.next=6,oe(t);case 6:return n=e.sent,console.log("tx",n),e.next=10,z(r);case 10:return e.next=12,ee(t);case 12:return console.log("You're logged in!"),console.log("window.jwt",window.jwt),e.abrupt("return",!0);case 17:return e.prev=17,e.t0=e.catch(1),console.log("Error",e.t0),e.abrupt("return",!1);case 21:case"end":return e.stop()}}),e,null,[[1,17]])})))).apply(this,arguments)}function le(){le=function(){return e};var e={},t=Object.prototype,r=t.hasOwnProperty,n="function"==typeof Symbol?Symbol:{},o=n.iterator||"@@iterator",a=n.asyncIterator||"@@asyncIterator",c=n.toStringTag||"@@toStringTag";function s(e,t,r){return Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}),e[t]}try{s({},"")}catch(A){s=function(e,t,r){return e[t]=r}}function i(e,t,r,n){var o=t&&t.prototype instanceof p?t:p,a=Object.create(o.prototype),c=new k(n||[]);return a._invoke=function(e,t,r){var n="suspendedStart";return function(o,a){if("executing"===n)throw new Error("Generator is already running");if("completed"===n){if("throw"===o)throw a;return S()}for(r.method=o,r.arg=a;;){var c=r.delegate;if(c){var s=b(c,r);if(s){if(s===l)continue;return s}}if("next"===r.method)r.sent=r._sent=r.arg;else if("throw"===r.method){if("suspendedStart"===n)throw n="completed",r.arg;r.dispatchException(r.arg)}else"return"===r.method&&r.abrupt("return",r.arg);n="executing";var i=u(e,t,r);if("normal"===i.type){if(n=r.done?"completed":"suspendedYield",i.arg===l)continue;return{value:i.arg,done:r.done}}"throw"===i.type&&(n="completed",r.method="throw",r.arg=i.arg)}}}(e,r,c),a}function u(e,t,r){try{return{type:"normal",arg:e.call(t,r)}}catch(A){return{type:"throw",arg:A}}}e.wrap=i;var l={};function p(){}function d(){}function h(){}var f={};s(f,o,(function(){return this}));var g=Object.getPrototypeOf,m=g&&g(g(C([])));m&&m!==t&&r.call(m,o)&&(f=m);var y=h.prototype=p.prototype=Object.create(f);function v(e){["next","throw","return"].forEach((function(t){s(e,t,(function(e){return this._invoke(t,e)}))}))}function w(e,t){var n;this._invoke=function(o,a){function c(){return new t((function(n,c){!function n(o,a,c,s){var i=u(e[o],e,a);if("throw"!==i.type){var l=i.arg,p=l.value;return p&&"object"==typeof p&&r.call(p,"__await")?t.resolve(p.__await).then((function(e){n("next",e,c,s)}),(function(e){n("throw",e,c,s)})):t.resolve(p).then((function(e){l.value=e,c(l)}),(function(e){return n("throw",e,c,s)}))}s(i.arg)}(o,a,n,c)}))}return n=n?n.then(c,c):c()}}function b(e,t){var r=e.iterator[t.method];if(void 0===r){if(t.delegate=null,"throw"===t.method){if(e.iterator.return&&(t.method="return",t.arg=void 0,b(e,t),"throw"===t.method))return l;t.method="throw",t.arg=new TypeError("The iterator does not provide a 'throw' method")}return l}var n=u(r,e.iterator,t.arg);if("throw"===n.type)return t.method="throw",t.arg=n.arg,t.delegate=null,l;var o=n.arg;return o?o.done?(t[e.resultName]=o.value,t.next=e.nextLoc,"return"!==t.method&&(t.method="next",t.arg=void 0),t.delegate=null,l):o:(t.method="throw",t.arg=new TypeError("iterator result is not an object"),t.delegate=null,l)}function x(e){var t={tryLoc:e[0]};1 in e&&(t.catchLoc=e[1]),2 in e&&(t.finallyLoc=e[2],t.afterLoc=e[3]),this.tryEntries.push(t)}function E(e){var t=e.completion||{};t.type="normal",delete t.arg,e.completion=t}function k(e){this.tryEntries=[{tryLoc:"root"}],e.forEach(x,this),this.reset(!0)}function C(e){if(e){var t=e[o];if(t)return t.call(e);if("function"==typeof e.next)return e;if(!isNaN(e.length)){var n=-1,a=function t(){for(;++n<e.length;)if(r.call(e,n))return t.value=e[n],t.done=!1,t;return t.value=void 0,t.done=!0,t};return a.next=a}}return{next:S}}function S(){return{value:void 0,done:!0}}return d.prototype=h,s(y,"constructor",h),s(h,"constructor",d),d.displayName=s(h,c,"GeneratorFunction"),e.isGeneratorFunction=function(e){var t="function"==typeof e&&e.constructor;return!!t&&(t===d||"GeneratorFunction"===(t.displayName||t.name))},e.mark=function(e){return Object.setPrototypeOf?Object.setPrototypeOf(e,h):(e.__proto__=h,s(e,c,"GeneratorFunction")),e.prototype=Object.create(y),e},e.awrap=function(e){return{__await:e}},v(w.prototype),s(w.prototype,a,(function(){return this})),e.AsyncIterator=w,e.async=function(t,r,n,o,a){void 0===a&&(a=Promise);var c=new w(i(t,r,n,o),a);return e.isGeneratorFunction(r)?c:c.next().then((function(e){return e.done?e.value:c.next()}))},v(y),s(y,c,"Generator"),s(y,o,(function(){return this})),s(y,"toString",(function(){return"[object Generator]"})),e.keys=function(e){var t=[];for(var r in e)t.push(r);return t.reverse(),function r(){for(;t.length;){var n=t.pop();if(n in e)return r.value=n,r.done=!1,r}return r.done=!0,r}},e.values=C,k.prototype={constructor:k,reset:function(e){if(this.prev=0,this.next=0,this.sent=this._sent=void 0,this.done=!1,this.delegate=null,this.method="next",this.arg=void 0,this.tryEntries.forEach(E),!e)for(var t in this)"t"===t.charAt(0)&&r.call(this,t)&&!isNaN(+t.slice(1))&&(this[t]=void 0)},stop:function(){this.done=!0;var e=this.tryEntries[0].completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(e){if(this.done)throw e;var t=this;function n(r,n){return c.type="throw",c.arg=e,t.next=r,n&&(t.method="next",t.arg=void 0),!!n}for(var o=this.tryEntries.length-1;o>=0;--o){var a=this.tryEntries[o],c=a.completion;if("root"===a.tryLoc)return n("end");if(a.tryLoc<=this.prev){var s=r.call(a,"catchLoc"),i=r.call(a,"finallyLoc");if(s&&i){if(this.prev<a.catchLoc)return n(a.catchLoc,!0);if(this.prev<a.finallyLoc)return n(a.finallyLoc)}else if(s){if(this.prev<a.catchLoc)return n(a.catchLoc,!0)}else{if(!i)throw new Error("try statement without catch or finally");if(this.prev<a.finallyLoc)return n(a.finallyLoc)}}}},abrupt:function(e,t){for(var n=this.tryEntries.length-1;n>=0;--n){var o=this.tryEntries[n];if(o.tryLoc<=this.prev&&r.call(o,"finallyLoc")&&this.prev<o.finallyLoc){var a=o;break}}a&&("break"===e||"continue"===e)&&a.tryLoc<=t&&t<=a.finallyLoc&&(a=null);var c=a?a.completion:{};return c.type=e,c.arg=t,a?(this.method="next",this.next=a.finallyLoc,l):this.complete(c)},complete:function(e,t){if("throw"===e.type)throw e.arg;return"break"===e.type||"continue"===e.type?this.next=e.arg:"return"===e.type?(this.rval=this.arg=e.arg,this.method="return",this.next="end"):"normal"===e.type&&t&&(this.next=t),l},finish:function(e){for(var t=this.tryEntries.length-1;t>=0;--t){var r=this.tryEntries[t];if(r.finallyLoc===e)return this.complete(r.completion,r.afterLoc),E(r),l}},catch:function(e){for(var t=this.tryEntries.length-1;t>=0;--t){var r=this.tryEntries[t];if(r.tryLoc===e){var n=r.completion;if("throw"===n.type){var o=n.arg;E(r)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(e,t,r){return this.delegate={iterator:C(e),resultName:t,nextLoc:r},"next"===this.method&&(this.arg=void 0),l}},e}function pe(){return de.apply(this,arguments)}function de(){return(de=Object(p.a)(le().mark((function e(){var t,r,n;return le().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=new h.providers.Web3Provider(window.ethereum,"any"),e.next=3,window.ethereum.request({method:"eth_requestAccounts"});case 3:return e.sent,e.next=6,t.send("eth_requestAccounts",[]);case 6:return window.ethereum.on("accountsChanged",(function(e){})),r=t.getSigner(),r="0",r=t.getSigner(),e.next=12,r.getAddress();case 12:return n=e.sent,e.abrupt("return",n);case 14:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function he(e,t){return fe.apply(this,arguments)}function fe(){return(fe=Object(p.a)(le().mark((function e(t,r){var n,o,a,c,s,i,u,l=arguments;return le().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return n=l.length>2&&void 0!==l[2]?l[2]:"0x8967BCF84170c91B0d24D4302C2376283b0B3a07",console.log("Sending OCEAN initiated"),o=n,a=[{name:"transfer",type:"function",inputs:[{name:"_to",type:"address"},{type:"uint256",name:"_tokens"}],constant:!1,outputs:[{name:"",type:"bool"}],payable:!1}],console.log("Parameters defined"),c=new h.providers.Web3Provider(window.ethereum,"any"),e.next=8,c.send("eth_requestAccounts",[]);case 8:s=c.getSigner(),i=new f.Contract(o,a,s),console.log("Contract defined"),u=h.utils.parseUnits(r,18),console.log("numberOfTokens: ".concat(u)),console.log("Ready to transfer"),i.transfer(t,u).then((function(e){console.dir(e),console.log("sent token")})),console.log("Done: see address below on etherscan"),console.log(t);case 17:case"end":return e.stop()}}),e)})))).apply(this,arguments)}var ge=function(e){Object(u.a)(r,e);var t=Object(l.a)(r);function r(){var e;Object(i.a)(this,r);for(var n=arguments.length,a=new Array(n),c=0;c<n;c++)a[c]=arguments[c];return(e=t.call.apply(t,[this].concat(a))).state={walletAddress:"not",transaction:"",isFocused:!1,encryptedString:"",encryptedSymmetricKey:"",decryptedString:"",loggedIn:!1},e.render=function(){var t=e.props.theme,r={};if(t){var n="0px solid ".concat(e.state.isFocused?t.primaryColor:"gray"),a="".concat(e.state.isFocused?"#4F8BF9":"#FF4B4B");r.border=n,r.outline=n,r.backgroundColor=a,r.color="white",r.borderRadius="0.2rem",r.height="2em"}var c=e.props.args.message;return o.a.createElement("span",null,o.a.createElement("button",{style:r,onClick:e.onClicked,disabled:e.props.disabled,onFocus:e._onFocus,onBlur:e._onBlur,onMouseOver:e._onFocus,onMouseOut:e._onBlur},c))},e.onClicked=Object(p.a)(le().mark((function t(){var r,n,o,a,c,s,i,u,l;return le().wrap((function(t){for(;;)switch(t.prev=t.next){case 0:if("wallet"!==e.props.args.key){t.next=7;break}return t.next=3,pe();case 3:r=t.sent,e.setState((function(){return{walletAddress:r}}),(function(){return d.a.setComponentValue(e.state.walletAddress)})),t.next=44;break;case 7:if("send"!==e.props.args.key){t.next=14;break}return t.next=10,he(e.props.args.to_address,e.props.args.amount,e.props.args.contract_address);case 10:n=t.sent,e.setState((function(){return{transaction:n}}),(function(){return d.a.setComponentValue(e.state.transaction)})),t.next=44;break;case 14:if("encrypt"!==e.props.args.key){t.next=23;break}return t.next=17,J(e.props.args.message_to_encrypt,e.props.args.chain_name);case 17:o=t.sent,a=o.encryptedRealString,c=o.encryptedSymmetricKey,e.setState((function(){return{encryptedString:a,encryptedSymmetricKey:c}}),(function(){return d.a.setComponentValue({encryptedRealString:a,encryptedSymmetricKey:c})})),t.next=44;break;case 23:if("decrypt"!==e.props.args.key){t.next=32;break}return t.next=26,Y(e.props.args.encrypted_string,e.props.args.encrypted_symmetric_key,e.props.args.chain_name);case 26:s=t.sent,i=s.decryptedString,e.setState((function(){return{decryptedString:i}}),(function(){return d.a.setComponentValue(i)})),console.log("State of encrypted string3:",e.state.encryptedString),t.next=44;break;case 32:if("login"!==e.props.args.key){t.next=39;break}return t.next=35,ce(e.props.args.auth_token_contract_address,e.props.args.chain_name,e.props.args.contract_type,e.props.args.num_tokens);case 35:u=t.sent,e.setState((function(){return{loggedIn:u}}),(function(){return d.a.setComponentValue(u)})),t.next=44;break;case 39:if("mint_and_login"!==e.props.args.key){t.next=44;break}return t.next=42,ie(e.props.args.chain_name,e.props.args.contract_type);case 42:l=t.sent,e.setState((function(){return{loggedIn:l}}),(function(){return d.a.setComponentValue(l)}));case 44:case"end":return t.stop()}}),t)}))),e._onFocus=function(){e.setState({isFocused:!0})},e._onBlur=function(){e.setState({isFocused:!1})},e}return Object(s.a)(r)}(d.b),me=Object(d.c)(ge);c.a.render(o.a.createElement(o.a.StrictMode,null,o.a.createElement(me,null)),document.getElementById("root"))}},[[215,1,2]]]);
//# sourceMappingURL=main.35445d24.chunk.js.map