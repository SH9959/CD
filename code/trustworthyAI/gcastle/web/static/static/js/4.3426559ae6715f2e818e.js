webpackJsonp([4],{"Qkm/":function(t,e){},QqLt:function(t,e){},mRzr:function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var a={name:"readme",components:{markdown:n("SHp6").a},data:function(){return{name:"",content:"",mdtext:""}},watch:{content:{deep:!0,handler:function(t,e){}}},methods:{},created:function(){this.name=this.$route.params.name},activated:function(){var t=this;this.name=this.$route.params.name,this.$http.get("static/md/"+this.name+".md").then(function(e){t.mdtext=e.body})},deactivated:function(){this.mdtext=""}},i={render:function(){var t=this.$createElement,e=this._self._c||t;return e("div",[e("markdown",{attrs:{mdtext:this.mdtext}})],1)},staticRenderFns:[]};var r={name:"realWorldData",components:{readme:n("VU/8")(a,i,!1,function(t){n("QqLt")},"data-v-447576f6",null).exports},data:function(){return{name:"",content:"",mdtext:""}},watch:{content:{deep:!0,handler:function(t,e){}}},methods:{goBack:function(){this.$router.push({path:this.$route.meta.parentPath})}},created:function(){this.name=this.$route.params.name},activated:function(){this.name=this.$route.params.name},deactivated:function(){}},o={render:function(){var t=this.$createElement,e=this._self._c||t;return e("div",[e("el-page-header",{attrs:{content:this.$t("readme.detail")},on:{back:this.goBack}}),this._v(" "),e("el-divider"),this._v(" "),e("readme")],1)},staticRenderFns:[]};var c=n("VU/8")(r,o,!1,function(t){n("Qkm/")},"data-v-67414be0",null);e.default=c.exports}});