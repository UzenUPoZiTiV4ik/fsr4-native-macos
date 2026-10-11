// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// ============================================================================
// 4.1.1 pass12 2160-class reimplementation (our_reimplementation)
// Global 5129 geometry x2: FFX_FSR4_PASS_24, ConvNextBlock at 1920x1080.
// Transformed from reimpl_pass12.hlsl by the 2x law confirmed against AMD
// 2160 container 0401_00b14220_7dcb64562f01cc7a.dxbc
// (fsr4_model_v07_fp8_no_scale_pass12, 38416 B = same size as 1080 quality):
//   entry guard 959/539 -> 1919/1079; read stride 15392 -> 30752;
//   read plane base 12537792 -> 49958592 (= pass11-2160 output plane);
//   write plane base 0 unchanged, write stride 15392 -> 30752;
//   store index = read-window address + small delta (no extra immediates);
//   CBV reg13 guard (1920,1080) via doubled CBV slice; OOB sink unchanged.
// Weights are DXIL literals identical across classes (48-line diff is
// geometry-only); t18 biases, requant shifts, tail float scales unchanged.
// NOTE: #ifdef DIAG* blocks are inactive 1080-legacy debug scaffolding.
// dispatch 30,1080,1 (ceil(1920/64) x 1080), numthreads 64x1x1, root compute.
// scratch class 2160 = 83232256 B.
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

cbuffer MLSR_Optimized_Constants : register(b0)
{
    uint4 ml2c_cb[17]; // reg3.xy = logical tensor width/height
}

static const uint W1[36][16] = {
    { 0x02020700u, 0x060202fdu, 0x0305faffu, 0xfaf7ebffu, 0xfef004f5u, 0xff1c0d0bu, 0x06070ef8u, 0x02fff006u, 0xfb10fc06u, 0x07fdf800u, 0xfb0606feu, 0xfd020905u, 0x08080402u, 0x01f6fa0au, 0xeef6fcfbu, 0x0c020b15u }, // block 0 (dxil label 94)
    { 0x010711feu, 0xff0efd00u, 0xf5ff0609u, 0xff0af60du, 0x06130b0au, 0xe82b09fcu, 0xf7f6f109u, 0x060ffd06u, 0xfeec0b03u, 0x1006ff06u, 0x01f51af0u, 0xfafdf607u, 0x0aeb00f8u, 0xf5e5fb06u, 0xfaf1f903u, 0xfcff04f5u }, // block 1 (dxil label 129)
    { 0x02080402u, 0x0c02fd01u, 0x0005fc02u, 0xfdff010au, 0x02f901feu, 0xfb0cfc04u, 0xfffb0903u, 0x03081005u, 0xfafb0700u, 0x05050af7u, 0x0c01faefu, 0xf000e6eeu, 0x03edf103u, 0x0205fdffu, 0x01f604feu, 0x03f502ffu }, // block 2 (dxil label 161)
    { 0xfb0a11fau, 0x020c050au, 0xf7f1f6ffu, 0x041002ffu, 0x021a1a0cu, 0xeb0d0e09u, 0xf100ea05u, 0x0e1bff08u, 0xf7f1f9ffu, 0x0deef8fcu, 0xf6f608e9u, 0xfbf908fdu, 0x050004ffu, 0xf50b010bu, 0xf8eff8f2u, 0x03f001f6u }, // block 3 (dxil label 193)
    { 0xe9cc0609u, 0xe8eb07e4u, 0x0612142bu, 0xfdbb26d7u, 0xe8cbf0fau, 0xf5ac9af7u, 0x24b902d5u, 0x315705f5u, 0x0717f8f3u, 0xf4daf72cu, 0xe7dbff28u, 0x2f2446feu, 0xebda1d1au, 0xadcceccbu, 0x1604d6f7u, 0x19061eebu }, // block 4 (dxil label 225)
    { 0x05fb02f3u, 0x0cfd03feu, 0x04f6f401u, 0xf706eeedu, 0x03fc0c02u, 0xef12fbf7u, 0xff02fc16u, 0x0600ffefu, 0xf5030107u, 0x08f60afcu, 0xfa02fbe2u, 0xf6f9fffdu, 0x0904fc06u, 0x00fd0107u, 0x00eef804u, 0x0103f6f0u }, // block 5 (dxil label 257)
    { 0x01080302u, 0x0bfcfe03u, 0x0103fa01u, 0xff03fc07u, 0xf8f20cf8u, 0xf40c030bu, 0x01f10f03u, 0x04ff0606u, 0xfef60100u, 0x01070af7u, 0x0608fdf5u, 0xf4ffecf0u, 0xfef0feffu, 0x06fa0102u, 0xfffa04feu, 0xfdfcfefcu }, // block 6 (dxil label 289)
    { 0x06fc0fefu, 0x02f501fcu, 0xfdf9f7f7u, 0xfdfaea06u, 0xeffa050eu, 0xfa160dfbu, 0x0d030314u, 0xfa02ffdfu, 0x06030b03u, 0x080205ffu, 0xfa00f8f0u, 0xfdf9eb00u, 0x0ff6f9ffu, 0x06f1f607u, 0x04f0fd0au, 0xfff903f8u }, // block 7 (dxil label 321)
    { 0xfe03040au, 0x0a1202feu, 0x02fff904u, 0xfbf8f0f8u, 0xfef601f9u, 0xfe070302u, 0x04010c08u, 0xfb03f902u, 0xfe0dfd07u, 0x01fe05fbu, 0x010dfff4u, 0xf5060100u, 0x05fbfd04u, 0x06fbfc06u, 0x02fe0402u, 0x01f402feu }, // block 8 (dxil label 353)
    { 0x01fafffeu, 0x02fe02fau, 0x07fafd0cu, 0xfc0000fcu, 0x020e0f02u, 0xffeffdf5u, 0xf700f605u, 0xfafdfffeu, 0x0008f706u, 0xfe13fe07u, 0xf3030dffu, 0x0c040716u, 0x01150503u, 0xf7e1fcfeu, 0xfffbed09u, 0x040efff3u }, // block 9 (dxil label 468)
    { 0xfc0cfe05u, 0x0bff04ffu, 0x07fffe04u, 0xfbf7fef5u, 0x030202ffu, 0xe7fafefdu, 0xfc040917u, 0x0107f8feu, 0xff000b0cu, 0x02fd0dfbu, 0xfa08fc01u, 0xfe09f6f4u, 0x0000f9ffu, 0x03ef0201u, 0x0109ff09u, 0x0bfb0ef9u }, // block 10 (dxil label 500)
    { 0x05fe0504u, 0x02f9fd00u, 0xfc000006u, 0xfefe0007u, 0xfc00fe04u, 0xfd0101f4u, 0x010afe0au, 0xf6fffdffu, 0xfeedf6ffu, 0xf803fe01u, 0xfef200fau, 0x0a09fdf8u, 0xfa1505fdu, 0x08e90502u, 0xfc03ecffu, 0x0613ffeau }, // block 11 (dxil label 532)
    { 0x080b0501u, 0x0bf9f9fdu, 0xf9140409u, 0xf1f9f8feu, 0xf601ff0bu, 0xe52204fbu, 0x04fb0506u, 0x0100f9f4u, 0xfe180805u, 0x05ee05fcu, 0xfbfdf205u, 0x0001e9fdu, 0xfc07f60au, 0xfbeefcf4u, 0xfd08f7f5u, 0x0cfb12f9u }, // block 12 (dxil label 564)
    { 0xfe26d6feu, 0x1cdde904u, 0x204f23c9u, 0xdc21e6ffu, 0xfa250ce7u, 0xb2fcfc09u, 0xd8011223u, 0x0f1931fdu, 0x07200c0cu, 0x1fc82de6u, 0x2103f723u, 0xcd34ddddu, 0xedf01bcau, 0x09dacfd3u, 0xf6dbf409u, 0x15eae012u }, // block 13 (dxil label 596)
    { 0xfa06030bu, 0x04070507u, 0x0004fb07u, 0x0207fcf7u, 0x040808ffu, 0x00f807feu, 0x05fc020eu, 0xf501f4f4u, 0x020b0915u, 0xf22102f6u, 0xf302fb0du, 0x0a03fcffu, 0xf21ef510u, 0xf4fb0ef7u, 0x030cf1f8u, 0x06fe02f9u }, // block 14 (dxil label 628)
    { 0x04030203u, 0x03f8f901u, 0x01fcfd02u, 0x01fb0508u, 0xfa040306u, 0x01f906f8u, 0x0307f905u, 0xf600fafeu, 0xfef4fbf9u, 0xf0fefdffu, 0xfcf30302u, 0x0701fffdu, 0xf90f0406u, 0x0b010b00u, 0x0009f401u, 0x060cf6ecu }, // block 15 (dxil label 660)
    { 0x000e0903u, 0x12030007u, 0xfd160401u, 0xf9fe02ffu, 0x0506fbfcu, 0x06e503ffu, 0x02090a15u, 0xf4fcfcfeu, 0xf90a0f0bu, 0xec050dffu, 0xf5020112u, 0x0001faf5u, 0x070ff110u, 0x01f906f7u, 0x060bfc0eu, 0x0c0a06f7u }, // block 16 (dxil label 692)
    { 0xfc0109ffu, 0xfa0103feu, 0x01fdfe03u, 0xff07f800u, 0xf7050304u, 0xfe0906fcu, 0xfd0c07fbu, 0xf5fdfafcu, 0x010605fbu, 0x00e7fdfeu, 0xfdf7f705u, 0x080b0b01u, 0xff14f90bu, 0xfcf503fdu, 0x0113f908u, 0xfb0c00f7u }, // block 17 (dxil label 724)
    { 0x01ff020au, 0xf9fdfafdu, 0xf7fb050eu, 0x070bfa0bu, 0x0106fc03u, 0xf708f704u, 0xfc020c00u, 0x060b0009u, 0xff0508ffu, 0x02fb04feu, 0x0102f8f9u, 0x0202e7fau, 0xfd110302u, 0x07fafd05u, 0xfa0301ffu, 0x01030a06u }, // block 18 (dxil label 819)
    { 0xf9e7f1f6u, 0xfae706f7u, 0x040c0811u, 0xf5eefef4u, 0x071302f7u, 0x01e6f403u, 0xff02ff08u, 0xfb0502fdu, 0x00f5ff07u, 0xfe12fdfdu, 0xfcfefe05u, 0x0b08fe0bu, 0xfcfd09ffu, 0x0016fa05u, 0xfdf80ffcu, 0x030afc03u }, // block 19 (dxil label 851)
    { 0xfafff202u, 0x01f6ff01u, 0x04f4fefeu, 0x030bf7fcu, 0xfaf9f9fcu, 0x020301feu, 0x04fbfffeu, 0xff0b05fbu, 0x000b0301u, 0x00fd0000u, 0xfdfff601u, 0x0101fe02u, 0x020609ffu, 0x060103fcu, 0x03fdfe00u, 0xfa01f7f9u }, // block 20 (dxil label 883)
    { 0xf7feee08u, 0xfa10ff02u, 0xfb0b02feu, 0x07ecf4fau, 0x0204fc03u, 0xf3edf6f8u, 0x03fa0407u, 0xfd05fd04u, 0xf706070eu, 0x051e050bu, 0xfcf2f7f4u, 0x15080b04u, 0xfaf909f2u, 0x01f5fff9u, 0x00f004fdu, 0xfa060af8u }, // block 21 (dxil label 915)
    { 0xd8f4dd02u, 0x2a4bd916u, 0xec2d23d3u, 0x3f050ad8u, 0x1bef05f9u, 0x1013e7f5u, 0x1d131afbu, 0xe2b0f80cu, 0x2421ba22u, 0x2615f626u, 0x381af42bu, 0x1dfddeefu, 0xe8470b0fu, 0xea2fe1feu, 0x0542ffc8u, 0x1805b7e2u }, // block 22 (dxil label 947)
    { 0xf9fcf912u, 0xee1c0705u, 0xf70efcf5u, 0x0e01ed0au, 0x08fd0c15u, 0xf8120600u, 0xfffdff10u, 0x0915fa0cu, 0x0409fafcu, 0x09f404fau, 0x07f8faf9u, 0xf6fcf7f6u, 0x000304f5u, 0x1cf200fcu, 0x04fd0c03u, 0xe9030afdu }, // block 23 (dxil label 979)
    { 0xfffdf8fau, 0xf8f7fef8u, 0xfcfeff03u, 0xfd0df9f7u, 0xfffdfff9u, 0xf9f7fffeu, 0xffff0003u, 0x0005f800u, 0xff030600u, 0x020204ffu, 0xfe01f7ffu, 0x0002fdfcu, 0x00020402u, 0x0e060402u, 0x02030401u, 0xfbfff6f8u }, // block 24 (dxil label 1011)
    { 0x02fff8f8u, 0xfdf0fa0bu, 0xee130e02u, 0x03f40210u, 0xfe030c0du, 0x04010506u, 0x08fd0016u, 0x02060404u, 0xfff5f8fcu, 0x03ed06f7u, 0x02fe0200u, 0xfd01fffbu, 0x00fa0100u, 0x010e04f9u, 0x08f503fdu, 0xfb06f602u }, // block 25 (dxil label 1043)
    { 0xfc0b0702u, 0x07f90303u, 0x0102fafau, 0xfb08f505u, 0x010308ffu, 0x0dfd0703u, 0x070d0501u, 0xf5fffdfeu, 0x0005f70au, 0x02070500u, 0x03030207u, 0xf5fef1f9u, 0x0202f404u, 0x0001fcfeu, 0x01030801u, 0xfd0207ffu }, // block 26 (dxil label 1075)
    { 0x000c0a03u, 0x010a0200u, 0x030ff5f9u, 0x0002f7fau, 0x00060203u, 0xfd03fbfeu, 0xfd04f8fdu, 0xfefe0800u, 0xfc0706fdu, 0x09fafa0du, 0xff0106fau, 0xfb03f704u, 0xfef902f9u, 0x0a040904u, 0x00fb0404u, 0x02fc0208u }, // block 27 (dxil label 1170)
    { 0x021605fcu, 0xfaf901fdu, 0x0005f002u, 0x0005fdf9u, 0x030efff5u, 0xfceffe05u, 0xf90cf8fcu, 0xfc0109fdu, 0xfdfffdfau, 0xf606fe08u, 0x04f403fcu, 0x061dfffeu, 0xfd0efe0au, 0x02330c02u, 0xf2ff060du, 0x03fbf6f8u }, // block 28 (dxil label 1202)
    { 0xff020505u, 0x010405fcu, 0x0402f701u, 0x000700f9u, 0xfefefffcu, 0x0200fffeu, 0xff020504u, 0x03000a00u, 0xff0e0601u, 0x0a06fe06u, 0xff0102feu, 0xfa070101u, 0x03fa0905u, 0x00050400u, 0xfff8070au, 0x02fdfefdu }, // block 29 (dxil label 1234)
    { 0x010001fcu, 0xf8f106fau, 0x08faf30fu, 0xfe11fdf5u, 0x05f7f7f2u, 0xf5f400fau, 0x0afb0110u, 0xf9fa00feu, 0xf60a0202u, 0xf60cfd0cu, 0xf8ffffe7u, 0x0b160bf9u, 0x040306fdu, 0x04ff10fcu, 0xf6fd011bu, 0xfc07f9f3u }, // block 30 (dxil label 1266)
    { 0xf908f615u, 0xfdfcfb06u, 0xde040ddau, 0xff0e1648u, 0xf2459d12u, 0x739f0628u, 0x2416f7ebu, 0xf04dd190u, 0xf908e0e6u, 0xeac231f2u, 0xeb0ee7f0u, 0xe1140322u, 0xffd2c2f5u, 0xb612f9f4u, 0xfeff32e6u, 0xf3d72a2fu }, // block 31 (dxil label 1298)
    { 0x09fd0cf6u, 0xfdfcfc03u, 0xfe02fd0au, 0x010b0606u, 0x07f207f7u, 0xfa0bf4ffu, 0x00fe0207u, 0x08ff0608u, 0x0009f7fdu, 0xfef30002u, 0x06f70107u, 0xf805fffau, 0x02fefcf7u, 0x0e09fdffu, 0xfbf10c19u, 0xfffb01efu }, // block 32 (dxil label 1330)
    { 0x000700ffu, 0x03010001u, 0x0101f9ffu, 0x0202fefcu, 0xfefffb02u, 0x01fe0000u, 0xfe010607u, 0x03000400u, 0xf8070a02u, 0x0d060609u, 0x000506fau, 0xfd08fb00u, 0xfef6ff04u, 0x080e0107u, 0x03f80f0cu, 0x04f701fdu }, // block 33 (dxil label 1362)
    { 0xff0a07f8u, 0xfcf6fe05u, 0x00f6f803u, 0x04080700u, 0x010100fau, 0xf806f702u, 0x05f2f6fdu, 0x0bff0508u, 0x05f2f602u, 0xfa0504fcu, 0x0bef070du, 0x000afa01u, 0xfc000403u, 0xf61703f7u, 0xf7f20209u, 0xff01f3f0u }, // block 34 (dxil label 1394)
    { 0x04fe00fbu, 0x10050007u, 0x0302fe03u, 0xfffafffau, 0x03f905fdu, 0x0cfff701u, 0xfe01fbfdu, 0x07f20901u, 0xf6fbf40fu, 0xf91a05fdu, 0x07030b03u, 0xf709fb01u, 0x03fa01fbu, 0x09edfb05u, 0xfef2f908u, 0x0d040901u }, // block 35 (dxil label 1426)
};

static const uint W2[8][16] = {
    { 0xc8f2d5fbu, 0xfe00e63eu, 0x233b2d17u, 0x0f070ccfu, 0xfde616e4u, 0xe8122eb1u, 0x1f1a1efcu, 0x04efe30fu, 0x19196000u, 0x0ce4f744u, 0x3417e830u, 0xebc26bc7u, 0x00ece9f4u, 0x06faddcbu, 0xfce6ff1eu, 0x05fd04dcu }, // block 0 (dxil label 1721)
    { 0xff3324fbu, 0x00f52f0bu, 0xe008fd06u, 0x2bfe2238u, 0x0de6142cu, 0x20f313dau, 0x2eba0d16u, 0x05d8ccfau, 0x2ed113d1u, 0xe910e809u, 0xfa210ee9u, 0x261033e4u, 0xeef3ef14u, 0x022b20e1u, 0xd4c80ce4u, 0x0615e108u }, // block 1 (dxil label 1757)
    { 0xe540fcc5u, 0xf0f10e29u, 0xfad9ddf3u, 0xe8cfcfcdu, 0xf5cb03fcu, 0xf6114ff5u, 0x09dc4ecau, 0x29cabfccu, 0xf92b02e7u, 0x0acdb8e9u, 0xcb1c2806u, 0x10f2ee07u, 0x0415f4f5u, 0x36e3fa3cu, 0xfefc12c7u, 0x21fc0007u }, // block 2 (dxil label 1790)
    { 0x13f2ec19u, 0x0b21fb18u, 0x0a0be718u, 0x070ae948u, 0xdb1df030u, 0x2127ee06u, 0xcb23f4dfu, 0xf6de18f8u, 0xdd19fff4u, 0xf6ee27e1u, 0xed0ab124u, 0xf5df0cfeu, 0x06f8fce9u, 0xf037130cu, 0x0d1bf8eau, 0xb8e2e7f3u }, // block 3 (dxil label 1823)
    { 0xacf6e40cu, 0x03151d05u, 0x1dfe064bu, 0x1a1749bfu, 0x0147dbe4u, 0x131614c6u, 0x1a25210du, 0x1dc7f62eu, 0xc21618f3u, 0x0506e1ddu, 0x01fffd01u, 0xb7ecf8ffu, 0xe2201dd3u, 0xd0e82cbfu, 0x04160c16u, 0xf40f0de9u }, // block 4 (dxil label 1856)
    { 0x1df01626u, 0x17d6fdcbu, 0x22ba0e19u, 0x06d5c8fbu, 0x3b052008u, 0xf7fff5ebu, 0xebe1eacfu, 0xe90bcf20u, 0x14120be8u, 0xe9f4e404u, 0xd9f4d3edu, 0xda0627e7u, 0x0004f6bbu, 0xe2fd0b14u, 0x08e9c734u, 0x0707e048u }, // block 5 (dxil label 1889)
    { 0xf9fdfdebu, 0xe9f0ebd9u, 0x3eeffafcu, 0xfefe0746u, 0x111d021cu, 0xf41b191fu, 0x21e6ecfau, 0xc91e2c1eu, 0x00000ffau, 0x03fe072au, 0x411cf117u, 0xe8e61dfcu, 0x1104ede6u, 0x0ad60b30u, 0xdde44e11u, 0x11eb2041u }, // block 6 (dxil label 1922)
    { 0x07211707u, 0x04d81439u, 0xd8db0a24u, 0x4437f100u, 0x11156806u, 0x14effb4fu, 0x4925da3au, 0xeac376bdu, 0x231befe0u, 0xee1cedffu, 0x2adf2e1au, 0x2611013cu, 0xf416ee3fu, 0xddede407u, 0x030742f8u, 0xffca1118u }, // block 7 (dxil label 1955)
};

static const uint W3[8][16] = {
    { 0xeb242a21u, 0x3102120cu, 0x15fd2523u, 0xe5f804f7u, 0xf9d41a01u, 0x1b14e005u, 0xc3f9cfeeu, 0xb949fedcu, 0x160710edu, 0x110747dcu, 0x1b260af3u, 0xf91af1deu, 0xcca6c8c2u, 0xea14e9d8u, 0x00d0b1ffu, 0xe2b2ef38u }, // block 0 (dxil label 2253)
    { 0x1cf21a1bu, 0x4824dfedu, 0xecf908dfu, 0xcb03e712u, 0x13030b0cu, 0xda052232u, 0xf4cf0af8u, 0x0e152aceu, 0x29fc012cu, 0x00fbf5c6u, 0x17d0e929u, 0x100704c5u, 0x132607c7u, 0x00282102u, 0x19050e05u, 0xec1d52bdu }, // block 1 (dxil label 2288)
    { 0xd318b2f7u, 0x38010333u, 0xfbf5f30cu, 0x0df1eb48u, 0xf1981aefu, 0xe50d1df1u, 0x1508fa1cu, 0xf8ecff11u, 0xe1e30ff1u, 0x15182dfbu, 0xf900f9f1u, 0x0217120cu, 0x11300313u, 0xf8ff1319u, 0x19f81ef7u, 0x1519f8f4u }, // block 2 (dxil label 2370)
    { 0x00f7b8d6u, 0x2706d5bau, 0xb33601cbu, 0x0203d31cu, 0x16160cecu, 0x2cfd1cd2u, 0x1dfaef0bu, 0xf5fb77f7u, 0xfaf8f30fu, 0x02dc07e6u, 0xf0dffe22u, 0x1f1c24d9u, 0x1ee726e8u, 0xec221301u, 0xf106f30eu, 0x21f1e0ecu }, // block 3 (dxil label 2402)
    { 0xf32827f2u, 0x181237ebu, 0xfd471affu, 0xee2ccb06u, 0x1301021bu, 0x01ee0238u, 0xff1510d4u, 0x092cf8ceu, 0x1e1810f2u, 0x060412d9u, 0xef010408u, 0xdb0216f9u, 0xd1281df3u, 0xf93437ddu, 0x15e307c1u, 0xc60cf1ffu }, // block 4 (dxil label 2436)
    { 0xfc1fda53u, 0xead8d9f5u, 0x130fd2e8u, 0x29c4d520u, 0xede1fe04u, 0xf3fcf3e0u, 0xd60f17e3u, 0xc83be8dau, 0x40290102u, 0x0a000df4u, 0x31d50316u, 0xf4ddfa11u, 0x204ae904u, 0xcb1bc4dau, 0x19e33724u, 0xf2eddbbeu }, // block 5 (dxil label 2468)
    { 0x17110915u, 0xe4063825u, 0xeef72ad3u, 0xf8121ce2u, 0x08cbea15u, 0x0101d70fu, 0x3cf805e3u, 0x0af3d9f4u, 0x0707220du, 0x241017e2u, 0xeb040d11u, 0xe0fbf803u, 0xfa0b1bedu, 0x1e0abed4u, 0x2b150d21u, 0xfed01c18u }, // block 6 (dxil label 2502)
    { 0x0efd0904u, 0xea0503cbu, 0x04fe050cu, 0xfd0a0713u, 0x0423e6f0u, 0x032ddff9u, 0xf4291737u, 0xe9041edcu, 0x09fa0000u, 0x0a1cf5c8u, 0x14f6fdffu, 0x4201f714u, 0x181cf72bu, 0x48d8ed44u, 0xfcbeec47u, 0xdcf40207u }, // block 7 (dxil label 2534)
};

// window offsets (dy, dxb) in block order: rows ty..ty+2, cols tx..tx+2
static const uint2 OFF[9] = {
    {0, 0}, {0, 16}, {0, 32},
    {1, 0}, {1, 16}, {1, 32},
    {2, 0}, {2, 16}, {2, 32}
};

int requant_shift(int x, uint shift, int add)
{
    uint bit = (uint(x) >> shift) & 1u;
    return (x + add + int(bit)) >> int(shift); // arithmetic shift
}

uint pack_bytes(int4 q)
{
    // SM 6.6 intrinsic -> DXIL Pack4x8 packMode 2 (SClamp), identical to the
    // 4.1.1 container's requant packs. (A hand-written clamp+byte-compose gets
    // pattern-matched by dxc into a TRUNCATING pack4x8 mode 0, which is what
    // initially caused 87-byte residuals here.)
    return (uint)pack_clamp_s8(q);
}

int int16_wrap(int v)
{
    return ((v & 0xFFFF) ^ 0x8000) - 0x8000;
}

[numthreads(64, 1, 1)]
void reimpl_pass12_2160(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint tz = tid.z;
    if (tx > 1919u || ty > 1079u)
        return;
#if defined(DIAG) || defined(DIAG3) || defined(DIAG4)
    if (ty >= 90u)
        return;
#endif

    // ---- stage 1: 3x3 depthwise conv 16ch -> 16ch, bias0 = t18@125312 ----
    int A8[16];
    [unroll]
    for (int w = 0; w < 4; w++) {
        const uint4 v = t18_initializer.Load4(125312u + (tz << 6) + 16u * w);
        A8[4 * w + 0] = asint(v.x);
        A8[4 * w + 1] = asint(v.y);
        A8[4 * w + 2] = asint(v.z);
        A8[4 * w + 3] = asint(v.w);
    }

    uint4 rowD[9]; // per offset: 4 packed dwords = 16 int8 channels
    [unroll]
    for (int o = 0; o < 9; o++) {
        const uint addr = 49958592u + 30752u * (ty + OFF[o].x) + 16u * tx + OFF[o].y;
        rowD[o] = ScratchBuffer.Load4(addr);
    }

    [unroll]
    for (int g = 0; g < 4; g++) {
        [unroll]
        for (int o2 = 0; o2 < 9; o2++) {
            [unroll]
            for (int k = 0; k < 4; k++) {
                int acc = A8[4 * g + k];
                [unroll]
                for (int i2 = 0; i2 < 4; i2++)
                    acc = dot4add_i8packed(rowD[o2][i2], W1[g * 9 + o2][4 * k + i2], acc);
                A8[4 * g + k] = acc;
            }
        }
    }

#ifdef DIAG3
    if ((tx == 252u && ty == 32u) || (tx == 267u && ty == 41u)) {
        const uint dbase = 208u * (ty * 960u + tx);
        [unroll]
        for (int q = 0; q < 9; q++)
            ScratchBuffer.Store4(dbase + 16u * q, rowD[q]);
        ScratchBuffer.Store4(dbase + 144u, uint4(asuint(A8[0]), asuint(A8[1]), asuint(A8[2]), asuint(A8[3])));
        ScratchBuffer.Store4(dbase + 160u, uint4(asuint(A8[4]), asuint(A8[5]), asuint(A8[6]), asuint(A8[7])));
        ScratchBuffer.Store4(dbase + 176u, uint4(asuint(A8[8]), asuint(A8[9]), asuint(A8[10]), asuint(A8[11])));
        ScratchBuffer.Store4(dbase + 192u, uint4(asuint(A8[12]), asuint(A8[13]), asuint(A8[14]), asuint(A8[15])));
        return;
    }
#endif
    // ---- requant 1: round-shift >>8, pack s8 -> conv_result[4] ----
    uint convres[4];
    [unroll]
    for (int f = 0; f < 16; f += 4) {
        int4 q;
        [unroll]
        for (int j = 0; j < 4; j++)
            q[j] = requant_shift(A8[f + j], 7u, 63);
        convres[f / 4] = pack_bytes(q);
    }

#ifdef DIAG4
    if ((tx == 252u && ty == 32u) || (tx == 267u && ty == 41u)) {
        const uint dbase = 208u * (ty * 960u + tx);
        ScratchBuffer.Store4(dbase + 208u, uint4(convres[0], convres[1], convres[2], convres[3]));
    }
#endif
    // ---- stage 2: pointwise 16 -> 32 + ReLU, bias1 = t18@125952 ----
    int A10[32];
    [unroll]
    for (int w2 = 0; w2 < 8; w2++) {
        const uint4 v = t18_initializer.Load4(125952u + 16u * w2);
        A10[4 * w2 + 0] = asint(v.x);
        A10[4 * w2 + 1] = asint(v.y);
        A10[4 * w2 + 2] = asint(v.z);
        A10[4 * w2 + 3] = asint(v.w);
    }

    [unroll]
    for (int b = 0; b < 8; b++) {
        [unroll]
        for (int k2 = 0; k2 < 4; k2++) {
            int acc2 = A10[4 * b + k2];
            [unroll]
            for (int i3 = 0; i3 < 4; i3++)
                acc2 = dot4add_i8packed(convres[i3], W2[b][4 * k2 + i3], acc2);
            A10[4 * b + k2] = acc2;
        }
    }

    // ---- requant 2: ReLU + round-shift >>7, pack s8 -> relu_result[8] ----
    uint relu_d[8];
    [unroll]
    for (int f2 = 0; f2 < 32; f2 += 4) {
        int4 q2;
        [unroll]
        for (int j2 = 0; j2 < 4; j2++)
            q2[j2] = requant_shift(max(A10[f2 + j2], 0), 8u, 127);
        relu_d[f2 / 4] = pack_bytes(q2);
    }

    // ---- stage 3: pointwise 32 -> 16, bias2 = t18@126592, raw int32 acc ----
    int A12[16];
    [unroll]
    for (int w3 = 0; w3 < 4; w3++) {
        const uint4 v = t18_initializer.Load4(126592u + 16u * w3);
        A12[4 * w3 + 0] = asint(v.x);
        A12[4 * w3 + 1] = asint(v.y);
        A12[4 * w3 + 2] = asint(v.z);
        A12[4 * w3 + 3] = asint(v.w);
    }

    [unroll]
    for (int p = 0; p < 4; p++) {
        [unroll]
        for (int h = 0; h < 2; h++) {
            [unroll]
            for (int k3 = 0; k3 < 4; k3++) {
                int acc3 = A12[4 * p + k3];
                [unroll]
                for (int i4 = 0; i4 < 4; i4++)
                    acc3 = dot4add_i8packed(relu_d[4 * h + i4], W3[2 * p + h][4 * k3 + i4], acc3);
                A12[4 * p + k3] = acc3;
            }
        }
    }

#ifdef DIAG4
    if ((tx == 252u && ty == 32u) || (tx == 267u && ty == 41u)) {
        const uint dbase = 208u * (ty * 960u + tx);
        ScratchBuffer.Store4(dbase + 224u, uint4(relu_d[0], relu_d[1], relu_d[2], relu_d[3]));
        ScratchBuffer.Store4(dbase + 240u, uint4(relu_d[4], relu_d[5], relu_d[6], relu_d[7]));
        ScratchBuffer.Store4(dbase + 256u, uint4(asuint(A12[0]), asuint(A12[1]), asuint(A12[2]), asuint(A12[3])));
        ScratchBuffer.Store4(dbase + 272u, uint4(asuint(A12[4]), asuint(A12[5]), asuint(A12[6]), asuint(A12[7])));
        ScratchBuffer.Store4(dbase + 288u, uint4(asuint(A12[8]), asuint(A12[9]), asuint(A12[10]), asuint(A12[11])));
        ScratchBuffer.Store4(dbase + 304u, uint4(asuint(A12[12]), asuint(A12[13]), asuint(A12[14]), asuint(A12[15])));
        return;
    }
#endif
    // ---- tail: residual + float requant, pack s8, store 4 dwords ----
    const uint centerAddr = 49958592u + 30752u * (ty + 1u) + 16u * tx + 16u;
    const uint4 center = ScratchBuffer.Load4(centerAddr);
#ifdef DIAG2
    if ((tx == 252u && ty == 32u) || (tx == 267u && ty == 41u)) {
        const uint dbase = 192u * (ty * 960u + tx);
        ScratchBuffer.Store4(dbase, uint4(asuint(A8[0]), asuint(A8[1]), asuint(A8[2]), asuint(A8[3])));
        ScratchBuffer.Store4(dbase + 16u, uint4(asuint(A8[4]), asuint(A8[5]), asuint(A8[6]), asuint(A8[7])));
        ScratchBuffer.Store4(dbase + 32u, uint4(asuint(A8[8]), asuint(A8[9]), asuint(A8[10]), asuint(A8[11])));
        ScratchBuffer.Store4(dbase + 48u, uint4(asuint(A8[12]), asuint(A8[13]), asuint(A8[14]), asuint(A8[15])));
        ScratchBuffer.Store4(dbase + 64u, uint4(convres[0], convres[1], convres[2], convres[3]));
        ScratchBuffer.Store4(dbase + 80u, uint4(relu_d[0], relu_d[1], relu_d[2], relu_d[3]));
        ScratchBuffer.Store4(dbase + 96u, uint4(relu_d[4], relu_d[5], relu_d[6], relu_d[7]));
        ScratchBuffer.Store4(dbase + 112u, uint4(asuint(A12[0]), asuint(A12[1]), asuint(A12[2]), asuint(A12[3])));
        ScratchBuffer.Store4(dbase + 128u, uint4(asuint(A12[4]), asuint(A12[5]), asuint(A12[6]), asuint(A12[7])));
        ScratchBuffer.Store4(dbase + 144u, center);
        return;
    }
#endif
#ifdef DIAG
    if (ty < 90u) {
        const uint dbase = 96u * (ty * 960u + tx);
        ScratchBuffer.Store4(dbase, uint4(asuint(A12[0]), asuint(A12[1]), asuint(A12[2]), asuint(A12[3])));
        ScratchBuffer.Store4(dbase + 16u, uint4(asuint(A12[4]), asuint(A12[5]), asuint(A12[6]), asuint(A12[7])));
        ScratchBuffer.Store4(dbase + 32u, uint4(asuint(A12[8]), asuint(A12[9]), asuint(A12[10]), asuint(A12[11])));
        ScratchBuffer.Store4(dbase + 48u, uint4(asuint(A12[12]), asuint(A12[13]), asuint(A12[14]), asuint(A12[15])));
        ScratchBuffer.Store4(dbase + 64u, center);
        uint4 cr = uint4(convres[0], convres[1], convres[2], convres[3]);
        ScratchBuffer.Store4(dbase + 80u, cr);
        return;
    }
#endif
    const bool inRange = tx < ml2c_cb[13].x && ty < ml2c_cb[13].y;
    const uint outRow = 0u + 30752u * (ty + 1u) + 16u * (tx + 1u);

    [unroll]
    for (int d = 0; d < 4; d++) {
        int4 lanes;
        [unroll]
        for (int j3 = 0; j3 < 4; j3++) {
            const int cbyte = (int)(((center[d] >> (8 * j3)) & 0xFFu) << 24) >> 24;
            const float t1 = float(cbyte) * 0.03125f;
            const float t2 = float(A12[4 * d + j3]) * 0.00048828125f;
            const float r = round((t1 + t2) * 16.0f);
            lanes[j3] = int16_wrap(int(r));
        }
        const uint val = inRange ? pack_bytes(lanes) : 0u;
        const uint addr = inRange ? (outRow + 4u * d) : 134217728u;
        ScratchBuffer.Store(addr, val);
    }
}
