// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// 2160-class variant of reimpl_pass1.hlsl (mechanical 2x-spatial transform, no other changes).
// 1080 reference container: analysis/extracted/4.1.1-official/0078_002cce40_38eaa0412dcd046b.dxbc (1080, guards 959/539, stride 15392, write base 8342464)
// 2160 AMD container (ground truth): analysis/extracted/4.1.1-official/0273_007de1d0_bb2d3f210a38e89d.dxbc (2160, guards 3839/2159, stride 61472, write base 132902464)
// Verified transform (full-DXIL diff, identical line counts, immediates-only deltas):
//   guards x2, inRange x2, full-res stride 15392->61472, half-res stride 7712->15392,
//   plane spacing 2097664->8342464, bases relocated per container (see 2160 line above).
// Weights / biases / requant / tail / loop structure: byte-identical to the 1080 file.
// ============================================================================
// 4.1.1 pass1 (Global 5096, container 38eaa041...) reimplementation (our_reimplementation)
// ConvNextBlock: 3x3 dw conv 16->16 -> requant -> pw 16->32 + ReLU -> requant
//                -> pw 32->16 -> residual + float requant -> store.
//
// Semantics reconstructed from the 4.0.2 ML2Code ConvNextBlock topology
// (ml2code_runtime/operators/int8_NHWC/Fused/ConvNextBlock.hlsli) and the
// 4.1.1 container DXIL (pass1 (Global 5096, container 38eaa041...)). Weight tables are extracted from the
// container's compile-time constants; biases are read from the captured
// runtime initializer (t18): bias0 i32[16] @3456, bias1 i32[32] @4096,
// bias2 i32[16] @4736. The semantics were validated byte-exact against the
// AMD oracle by bench/tools/pass2_sim.py before this HLSL was authored.
// Plane roles: reads NHWC plane at scratch base 0, writes the plane
// at 8342464; requant1 (>>7,+63), requant2 (>>8,+127),
// tail scale x64; guard cbuffer reg2.
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

cbuffer MLSR_Optimized_Constants : register(b0)
{
    uint4 ml2c_cb[17]; // reg3.xy = logical tensor width/height
}

static const uint W1[36][16] = {
    { 0xfdfb01fbu, 0x1003ff00u, 0x020000fcu, 0xf0fef109u, 0xf9f8fe04u, 0x1002fffdu, 0xfbf91200u, 0xf603f80eu, 0xfe000107u, 0xeb01fefcu, 0x0ffefe01u, 0x0c00f912u, 0xfbff06fdu, 0x1004fe07u, 0xf5f60ef1u, 0xf90afafdu }, // block 0 (dxil label 93)
    { 0xfae8f310u, 0xfb11ef04u, 0xf00aff22u, 0xf508daf9u, 0xf7f7ff1eu, 0x06fcf507u, 0xff0bf80fu, 0xeef4d211u, 0x09fff9f2u, 0x0003fe06u, 0xf40a0c0fu, 0x03f01aecu, 0x0bf6f70au, 0xfbf4feebu, 0x0b0bee1bu, 0xfcf7e505u }, // block 1 (dxil label 128)
    { 0x07fdf7ebu, 0xfc17fd03u, 0xf901fe0bu, 0x0d0b06f5u, 0xfcfdf9f6u, 0x0d0cf802u, 0xf8f903feu, 0xf90401f8u, 0x01ff050cu, 0xf5e7f904u, 0x011c0010u, 0x0bfff205u, 0x03f7fefdu, 0x08f7fe08u, 0xf40603ffu, 0x09f8faf8u }, // block 2 (dxil label 160)
    { 0x00090be5u, 0xdc25010au, 0x05f910f2u, 0xfb0707f7u, 0x0b1efef7u, 0xee1e10fcu, 0xeff9fc01u, 0x181a0dfcu, 0xf50d0ffeu, 0x16fcf50fu, 0x00fe11e0u, 0xd6f6e20cu, 0x0e06f5f7u, 0xe51306f6u, 0x020df8fau, 0x0602f707u }, // block 3 (dxil label 192)
    { 0x1708474fu, 0xd342edbdu, 0xf1d157f4u, 0x33d4f8bau, 0xda28d32bu, 0x23afd81fu, 0x1ff523f6u, 0x3bdfe1ecu, 0x23cb91feu, 0xef0cf9e9u, 0xf73c2dbau, 0xe1074d05u, 0xf207dbe8u, 0xe5073a16u, 0x13db172au, 0xbeeef751u }, // block 4 (dxil label 224)
    { 0xf0f0f5f0u, 0xe814f9fbu, 0xf5060d01u, 0xfcf2efffu, 0xfdfbf8fau, 0xe30d02eau, 0x17f8f0fcu, 0xf701ef13u, 0xfcfbffecu, 0x160a0703u, 0xf5190af1u, 0x0afdfefcu, 0x020a1301u, 0x1a06f8fcu, 0xee10120fu, 0x0b0221f1u }, // block 5 (dxil label 256)
    { 0xfffefaf6u, 0xfd05ff00u, 0x04fef706u, 0x03060306u, 0xfd0505feu, 0x00ff0008u, 0xfd0706f6u, 0xf9fa06fdu, 0x0414f4f6u, 0xe80605f5u, 0xfa05fb0fu, 0x1c1106fbu, 0x020301f6u, 0x020004ffu, 0xf9f0fa09u, 0x0503fff4u }, // block 6 (dxil label 288)
    { 0xfcf60c08u, 0xf1f10008u, 0x080f1002u, 0xf8f20404u, 0x0e01f0f4u, 0xf517f4e8u, 0xfefa01f9u, 0xf101f019u, 0xef132406u, 0x1b010515u, 0x02e4fa06u, 0x0000e9f3u, 0xfdfb0c07u, 0xfceb040cu, 0x03180c03u, 0x0feb0bfbu }, // block 7 (dxil label 320)
    { 0xfefffffdu, 0x060009fdu, 0x0109fd00u, 0xff00ff00u, 0xfcfa04ffu, 0x06fd030bu, 0x00ff1500u, 0x08fe0706u, 0xfc08fd02u, 0xec0603f7u, 0x02fd0102u, 0x0d0cfe03u, 0x01040407u, 0xfdf9fd00u, 0x09fd02ffu, 0xfc00fd0du }, // block 8 (dxil label 352)
    { 0x03fb0b06u, 0xfdfe000au, 0x000f03fdu, 0xff0700ffu, 0x02040105u, 0x04f80506u, 0xf901fef8u, 0xf9fefaf7u, 0xfafe0805u, 0xefec0405u, 0x09050beau, 0x02fff206u, 0x0cf6fcffu, 0xf6f20201u, 0x040405feu, 0x0dfc0affu }, // block 9 (dxil label 459)
    { 0xfd05e7d6u, 0xdf02fde9u, 0x17eeee0eu, 0x12e514f5u, 0xfc000129u, 0x11f100f8u, 0x000c030eu, 0x000611fcu, 0x0504fcf6u, 0xfaf100f3u, 0x140ee405u, 0xfaeded13u, 0xff0efd07u, 0x000d0ff0u, 0x0aee04fcu, 0x0dff2d01u }, // block 10 (dxil label 491)
    { 0x04fa09f6u, 0xfff80708u, 0xf90f0afcu, 0x0400fd07u, 0x00f8f6ffu, 0xf2ff0901u, 0xfb0b08f0u, 0x0bfc03f8u, 0xfbfc0319u, 0x05db0404u, 0x050c01f2u, 0xfefbe8feu, 0x03f6fdf8u, 0x01ef0100u, 0xfc1e0cd9u, 0x00ec0908u }, // block 11 (dxil label 523)
    { 0x09fc0306u, 0x04f0fcfbu, 0x15190608u, 0xe5fbf410u, 0xfaf80116u, 0x08fafb04u, 0x05fa000eu, 0xfff407f3u, 0xfc0800f8u, 0x0ef5fdf2u, 0x010aef00u, 0xfef6f706u, 0x0ae8f128u, 0x06fff205u, 0x03120611u, 0xfa0d0200u }, // block 12 (dxil label 555)
    { 0xd6ffdf5cu, 0x0fe41ef8u, 0x11cefef9u, 0xf4e1fae1u, 0xf41fe9e3u, 0x2c1bbc03u, 0x4b194000u, 0xdf29d7cfu, 0xe5c638e1u, 0x0c43f007u, 0xf95f6716u, 0xeec35b20u, 0xccef20fdu, 0x211bd2bfu, 0xeec9b2edu, 0x3acca617u }, // block 13 (dxil label 587)
    { 0x05fcf4eau, 0x0e07010bu, 0xf804fe09u, 0x0e00fcf1u, 0x01e6e309u, 0xf2f8000cu, 0xeb1c04e6u, 0x10d9f4f6u, 0x0aede801u, 0xdefe14fdu, 0x09e1e4dau, 0x13ede1f5u, 0x07f2fc0cu, 0x17f60209u, 0xf9240c08u, 0x09f413ffu }, // block 14 (dxil label 619)
    { 0xfafb01f6u, 0x060809ffu, 0xfbfeff0fu, 0x0b070bfau, 0xfefdfd0au, 0x04fdfbffu, 0xfff102fcu, 0xfe0000fdu, 0x0009fd06u, 0x00fd0403u, 0xf9f80af6u, 0x05fd07f2u, 0x01ff0102u, 0xf6f5feffu, 0x0b0bf20eu, 0xfe01f602u }, // block 15 (dxil label 651)
    { 0x10000aeau, 0x0104071bu, 0x07090af6u, 0x0d100af5u, 0x14020000u, 0xf50e120du, 0x0bdefafdu, 0x0619eaeau, 0xfd0e0e0bu, 0x0b061103u, 0xfee6f005u, 0x0d01f1f1u, 0x050808f0u, 0x01160b15u, 0x01060606u, 0x041afc0bu }, // block 16 (dxil label 683)
    { 0xfd08fe0du, 0xf001fdf9u, 0x0a03fb00u, 0xef09020cu, 0xf70d0300u, 0x010a03e8u, 0xfb01f4fdu, 0xf309f9fdu, 0xfe010303u, 0xf7fcf9fcu, 0x0a0703f3u, 0x000cf316u, 0xfc041609u, 0x0df50bf6u, 0x0afa03f4u, 0xf2fdf504u }, // block 17 (dxil label 715)
    { 0x04fb07fcu, 0xfb010308u, 0x0504f9ffu, 0x040204f0u, 0xfafe0cf3u, 0x0605fe0bu, 0x00fa06f4u, 0xf605f0fdu, 0x000002ffu, 0x02f40affu, 0x020401f1u, 0xf1fde713u, 0xf6020a0du, 0x0a06fb0bu, 0xf4f707fau, 0xf507e8f9u }, // block 18 (dxil label 810)
    { 0xf70afaf8u, 0xff12fbf5u, 0x0cf70b0cu, 0xf1020f0cu, 0x03f6eef8u, 0x0103f4f5u, 0x0a10e419u, 0xfa06e812u, 0xfdf4f715u, 0x0f08f503u, 0xeb08eb23u, 0x010adaefu, 0xf804fc04u, 0x0df5f5f2u, 0x0807fc1fu, 0xfff11affu }, // block 19 (dxil label 842)
    { 0xfffffde5u, 0xf9110efcu, 0xf7090ae5u, 0x0afd08f4u, 0x04f9fffdu, 0xf800010au, 0xfb00fc05u, 0x0e09f4f2u, 0x0ef9010bu, 0xf0e60108u, 0x081f0008u, 0x0204f105u, 0xf8f70018u, 0x07ff000au, 0xedfb0211u, 0x0706e7f8u }, // block 20 (dxil label 874)
    { 0x02eef60fu, 0xfaf8ebfdu, 0x1315062du, 0xfc020d07u, 0xff01f5e7u, 0xfe0df1ffu, 0x0508eff5u, 0xfd06010du, 0x021d12f3u, 0xfb080d09u, 0xfef4fbd9u, 0xf6fff102u, 0xea13090cu, 0x1ceefbf9u, 0x0012fa0au, 0xf8ef0d0fu }, // block 21 (dxil label 906)
    { 0x55050022u, 0x04ff371bu, 0xcd0c1828u, 0xc6f22a43u, 0xf5100109u, 0x65b7d30bu, 0x0204f5e8u, 0xe52b2e17u, 0xf9ea1826u, 0x64100dfau, 0xeefabf11u, 0xff0d455eu, 0xfaee1ebbu, 0x2015c5c8u, 0x3f31d405u, 0xf203d22au }, // block 22 (dxil label 938)
    { 0xfdebe6edu, 0xdcfbfbfcu, 0x1019fbefu, 0x01e6f60cu, 0xf7f4edffu, 0x010bf8fcu, 0xf5f5f3fau, 0x0cffe9fdu, 0xfdfbf5e5u, 0x1318fdffu, 0xee0304fbu, 0x02fd07fbu, 0xfbfde609u, 0x080f0d09u, 0xe4ccf4f2u, 0x13f5eee7u }, // block 23 (dxil label 970)
    { 0xfcf6020fu, 0x0bf0fd01u, 0x0f04fd09u, 0xebfff602u, 0x02fbf9fbu, 0xfe0001fcu, 0xfeff0203u, 0x04fd0bf8u, 0xff09faf0u, 0xfb120302u, 0xf5ef01fcu, 0x0c030bf7u, 0xfa03feffu, 0x0a130903u, 0xe9e812feu, 0x11061cecu }, // block 24 (dxil label 1002)
    { 0xec060afau, 0x020fff09u, 0xfd06fe18u, 0x0f0805fau, 0xf1041907u, 0x10f50817u, 0x03fcf80cu, 0x1af80ef4u, 0x08010306u, 0xf6f40708u, 0x10f0fcfbu, 0xf607f405u, 0x1cfd020eu, 0xfef50f0bu, 0x0cf5f5efu, 0x0907f3f6u }, // block 25 (dxil label 1034)
    { 0xfc070b01u, 0x0cf513fau, 0x05fff7f9u, 0xee05eaf7u, 0xfe0402ffu, 0xfd0101f9u, 0x0709fdffu, 0xf40cfa06u, 0xfe02fffcu, 0xf90806fau, 0xfefdfffdu, 0x05fe03fbu, 0xfb06f507u, 0xef0ef0f8u, 0xff120406u, 0xf50d0e15u }, // block 26 (dxil label 1066)
    { 0xfdfff4fbu, 0xf410fbf4u, 0x00f0f707u, 0x10fd0df5u, 0x00fd02feu, 0xfef804fbu, 0x0a0a03f9u, 0xf8fbfa0eu, 0x0a09fb00u, 0xfb08fffbu, 0x0000f90au, 0x0b0311fcu, 0x05fbf207u, 0xff09f402u, 0xf9f8fd1du, 0x0a030bf6u }, // block 27 (dxil label 1161)
    { 0x040d03e9u, 0xe9f00df6u, 0x1ee00bdeu, 0xf9efeefdu, 0xfcfdfb03u, 0xfa03fb04u, 0xf600f40fu, 0x05f8e9fau, 0xf90000f2u, 0xf2fcf800u, 0x07e5fdeeu, 0xfef3e1f4u, 0x0607fbf0u, 0xeb0a19f1u, 0x00dc0ddcu, 0x27fd58f6u }, // block 28 (dxil label 1193)
    { 0xf7fcfdf3u, 0x0816faf4u, 0xf6ebfdf4u, 0xfbea0105u, 0x0102020du, 0xfef50101u, 0x040a0103u, 0xfe030105u, 0x0004f806u, 0xf50b00fbu, 0xfdfbff0au, 0x09040bfbu, 0x01f9faf7u, 0x0005f5ffu, 0xfffbfcfbu, 0xfefdf90fu }, // block 29 (dxil label 1225)
    { 0x08faf0ffu, 0xfee7f0f1u, 0x0a12fd1eu, 0x13f9e0f7u, 0x050e0df4u, 0x000b0508u, 0xfff005f1u, 0xf908fdfdu, 0x01f4fa09u, 0x0901f709u, 0xfaf30efbu, 0xf501fcfdu, 0xf9d9022eu, 0x1fe7f5fdu, 0xfffb03ffu, 0xe8e4f0ffu }, // block 30 (dxil label 1257)
    { 0x0446f4d2u, 0x2c2df311u, 0xfb541132u, 0x2d5c18f2u, 0x076916c1u, 0x01a8fcf0u, 0xf0e2f249u, 0x06b0fff8u, 0x92d60c0du, 0xc112e81du, 0x1016e514u, 0xdfd80efdu, 0xfe021a0au, 0x26d023dfu, 0xe10911fbu, 0xdd0305e3u }, // block 31 (dxil label 1289)
    { 0x051d0921u, 0x0ad50705u, 0x13f90409u, 0xed020905u, 0xfaf7fa03u, 0x0f040a08u, 0xf4f809f8u, 0xfcfcfff9u, 0x09f7fa01u, 0x07f40204u, 0xf8210aebu, 0xfcfbfd0bu, 0x08f2f3f2u, 0x031cfa0bu, 0xecf0fd17u, 0x0904e4f4u }, // block 32 (dxil label 1321)
    { 0x08020509u, 0xf6f4fffcu, 0x0200f7ffu, 0xf905e806u, 0xff06fdfdu, 0xf9060202u, 0x0104ff03u, 0x08060b02u, 0xff06fc06u, 0xf703f9f7u, 0x0101fd0eu, 0x09080001u, 0x02fffcfeu, 0xf90904fcu, 0xfe03fb03u, 0x12060005u }, // block 33 (dxil label 1353)
    { 0xea08f114u, 0x06f1f6fau, 0xf41c040du, 0x03f10806u, 0x0bfcff02u, 0x01ff0707u, 0x040207f4u, 0xf009fd06u, 0x0ff7fdfeu, 0x060e0302u, 0x00fb03f3u, 0xf713f4fdu, 0x0f0005f7u, 0xed040f0eu, 0x08eefff2u, 0x0410eeffu }, // block 34 (dxil label 1385)
    { 0x0cf403f8u, 0x0607fe09u, 0xfbe00403u, 0x07edfbf7u, 0xfefc0601u, 0x08fe0402u, 0x000b0601u, 0x00fb0902u, 0xf90b0304u, 0x030303f6u, 0xf8fef80cu, 0x040001f0u, 0xfd0cf308u, 0xec02f0efu, 0x0505f505u, 0xf6060213u }, // block 35 (dxil label 1417)
};

static const uint W2[8][16] = {
    { 0xb3b33c1au, 0xbd67e75bu, 0xbb59361au, 0x4259084cu, 0xe3fa3e02u, 0xee022530u, 0xf1fcfdd8u, 0x14eca01bu, 0x121226e5u, 0xdcd7f015u, 0xd91bdd2fu, 0x1110b2ecu, 0x32df1009u, 0xfd1adf29u, 0xfa0330f6u, 0xeb69f01fu }, // block 0 (dxil label 1712)
    { 0xcc0124b0u, 0x2dc0eaf5u, 0x1c47640eu, 0xf0ffe315u, 0x0127ce2cu, 0x16f30d05u, 0x39120df1u, 0xf60ecacau, 0xe9f2d12bu, 0xc2c1e4b3u, 0xd40d05acu, 0xd3c74510u, 0xc9f9e645u, 0x1bf4eee4u, 0x44d63607u, 0xfab86521u }, // block 1 (dxil label 1748)
    { 0x13d4f3f5u, 0x10fbf80eu, 0xf417f131u, 0x09f4491cu, 0xea0cf7e6u, 0xe6fdf746u, 0xc834d304u, 0x1cdc0fefu, 0x4445c128u, 0x15ec3d37u, 0x292050e7u, 0x531549c1u, 0x1210fd16u, 0x09ff0bf4u, 0xef33020au, 0x131b0a2du }, // block 2 (dxil label 1781)
    { 0xb6f3db26u, 0x1d0204ecu, 0x1dd71b00u, 0xf2ab5b0fu, 0x10e9e30au, 0x0d2bb5bau, 0x1e27da24u, 0x0828d8fcu, 0xe3e62b1fu, 0xe5fefe24u, 0xe8d9f1f3u, 0x14f1de00u, 0x171b02f9u, 0xf0d6df38u, 0xd80cfc0du, 0xe8f4c9eau }, // block 3 (dxil label 1814)
    { 0x05900603u, 0xebdaffdeu, 0xefea042du, 0xe200cd39u, 0x51082aa9u, 0xa4f133dcu, 0xe4e91e3bu, 0x3107670fu, 0x0659d124u, 0xcd02f5c4u, 0xf3102948u, 0x19a8a13bu, 0x07140c2eu, 0x20ece70bu, 0xe1ee0b00u, 0xf10450e3u }, // block 4 (dxil label 1847)
    { 0xc5fad8f6u, 0xeaeddef5u, 0xe4c212f1u, 0x4b2ffd02u, 0x22e4f801u, 0xeef3b9f9u, 0x0d2deb31u, 0x2efa28fcu, 0xf74bbd0eu, 0xd013fed0u, 0xdf0c283au, 0x0b9bad36u, 0x0d2c08efu, 0x2cb70abdu, 0x2ed2e01au, 0xf2f3f7d2u }, // block 5 (dxil label 1880)
    { 0xb40f03fau, 0xe6f43b01u, 0xf306fbedu, 0xdb28c71cu, 0x2de6f4f1u, 0x0a35e93au, 0xe6fa26e3u, 0xde610515u, 0xbeb25030u, 0xb165db59u, 0xc6644a21u, 0x586eff5bu, 0x06d9f5d6u, 0xea0c1105u, 0x0f1c1beeu, 0x1af2c721u }, // block 6 (dxil label 1913)
    { 0x330e1aafu, 0xb9f42cfeu, 0xdeef1129u, 0x16f16a09u, 0xf40a11fau, 0x01e7fa03u, 0x3c10ce20u, 0xec07f513u, 0x2a07c615u, 0x01eafa04u, 0x1d1502d1u, 0xe4134928u, 0xc2de2501u, 0xc7f0f5d8u, 0xdbe43111u, 0xcb2404cbu }, // block 7 (dxil label 1946)
};

static const uint W3[8][16] = {
    { 0x26fad524u, 0xd512d6bau, 0x1c13cae1u, 0x35fcec3bu, 0xbcdae8d0u, 0x1014e9d4u, 0xf6f103f2u, 0x2c004bf6u, 0xf90cefe2u, 0x0bdfeff3u, 0xe10ef218u, 0xf21af1e3u, 0x96d71700u, 0x5f53ebe0u, 0xf2eb09beu, 0x22d5f5a8u }, // block 0 (dxil label 2244)
    { 0x24081506u, 0x0afb0cfeu, 0xc6e3e929u, 0x1b2bd6eeu, 0xff52d8bbu, 0x02b5040au, 0xf847462cu, 0x1bce0554u, 0x32190eefu, 0xf9e8e004u, 0xe2261815u, 0x35cc16e2u, 0x033f411bu, 0x13bf48bdu, 0xf2eb61fau, 0x110316b3u }, // block 1 (dxil label 2279)
    { 0x354d9fa8u, 0x9bd9fff1u, 0xf9fae621u, 0xc9090454u, 0x0a124903u, 0xbfd7e0dbu, 0x0329b9f3u, 0xfeeb2f5bu, 0xe91ff7efu, 0xd30a3ac3u, 0x03f6e90cu, 0xdff20037u, 0xd2ddf409u, 0x240df51eu, 0x0a22fef5u, 0x3013ebeeu }, // block 2 (dxil label 2361)
    { 0xb0e13790u, 0xda16cdc7u, 0x3659d54cu, 0x3eb70aabu, 0x031f67f5u, 0x0ef246f4u, 0x19090adcu, 0xe3e3da95u, 0x0731d437u, 0xf7d005cbu, 0xf8060920u, 0xe2efd641u, 0x031d000eu, 0x0df206d6u, 0x0f0e36ecu, 0xf8dbedfcu }, // block 3 (dxil label 2393)
    { 0x18021aa0u, 0x0523d9c9u, 0x09fc12ecu, 0xdf03eff4u, 0xebe4200bu, 0x30eafedeu, 0x23132449u, 0x050713c4u, 0xbee81f4bu, 0x0c13eef6u, 0x0fc31a08u, 0xfe0500f5u, 0x4fe6ca35u, 0x0702d3fbu, 0xed541206u, 0x090612f3u }, // block 4 (dxil label 2427)
    { 0x1cf71bfau, 0xcc0606dau, 0xdb61d6f5u, 0x180fdfd9u, 0xed09ff1du, 0xfaeed30bu, 0x28e815f3u, 0xd9f3f5fbu, 0x0a05114bu, 0x21fc0512u, 0x18af34edu, 0xeafdf3f6u, 0x14cc19ceu, 0x2033e2cfu, 0x14bec221u, 0xf14208e6u }, // block 5 (dxil label 2459)
    { 0xe0242366u, 0xf30c0822u, 0xf5fddef0u, 0xdcfc0804u, 0xff0c1aecu, 0x06fa0a63u, 0x32381af5u, 0xdad70aeeu, 0xca18ce0bu, 0xce12d628u, 0x2ff5effeu, 0xe202f342u, 0x63ee00abu, 0x06fae94fu, 0x19e2faf4u, 0x12e524f3u }, // block 6 (dxil label 2493)
    { 0x1cfc0ff2u, 0x480714ddu, 0xe29320fdu, 0x19ef1deeu, 0xf7f100f1u, 0xdd02231fu, 0x0a04f3f4u, 0xec10e5fdu, 0xdb5f15eau, 0x109bdb31u, 0x1ee62213u, 0xd701c8ebu, 0xf403f841u, 0xf3fcfdb6u, 0x3046befdu, 0x2303c40cu }, // block 7 (dxil label 2525)
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
void reimpl_pass1_4320(uint3 tid : SV_DispatchThreadID)
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint tz = tid.z;
    if (tx > 3839u || ty > 2159u)
        return;
#if defined(DIAG) || defined(DIAG3) || defined(DIAG4)
    if (ty >= 90u)
        return;
#endif

    // ---- stage 1: 3x3 depthwise conv 16ch -> 16ch, bias0 = t18@7168 ----
    int A8[16];
    [unroll]
    for (int w = 0; w < 4; w++) {
        const uint4 v = t18_initializer.Load4(3456u + (tz << 6) + 16u * w);
        A8[4 * w + 0] = asint(v.x);
        A8[4 * w + 1] = asint(v.y);
        A8[4 * w + 2] = asint(v.z);
        A8[4 * w + 3] = asint(v.w);
    }

    uint4 rowD[9]; // per offset: 4 packed dwords = 16 int8 channels
    [unroll]
    for (int o = 0; o < 9; o++) {
        const uint addr = 0u + 61472u * (ty + OFF[o].x) + 16u * tx + OFF[o].y;
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
    // ---- stage 2: pointwise 16 -> 32 + ReLU, bias1 = t18@7808 ----
    int A10[32];
    [unroll]
    for (int w2 = 0; w2 < 8; w2++) {
        const uint4 v = t18_initializer.Load4(4096u + 16u * w2);
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

    // ---- stage 3: pointwise 32 -> 16, bias2 = t18@8448, raw int32 acc ----
    int A12[16];
    [unroll]
    for (int w3 = 0; w3 < 4; w3++) {
        const uint4 v = t18_initializer.Load4(4736u + 16u * w3);
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
    const uint centerAddr = 0u + 61472u * (ty + 1u) + 16u * tx + 16u;
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
    const bool inRange = tx < ml2c_cb[2].x && ty < ml2c_cb[2].y;
    const uint outRow = 132902464u + 61472u * (ty + 1u) + 16u * (tx + 1u);

    [unroll]
    for (int d = 0; d < 4; d++) {
        int4 lanes;
        [unroll]
        for (int j3 = 0; j3 < 4; j3++) {
            const int cbyte = (int)(((center[d] >> (8 * j3)) & 0xFFu) << 24) >> 24;
            const float t1 = float(cbyte) * 0.015625f;               // 1/64
            const float t2 = float(A12[4 * d + j3]) * 0.000244140625f; // 2^-12
            const float r = round((t1 + t2) * 64.0f);
            lanes[j3] = int16_wrap(int(r));
        }
        const uint val = inRange ? pack_bytes(lanes) : 0u;
        const uint addr = inRange ? (outRow + 4u * d) : 134217728u;
        ScratchBuffer.Store(addr, val);
    }
}
