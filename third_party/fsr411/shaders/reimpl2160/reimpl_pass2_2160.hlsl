// SPDX-License-Identifier: MIT
// Reimplemented FSR 4.1.1 kernel: semantics reconstructed from the AMD 4.1.1
// provider (Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll,
// SHA-256 d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46,
// MIT-licensed per FidelityFX-SDK docs/license.md) and validated byte-exact
// against the AMD container. See README.md and docs/evidence411/.
// Copyright (C) 2025 Advanced Micro Devices, Inc. (upstream implementation)
// Copyright (C) 2026 int3rrobang (reimplementation)

// 2160-class variant of reimpl_pass2.hlsl (mechanical 2x-spatial transform, no other changes).
// 1080 reference container: analysis/extracted/4.1.1-official/0063_00278960_38bcb47664887939.dxbc (1080, guards 959/539, stride 15392, read base 8342464)
// 2160 AMD container (ground truth): analysis/extracted/4.1.1-official/0141_00482860_d3f4f98f637fa5bc.dxbc (2160, guards 1919/1079, stride 30752, read base 33273664)
// Verified transform (full-DXIL diff, identical line counts, immediates-only deltas):
//   guards x2, inRange x2, full-res stride 15392->30752, half-res stride 7712->15392,
//   plane spacing 2097664->8342464, bases relocated per container (see 2160 line above).
// Weights / biases / requant / tail / loop structure: byte-identical to the 1080 file.
// ============================================================================
// 4.1.1 pass2 reimplementation (our_reimplementation) - Global 5099
// ConvNextBlock: 3x3 dw conv 16->16 -> requant -> pw 16->32 + ReLU -> requant
//                -> pw 32->16 -> residual + float requant -> store.
//
// Semantics reconstructed from the 4.0.2 ML2Code ConvNextBlock topology
// (ml2code_runtime/operators/int8_NHWC/Fused/ConvNextBlock.hlsli) and the
// 4.1.1 container DXIL (38bcb476...). Weight tables are extracted from the
// container's compile-time constants; biases are read from the captured
// runtime initializer (t18): bias0 i32[16] @7168, bias1 i32[32] @7808,
// bias2 i32[16] @8448. The semantics were validated byte-exact against the
// AMD oracle by bench/tools/pass2_sim.py before this HLSL was authored.
// ============================================================================
ByteAddressBuffer t18_initializer : register(t18);
RWByteAddressBuffer ScratchBuffer : register(u11);

cbuffer MLSR_Optimized_Constants : register(b0)
{
    uint4 ml2c_cb[17]; // reg3.xy = logical tensor width/height
}

static const uint W1[36][16] = {
    { 0xfaf80114u, 0x0c05f6fdu, 0x03f3050bu, 0xfa010b07u, 0x0ffb0201u, 0xf5edf606u, 0xfb000402u, 0x00f7fef5u, 0xeffff90au, 0xfb1a14ebu, 0x1504e11du, 0x08230015u, 0x0302fb02u, 0xfbf201f7u, 0x120af606u, 0x01ff0b07u }, // block 0 (dxil label 94)
    { 0x050205f7u, 0x08040308u, 0x0108f911u, 0x07fe1afcu, 0xf321fef4u, 0x0d090800u, 0x07f903e9u, 0x06000a15u, 0x1818180bu, 0x1119fd16u, 0xe8f611f2u, 0x0e1618e9u, 0x02fb0619u, 0x04fdfcf1u, 0x05fffbfeu, 0x0b0807f9u }, // block 1 (dxil label 129)
    { 0xfd1202ebu, 0x11fbfd00u, 0xfe0df8fbu, 0xfb06080au, 0xfeea0716u, 0x01fdfd06u, 0xfe050e0bu, 0xf6090303u, 0xebf30000u, 0x12e6f408u, 0xf31e0ffcu, 0xf1ecee0cu, 0xfb040002u, 0x03010000u, 0x0104ff01u, 0xfd00f906u }, // block 2 (dxil label 161)
    { 0x14070de7u, 0xf2e106fdu, 0x16ff1003u, 0x04010003u, 0x0f1500eeu, 0x16000a05u, 0xedeb0cf3u, 0xf7f1f4f0u, 0x020bf004u, 0x070af11bu, 0xe6070ee3u, 0xefec0705u, 0xefeffb10u, 0x021a03f7u, 0x05f2f30au, 0xfa140017u }, // block 3 (dxil label 193)
    { 0xf72b0818u, 0xd73b2d29u, 0xd4ed25fbu, 0xeae2f5e7u, 0xbccdd610u, 0xd221f4e6u, 0x0123f3fdu, 0x0c17a1f9u, 0xd2d20ce8u, 0xfac8e8c8u, 0x05edf1f1u, 0x20ac0c0du, 0x194e1116u, 0x6ec5ef15u, 0x1e03ec44u, 0x16000399u }, // block 4 (dxil label 225)
    { 0xd5e2f711u, 0xfd0f07feu, 0x0c08f1eau, 0xf2f5e229u, 0x050a0913u, 0x00fefcfeu, 0x0d05fa0bu, 0x0df60cf8u, 0xfdf6ffebu, 0x00091407u, 0x0b13edfcu, 0xec08f514u, 0x00ff03e4u, 0x09050602u, 0x01f7f5feu, 0xf90af511u }, // block 5 (dxil label 257)
    { 0x062106f5u, 0xfc0603ffu, 0x00f0ec08u, 0x130900f0u, 0x0d0ffff6u, 0x02f40102u, 0x03f802ffu, 0x10020ee6u, 0x022710f9u, 0x03ff0cffu, 0xf9e7fefeu, 0x0611f2f4u, 0xf80108f5u, 0x0af6fe01u, 0x0513f402u, 0xec01e811u }, // block 6 (dxil label 289)
    { 0x1715f5f1u, 0xed15f7eeu, 0xeee313cau, 0xfd070004u, 0x00080c0bu, 0x0ce90202u, 0xff0c000au, 0xecfdfa0du, 0x0af0f3f7u, 0xfb030c02u, 0x0cfa06f8u, 0xf3fdec0du, 0x15f4fde8u, 0x00ff0d05u, 0x120102d9u, 0xd611dd2eu }, // block 7 (dxil label 321)
    { 0xfdf201f9u, 0xf9050709u, 0xffef0ee4u, 0xe60bec10u, 0xfc0c04ffu, 0x0e0506fcu, 0x04fcff03u, 0x0c0b06fdu, 0x0e110309u, 0xf7f6eefdu, 0xf5f3010du, 0x0afa0debu, 0xfd0f060eu, 0xfdf6f901u, 0x030afe02u, 0xfefd1700u }, // block 8 (dxil label 353)
    { 0x0decfe0eu, 0xe6ebf202u, 0x05090405u, 0xfdedfcf6u, 0xf5070b08u, 0x10270af6u, 0xf4d5fd1cu, 0xf71ff014u, 0x07fd08f6u, 0xf700f809u, 0x05fa1efcu, 0xf7f90afau, 0xfafaf811u, 0x05f903fdu, 0x0012faf8u, 0x0602080cu }, // block 9 (dxil label 468)
    { 0xf2190107u, 0x04070509u, 0x11f80bdeu, 0x04f6ff09u, 0x190d02e8u, 0x1207fcfau, 0x11e227eau, 0x030331ebu, 0x03f1fc07u, 0xf9f6f9f9u, 0xfa06f511u, 0x08fa05f9u, 0x12170df7u, 0xfa0f06f8u, 0x07e518f8u, 0x0b0b0df6u }, // block 10 (dxil label 500)
    { 0x00f10c08u, 0x0300fd04u, 0x02011411u, 0xf70c0d05u, 0x0003ff06u, 0x050306f9u, 0x0ae601f2u, 0xf7f2f7fdu, 0xfa10e8f7u, 0xf60700fdu, 0xfafff0f5u, 0x11fe0808u, 0xfd0afd03u, 0x07fd02fau, 0xfdf9f9f7u, 0x02ff0400u }, // block 11 (dxil label 532)
    { 0x00eef9eeu, 0x1d02fef7u, 0x091208fbu, 0xea02011du, 0xedd6f0e8u, 0x14fa02feu, 0x09e8f715u, 0x00dd0ee4u, 0x0930fc0du, 0xed181c07u, 0xd4cef607u, 0x1d170fe4u, 0x00d8fe13u, 0xea00f6fau, 0x1a0e12ffu, 0xfffc040fu }, // block 12 (dxil label 564)
    { 0x1ad2ff13u, 0xe20c31f7u, 0xa243cef8u, 0x2f1514e8u, 0xdc101b06u, 0xc0eec316u, 0xe607dad7u, 0x41f9eb27u, 0xcae61c04u, 0xfd3cf415u, 0xe3d119d9u, 0xd8bbf225u, 0x0f021d02u, 0xb3220e0cu, 0xf0d9223du, 0x27d821acu }, // block 13 (dxil label 596)
    { 0xff0d0606u, 0xf9f30800u, 0x0e0fe801u, 0x0a01ff07u, 0x02140b06u, 0x111813fcu, 0xf9eeef05u, 0x0f2b12efu, 0x00e503ddu, 0x0517e6f6u, 0xf20111dbu, 0xf9ff0209u, 0x110505f2u, 0xfdfb0809u, 0xfbff0ef4u, 0xf903fcf9u }, // block 14 (dxil label 628)
    { 0x060a07fdu, 0xfc00feffu, 0x08f3050eu, 0x040608f3u, 0x0a0d0af8u, 0x0ffb05feu, 0x14fbfbf8u, 0xfdf0f9fau, 0x0c09f8eau, 0xf2faf204u, 0xf81302edu, 0x01f60b08u, 0x01eefdffu, 0x04f5f900u, 0x0d10fffbu, 0xf5fbfd08u }, // block 15 (dxil label 660)
    { 0x0dedffffu, 0x0501fc0au, 0x071901e2u, 0xd504f220u, 0xeae9fc1au, 0x050101fbu, 0xef0bf60du, 0xe819f71au, 0x04ddfafcu, 0xeb0d0006u, 0x10130ae1u, 0xe411f730u, 0x03e7fef9u, 0xf1ffff0du, 0x07230ae8u, 0xde05ed25u }, // block 16 (dxil label 692)
    { 0xf30a0b00u, 0x0f0f0bfbu, 0x08f90704u, 0xfd06fc00u, 0x0bf5f0ecu, 0xfa010406u, 0xfdf01ec8u, 0xe2eae11du, 0xf3fa0203u, 0xfd150002u, 0xfcf6ff15u, 0x0208fcffu, 0xfafbfbf8u, 0x0efafc01u, 0xfd0203f7u, 0xf5fcfc0cu }, // block 17 (dxil label 724)
    { 0x07eaebf4u, 0x0de606f2u, 0x0ffd07feu, 0x04e2f9e7u, 0x0b0afef9u, 0xf6f5fbffu, 0xfb09fffcu, 0x04f405f4u, 0xff090308u, 0x000a000au, 0xfcff0ff0u, 0x03080a08u, 0x05210ae2u, 0x160a09feu, 0xf4e1f909u, 0x02130affu }, // block 18 (dxil label 819)
    { 0xe800f72du, 0xe417f201u, 0xda22da0cu, 0x080be110u, 0xf6050409u, 0x0b0ffe05u, 0xfd020901u, 0xf6070f0cu, 0x0cf918f0u, 0x06e80412u, 0x0021fb01u, 0xf6070104u, 0xfcf5ea04u, 0xe5d4f3ecu, 0x1b11f9e8u, 0x10defc03u }, // block 19 (dxil label 851)
    { 0x0b0d08efu, 0xf5fafd00u, 0xf11306f3u, 0x100f10f9u, 0x07fe04fdu, 0xf40703feu, 0xfef50a02u, 0x040f06feu, 0x04f9fa07u, 0xef0506f8u, 0x06fcf9f8u, 0x08fdf1fbu, 0x01f0fe0du, 0xfffcfc12u, 0x000705ffu, 0xfafbfbf8u }, // block 20 (dxil label 883)
    { 0xfcfd050fu, 0x0506e7edu, 0xf421f0f5u, 0xed07e727u, 0xfa0500f8u, 0x090911fdu, 0xe8e4fa0au, 0xfc02f9f8u, 0x0fe10cf6u, 0xf5e401feu, 0x1217ec04u, 0xffededeeu, 0xee320b0du, 0xfa1b171eu, 0xf1e6f511u, 0x1d0605dfu }, // block 21 (dxil label 915)
    { 0xe4e941e3u, 0x2d61e3eeu, 0x469c09f4u, 0xfce941d9u, 0x1df3f30fu, 0xc4b82801u, 0x2d6fed0au, 0x12d1f8f5u, 0xcbcb3a2eu, 0xfc0b3bcau, 0x43e0f9c9u, 0xe4cae142u, 0x3d0f21eau, 0x1c2203e2u, 0xf9d046d9u, 0xc9112829u }, // block 22 (dxil label 947)
    { 0xe7f5f307u, 0x19fb05f4u, 0x00eefc21u, 0x10e5eaf5u, 0xfcfd0607u, 0x0c07f5fcu, 0xfdf9fb21u, 0x0cf508e8u, 0x0d032011u, 0x010cfb11u, 0x141e1103u, 0x00111d05u, 0x0003f0f5u, 0xf10f0cfcu, 0xffe2fe00u, 0xfafceafau }, // block 23 (dxil label 979)
    { 0xfe0200feu, 0xecfafe07u, 0xec11eb18u, 0x0a05f8f4u, 0xfdfbfdfau, 0x00ff0004u, 0xfb00010bu, 0x01ff0500u, 0x110b00f0u, 0xfcf7feffu, 0xfbf007ffu, 0x09fcfcedu, 0xfa0509f6u, 0x0df405fau, 0xfc09f6e8u, 0xf5fef011u }, // block 24 (dxil label 1011)
    { 0x022301d8u, 0x39f4ffe9u, 0xfbdd0beau, 0x0ae5f1f0u, 0x011210ffu, 0x08f2fdf9u, 0xfef5fd0au, 0x0efd02edu, 0xf8090c1bu, 0xf3160507u, 0x1607ed0eu, 0x19041cfdu, 0x03030a0au, 0x1f000f0bu, 0x0404090cu, 0xf704fdfbu }, // block 25 (dxil label 1043)
    { 0x04fe0e09u, 0xf8f0f20fu, 0xfd26feecu, 0xeffc0b18u, 0x0504fc02u, 0xf900ff01u, 0x020bfff6u, 0xffff0bffu, 0x070102fbu, 0xf3fa01f9u, 0x05f6f61du, 0x0a0b0101u, 0xff06f901u, 0xf6faf8f9u, 0xf8fdf0f2u, 0x0dfc01f5u }, // block 26 (dxil label 1075)
    { 0x051f0aefu, 0x1a0c0c0cu, 0xf1ea09f5u, 0xfcff0904u, 0xf606fdffu, 0x0a1205fcu, 0xeff7fc07u, 0xfb10ff0du, 0x00fc0706u, 0xf0effa04u, 0xf70fee0fu, 0xf9000209u, 0x03390b12u, 0xf719fe00u, 0xedfb09f1u, 0x0d14150cu }, // block 27 (dxil label 1170)
    { 0x0af4f2e5u, 0xfdcf0cc4u, 0x1016be11u, 0x2a06fbecu, 0x15fb03f3u, 0xf5f3e609u, 0xfb2505f5u, 0x04f406e5u, 0x0fececf4u, 0xfff90fedu, 0x24da0fe2u, 0x0be702e6u, 0x08f50ae6u, 0x0debfef5u, 0x0bfd05f9u, 0xf5fd0102u }, // block 28 (dxil label 1202)
    { 0xf602fd16u, 0x0c03fc0du, 0x0315f601u, 0xeae3ec05u, 0xfbf8f502u, 0xf5eef903u, 0xf40d07f8u, 0x0aebfcf4u, 0xfdfcf9f1u, 0x01f7fd05u, 0x07f8f80bu, 0xf7f2f506u, 0x0b050600u, 0xfa0c08fcu, 0x0ce9030bu, 0xf809f6f7u }, // block 29 (dxil label 1234)
    { 0xf8352a20u, 0x04fa040fu, 0x08f4fb04u, 0x27fb03ceu, 0x03d3effbu, 0xefd6d902u, 0x072111edu, 0xfdc9f4e3u, 0xf4fefb00u, 0x00010304u, 0x0af91ee7u, 0x06f40df9u, 0xe8ce09ffu, 0xf7f61107u, 0x2611e32bu, 0x0c010b06u }, // block 30 (dxil label 1266)
    { 0xe525e4f4u, 0xdbe44811u, 0xbeadcd50u, 0x44d158dfu, 0xf944dd1cu, 0x29182cf0u, 0xf4b0d51du, 0xeaf7fb11u, 0x0c2445f9u, 0xe220f3fau, 0xc85bf4ffu, 0x1ffdc4e2u, 0x10fb09d4u, 0x2ef2e6c3u, 0xe645e01au, 0x2bcc01cbu }, // block 31 (dxil label 1298)
    { 0x09f6e9fcu, 0xef0100fau, 0xecc820b6u, 0xf5e6dbfeu, 0x0e0c0a06u, 0x11161205u, 0xf8060e06u, 0x0a2f11fau, 0x0007fffbu, 0xf30107f9u, 0xf1f8fef9u, 0xf514fd10u, 0x00fffd01u, 0xf20bf614u, 0xfe01fb0cu, 0xf4fa0502u }, // block 32 (dxil label 1330)
    { 0xfcd2f211u, 0xf0fb06fbu, 0x080e07e0u, 0xf2f8e118u, 0x110e0b00u, 0xf9e5f701u, 0xfbfe09fau, 0x10f704f0u, 0xf6f7fe18u, 0x0e0806fdu, 0x09fa0cf3u, 0xfb021019u, 0xfe0108f7u, 0x04f903f4u, 0x0dfc0305u, 0xf3ffed0cu }, // block 33 (dxil label 1362)
    { 0x143512fbu, 0x1fd60905u, 0xfaf2f50eu, 0x07ee08e7u, 0xebeaf414u, 0xff0d0c09u, 0x111af70bu, 0x0a0f0808u, 0xf8f80d16u, 0xe9f20b05u, 0x0204f51cu, 0x1602fce9u, 0x0a11070eu, 0xea13f613u, 0xfc080300u, 0x05121efeu }, // block 34 (dxil label 1394)
    { 0xf9ef08fdu, 0x11fefb07u, 0xfb06fa0du, 0xedfcf818u, 0x090302f9u, 0xf8faf602u, 0xfa0902e6u, 0xf1f5ff10u, 0xfe0dfd00u, 0x090904feu, 0x00010bf6u, 0x0afc16f7u, 0xf600fe09u, 0xfdf703f0u, 0x0ffefa1du, 0x0a050ff8u }, // block 35 (dxil label 1426)
};

static const uint W2[8][16] = {
    { 0x21d8e4fdu, 0x401a0bfau, 0xe8da0906u, 0x02f70ffdu, 0xe3002d20u, 0x4509dbeau, 0x0aef21f6u, 0x0d1d0befu, 0x17f8cef4u, 0xeef1df22u, 0x2feec7f5u, 0xf2f60fc4u, 0xb903dce9u, 0x4017ff1cu, 0x0deffb0au, 0x1be6f4f2u }, // block 0 (dxil label 1721)
    { 0xe8fc2ff4u, 0x1c21f410u, 0xd00f23d1u, 0xc1e23122u, 0x27e63502u, 0xdce3f0b1u, 0xe7bdd8d5u, 0xfbda17f1u, 0xff0f0225u, 0x21ec24f1u, 0xc8162400u, 0x11c012f7u, 0x17e2c00cu, 0x27f501e9u, 0x31f3c1edu, 0x27211113u }, // block 1 (dxil label 1757)
    { 0xf3cc123bu, 0xed151105u, 0x1b07c3f5u, 0xd41213f8u, 0x24fde604u, 0xc10414e5u, 0xfd0039ffu, 0xf20e1feau, 0x3b34dd0fu, 0x1313f4f8u, 0xffee27f5u, 0x10f4c901u, 0xf0f5db28u, 0x24d2d129u, 0xd0d9ac1cu, 0x0b14fb1eu }, // block 2 (dxil label 1790)
    { 0xf5f03c09u, 0xf1fb4234u, 0xc5d42eccu, 0x145ed9f6u, 0x9150341cu, 0xe44410cau, 0xbac51afdu, 0x0f1e1058u, 0xe0ebfbebu, 0xb70df0efu, 0x2301590au, 0xdbf5bbf3u, 0x27f5f6e7u, 0x09eae805u, 0xf42556feu, 0x03100711u }, // block 3 (dxil label 1823)
    { 0x1f08e9f7u, 0x210eee03u, 0xca07ef08u, 0xe21f03e5u, 0x20e8c71bu, 0xd5cbd313u, 0x00d34d9bu, 0xd5b3300cu, 0xfe0f15ccu, 0xe5e8fee0u, 0xfdf90e02u, 0x09fafdffu, 0xf9fad8ddu, 0xac022f2du, 0xddd0ccfbu, 0x0ddff70cu }, // block 4 (dxil label 1856)
    { 0x22ed5fd9u, 0xd91217a1u, 0xd3d29022u, 0x260feae2u, 0xc6fada06u, 0xced220e8u, 0x03f93805u, 0xf1242603u, 0x29ef27ecu, 0x48250b23u, 0x21e4f709u, 0x0be3060du, 0x1318e7e1u, 0x06e8e80bu, 0xe5f55f03u, 0x29f40d1cu }, // block 5 (dxil label 1889)
    { 0xdc1710ffu, 0xba05e701u, 0xfd020508u, 0x1b070d04u, 0xf203b3f7u, 0xf211eed0u, 0x3417cf28u, 0xe3b31efdu, 0xe50c14e7u, 0xce14f2ddu, 0xf92cedd5u, 0x45f530eau, 0x06010c0bu, 0x1609f2f2u, 0x0f0fbe00u, 0xf6eaeffdu }, // block 6 (dxil label 1922)
    { 0x12391902u, 0x040c10d1u, 0xce022a09u, 0x0ef5d509u, 0xcd02d7f8u, 0x5010f521u, 0x1d1c140au, 0x0901e7f2u, 0xffc5eb1au, 0xf723ddf1u, 0xd8f711ffu, 0x11dfae01u, 0xa7552a2bu, 0x0b320bc5u, 0xd81429f4u, 0xe330e650u }, // block 7 (dxil label 1955)
};

static const uint W3[8][16] = {
    { 0xe71bfdfeu, 0xe70224f6u, 0x1818f224u, 0xf8f5f9eau, 0x3cef0e0du, 0xeeeec31eu, 0x14ee39e6u, 0xfc011fe6u, 0xeecbd105u, 0xeadf270cu, 0xf5041902u, 0x1fde372fu, 0x473cf208u, 0xd5defbf0u, 0x29191f08u, 0xded892f1u }, // block 0 (dxil label 2253)
    { 0xf807fbe3u, 0xe14837f6u, 0xc134fbfeu, 0xfdfc1a38u, 0xeee4cef8u, 0xfd212239u, 0x03f4f8bbu, 0xea2bbe2du, 0xd4ea220fu, 0xf5bbf101u, 0x330c28fcu, 0xe41033ebu, 0x55e631f1u, 0x0f2d14dau, 0xe51e08d7u, 0x5a19b004u }, // block 1 (dxil label 2288)
    { 0xd4d2223eu, 0x082516cfu, 0xace231e6u, 0x03013ba3u, 0x01f1f21du, 0xe1f8b003u, 0x170bec0du, 0x00f62c23u, 0x2302ec13u, 0x03f1e013u, 0x1f0316eau, 0x2712edfau, 0x010a0a11u, 0x0901fe21u, 0x262502f6u, 0xdc1ae318u }, // block 2 (dxil label 2370)
    { 0x22cd32e4u, 0x42f9e8fau, 0xdcecbebau, 0xd3e542fcu, 0xed0e63f1u, 0xf4340593u, 0x070f1ac5u, 0xd512f222u, 0xfbddfe23u, 0xe3fcf6ffu, 0xfbe502c7u, 0x12ffd7f2u, 0xfd241d17u, 0x1ffa09f7u, 0xf41c03d0u, 0x1df101ffu }, // block 3 (dxil label 2402)
    { 0x1302e61au, 0x2020e1fcu, 0xdae4eae8u, 0xb51ce721u, 0x08e0bcf2u, 0xc0c018e5u, 0xf0061f03u, 0x11a80c15u, 0x1d26e2f0u, 0x162122d8u, 0xe3c9080du, 0x2807dd07u, 0xeeedfdc0u, 0xf5d32a3fu, 0x16c435c4u, 0xaa082b16u }, // block 4 (dxil label 2436)
    { 0xd410fef5u, 0x5ff8e8e8u, 0x3bd717f2u, 0x14f503f6u, 0xddfa0be1u, 0x04cef935u, 0x230d0d18u, 0x05430bf8u, 0xf7ee05e3u, 0xcc450121u, 0x20db152cu, 0x2224e300u, 0xbf1010cau, 0x4a02392au, 0x15bb020fu, 0xe0513938u }, // block 5 (dxil label 2468)
    { 0x0ff414bfu, 0x230cdb17u, 0xfbc5feeeu, 0x1126280bu, 0x1f02240cu, 0xeee6c7fau, 0xf50506fdu, 0xe9dcf116u, 0xfe210319u, 0x0bfb3f0fu, 0x22e4f7e8u, 0x251ac9f8u, 0xd700fe22u, 0xf20435f0u, 0xfc0ac100u, 0x07fa0640u }, // block 6 (dxil label 2502)
    { 0x1ff1250bu, 0xd10d10c6u, 0xecdd1b21u, 0xe2fff001u, 0x2f1902f5u, 0x37cdefd2u, 0x06f6fe11u, 0x0608d8e9u, 0x44f4181fu, 0xc611010fu, 0xedd403c1u, 0x2c0afcfbu, 0x2303aef0u, 0xcc14043du, 0x41064edau, 0xf0091618u }, // block 7 (dxil label 2534)
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
void reimpl_pass2_2160(uint3 tid : SV_DispatchThreadID)
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

    // ---- stage 1: 3x3 depthwise conv 16ch -> 16ch, bias0 = t18@7168 ----
    int A8[16];
    [unroll]
    for (int w = 0; w < 4; w++) {
        const uint4 v = t18_initializer.Load4(7168u + (tz << 6) + 16u * w);
        A8[4 * w + 0] = asint(v.x);
        A8[4 * w + 1] = asint(v.y);
        A8[4 * w + 2] = asint(v.z);
        A8[4 * w + 3] = asint(v.w);
    }

    uint4 rowD[9]; // per offset: 4 packed dwords = 16 int8 channels
    [unroll]
    for (int o = 0; o < 9; o++) {
        const uint addr = 33273664u + 30752u * (ty + OFF[o].x) + 16u * tx + OFF[o].y;
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
            q[j] = requant_shift(A8[f + j], 8u, 127);
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
        const uint4 v = t18_initializer.Load4(7808u + 16u * w2);
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
            q2[j2] = requant_shift(max(A10[f2 + j2], 0), 7u, 63);
        relu_d[f2 / 4] = pack_bytes(q2);
    }

    // ---- stage 3: pointwise 32 -> 16, bias2 = t18@8448, raw int32 acc ----
    int A12[16];
    [unroll]
    for (int w3 = 0; w3 < 4; w3++) {
        const uint4 v = t18_initializer.Load4(8448u + 16u * w3);
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
    const uint centerAddr = 33273664u + 30752u * (ty + 1u) + 16u * tx + 16u;
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
    const bool inRange = tx < ml2c_cb[3].x && ty < ml2c_cb[3].y;
    const uint outRow = 30752u * (ty + 1u) + 16u * (tx + 1u); // output plane base 0

    [unroll]
    for (int d = 0; d < 4; d++) {
        int4 lanes;
        [unroll]
        for (int j3 = 0; j3 < 4; j3++) {
            const int cbyte = (int)(((center[d] >> (8 * j3)) & 0xFFu) << 24) >> 24;
            const float t1 = float(cbyte) * 0.015625f;               // 1/64
            const float t2 = float(A12[4 * d + j3]) * 0.000244140625f; // 2^-12
            const float r = round((t1 + t2) * 32.0f);
            lanes[j3] = int16_wrap(int(r));
        }
        const uint val = inRange ? pack_bytes(lanes) : 0u;
        const uint addr = inRange ? (outRow + 4u * d) : 134217728u;
        ScratchBuffer.Store(addr, val);
    }
}
