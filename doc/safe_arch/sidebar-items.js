initSidebarItems({"fn":[["add_i16_m128i","Lanewise `a + b` with lanes as `i16`."],["add_i32_m128i","Lanewise `a + b` with lanes as `i32`."],["add_i64_m128i","Lanewise `a + b` with lanes as `i64`."],["add_i8_m128i","Lanewise `a + b` with lanes as `i8`."],["add_m128","Lanewise `a + b`."],["add_m128_s","Low lane `a + b`, other lanes unchanged."],["add_m128d","Lanewise `a + b`."],["add_m128d_s","Lowest lane `a + b`, high lane unchanged."],["add_saturating_i16_m128i","Lanewise saturating `a + b` with lanes as `i16`."],["add_saturating_i8_m128i","Lanewise saturating `a + b` with lanes as `i8`."],["add_saturating_u16_m128i","Lanewise saturating `a + b` with lanes as `u16`."],["add_saturating_u8_m128i","Lanewise saturating `a + b` with lanes as `u8`."],["average_u16_m128i","Lanewise average of the `u16` values."],["average_u8_m128i","Lanewise average of the `u8` values."],["bitand_m128","Bitwise `a & b`."],["bitand_m128d","Bitwise `a & b`."],["bitand_m128i","Bitwise `a & b`."],["bitandnot_m128","Bitwise `(!a) & b`."],["bitandnot_m128d","Bitwise `(!a) & b`."],["bitandnot_m128i","Bitwise `(!a) & b`."],["bitor_m128","Bitwise `a | b`."],["bitor_m128d","Bitwise `a | b`."],["bitor_m128i","Bitwise `a | b`."],["bitxor_m128","Bitwise `a ^ b`."],["bitxor_m128d","Bitwise `a ^ b`."],["bitxor_m128i","Bitwise `a ^ b`."],["byte_shl_imm_u128_m128i","Shifts all bits in the entire register left by a number of bytes."],["byte_shr_imm_u128_m128i","Shifts all bits in the entire register right by a number of bytes."],["byte_swap_i32","Swap the bytes of the given 32-bit value."],["byte_swap_i64","Swap the bytes of the given 64-bit value."],["cast_to_m128_from_m128d","Bit-preserving cast to `m128` from `m128d`"],["cast_to_m128_from_m128i","Bit-preserving cast to `m128` from `m128i`"],["cast_to_m128d_from_m128","Bit-preserving cast to `m128d` from `m128`"],["cast_to_m128d_from_m128i","Bit-preserving cast to `m128d` from `m128i`"],["cast_to_m128i_from_m128","Bit-preserving cast to `m128i` from `m128`"],["cast_to_m128i_from_m128d","Bit-preserving cast to `m128i` from `m128d`"],["cmp_eq_i32_m128_s","Low lane equality."],["cmp_eq_i32_m128d_s","Low lane `f64` equal to."],["cmp_eq_mask_i16_m128i","Lanewise `a == b` with lanes as `i16`."],["cmp_eq_mask_i32_m128i","Lanewise `a == b` with lanes as `i32`."],["cmp_eq_mask_i8_m128i","Lanewise `a == b` with lanes as `i8`."],["cmp_eq_mask_m128","Lanewise `a == b`."],["cmp_eq_mask_m128_s","Low lane `a == b`, other lanes unchanged."],["cmp_eq_mask_m128d","Lanewise `a == b`, mask output."],["cmp_eq_mask_m128d_s","Low lane `a == b`, other lanes unchanged."],["cmp_ge_i32_m128_s","Low lane greater than or equal to."],["cmp_ge_i32_m128d_s","Low lane `f64` greater than or equal to."],["cmp_ge_mask_m128","Lanewise `a >= b`."],["cmp_ge_mask_m128_s","Low lane `a >= b`, other lanes unchanged."],["cmp_ge_mask_m128d","Lanewise `a >= b`."],["cmp_ge_mask_m128d_s","Low lane `a >= b`, other lanes unchanged."],["cmp_gt_i32_m128_s","Low lane greater than."],["cmp_gt_i32_m128d_s","Low lane `f64` greater than."],["cmp_gt_mask_i16_m128i","Lanewise `a > b` with lanes as `i16`."],["cmp_gt_mask_i32_m128i","Lanewise `a > b` with lanes as `i32`."],["cmp_gt_mask_i8_m128i","Lanewise `a > b` with lanes as `i8`."],["cmp_gt_mask_m128","Lanewise `a > b`."],["cmp_gt_mask_m128_s","Low lane `a > b`, other lanes unchanged."],["cmp_gt_mask_m128d","Lanewise `a > b`."],["cmp_gt_mask_m128d_s","Low lane `a > b`, other lanes unchanged."],["cmp_le_i32_m128_s","Low lane less than or equal to."],["cmp_le_i32_m128d_s","Low lane `f64` less than or equal to."],["cmp_le_mask_m128","Lanewise `a <= b`."],["cmp_le_mask_m128_s","Low lane `a <= b`, other lanes unchanged."],["cmp_le_mask_m128d","Lanewise `a <= b`."],["cmp_le_mask_m128d_s","Low lane `a <= b`, other lanes unchanged."],["cmp_lt_i32_m128_s","Low lane less than."],["cmp_lt_i32_m128d_s","Low lane `f64` less than."],["cmp_lt_mask_i16_m128i","Lanewise `a < b` with lanes as `i16`."],["cmp_lt_mask_i32_m128i","Lanewise `a < b` with lanes as `i32`."],["cmp_lt_mask_i8_m128i","Lanewise `a < b` with lanes as `i8`."],["cmp_lt_mask_m128","Lanewise `a < b`."],["cmp_lt_mask_m128_s","Low lane `a < b`, other lanes unchanged."],["cmp_lt_mask_m128d","Lanewise `a < b`."],["cmp_lt_mask_m128d_s","Low lane `a < b`, other lane unchanged."],["cmp_neq_i32_m128_s","Low lane not equal to."],["cmp_neq_i32_m128d_s","Low lane `f64` less than."],["cmp_neq_mask_m128","Lanewise `a != b`."],["cmp_neq_mask_m128_s","Low lane `a != b`, other lanes unchanged."],["cmp_neq_mask_m128d","Lanewise `a != b`."],["cmp_neq_mask_m128d_s","Low lane `a != b`, other lane unchanged."],["cmp_nge_mask_m128","Lanewise `!(a >= b)`."],["cmp_nge_mask_m128_s","Low lane `!(a >= b)`, other lanes unchanged."],["cmp_nge_mask_m128d","Lanewise `!(a >= b)`."],["cmp_nge_mask_m128d_s","Low lane `!(a >= b)`, other lane unchanged."],["cmp_ngt_mask_m128","Lanewise `!(a > b)`."],["cmp_ngt_mask_m128_s","Low lane `!(a > b)`, other lanes unchanged."],["cmp_ngt_mask_m128d","Lanewise `!(a > b)`."],["cmp_ngt_mask_m128d_s","Low lane `!(a > b)`, other lane unchanged."],["cmp_nle_mask_m128","Lanewise `!(a <= b)`."],["cmp_nle_mask_m128_s","Low lane `!(a <= b)`, other lanes unchanged."],["cmp_nle_mask_m128d","Lanewise `!(a <= b)`."],["cmp_nle_mask_m128d_s","Low lane `!(a <= b)`, other lane unchanged."],["cmp_nlt_mask_m128","Lanewise `!(a < b)`."],["cmp_nlt_mask_m128_s","Low lane `!(a < b)`, other lanes unchanged."],["cmp_nlt_mask_m128d","Lanewise `!(a < b)`."],["cmp_nlt_mask_m128d_s","Low lane `!(a < b)`, other lane unchanged."],["cmp_ordered_mask_m128","Lanewise `(!a.is_nan()) & (!b.is_nan())`."],["cmp_ordered_mask_m128_s","Low lane `(!a.is_nan()) & (!b.is_nan())`, other lanes unchanged."],["cmp_ordered_mask_m128d","Lanewise `(!a.is_nan()) & (!b.is_nan())`."],["cmp_ordered_mask_m128d_s","Low lane `(!a.is_nan()) & (!b.is_nan())`, other lane unchanged."],["cmp_unord_mask_m128","Lanewise `a.is_nan() | b.is_nan()`."],["cmp_unord_mask_m128_s","Low lane `a.is_nan() | b.is_nan()`, other lanes unchanged."],["cmp_unord_mask_m128d","Lanewise `a.is_nan() | b.is_nan()`."],["cmp_unord_mask_m128d_s","Low lane `a.is_nan() | b.is_nan()`, other lane unchanged."],["convert_i32_replace_m128_s","Convert `i32` to `f32` and replace the low lane of the input."],["convert_i32_replace_m128d_s","Convert `i32` to `f64` and replace the low lane of the input."],["convert_i64_replace_m128d_s","Convert `i64` to `f64` and replace the low lane of the input."],["convert_m128_s_replace_m128d_s","Converts the lower `f32` to `f64` and replace the low lane of the input"],["convert_m128d_s_replace_m128_s","Converts the low `f64` to `f32` and replaces the low lane of the input."],["convert_to_i32_m128i_from_m128","Rounds the `f32` lanes to `i32` lanes."],["convert_to_i32_m128i_from_m128d","Rounds the two `f64` lanes to the low two `i32` lanes."],["convert_to_m128_from_i32_m128i","Rounds the four `i32` lanes to four `f32` lanes."],["convert_to_m128_from_m128d","Rounds the two `f64` lanes to the low two `f32` lanes."],["convert_to_m128d_from_lower2_i32_m128i","Rounds the lower two `i32` lanes to two `f64` lanes."],["convert_to_m128d_from_lower2_m128","Rounds the two `f64` lanes to the low two `f32` lanes."],["copy_i64_m128i_s","Copy the low `i64` lane to a new register, upper bits 0."],["copy_replace_low_f64_m128d","Copies the `a` value and replaces the low lane with the low `b` value."],["div_m128","Lanewise `a / b`."],["div_m128_s","Low lane `a / b`, other lanes unchanged."],["div_m128d","Lanewise `a / b`."],["div_m128d_s","Lowest lane `a / b`, high lane unchanged."],["extract_i16_as_i32_m128i","Gets an `i16` value out of an `m128i`, returns as `i32`."],["get_f32_from_m128_s","Gets the low lane as an individual `f32` value."],["get_f64_from_m128d_s","Gets the lower lane as an `f64` value."],["get_i32_from_m128_s","Converts the low lane to `i32` and extracts as an individual value."],["get_i32_from_m128d_s","Converts the lower lane to an `i32` value."],["get_i32_from_m128i_s","Converts the lower lane to an `i32` value."],["get_i64_from_m128d_s","Converts the lower lane to an `i64` value."],["get_i64_from_m128i_s","Converts the lower lane to an `i64` value."],["insert_i16_from_i32_m128i","Inserts the low 16 bits of an `i32` value into an `m128i`."],["load_f32_m128_s","Loads the `f32` reference into the low lane of the register."],["load_f32_splat_m128","Loads the `f32` reference into all lanes of a register."],["load_f64_m128d_s","Loads the reference into the low lane of the register."],["load_f64_splat_m128d","Loads the `f64` reference into all lanes of a register."],["load_i64_m128i_s","Loads the low `i64` into a register."],["load_m128","Loads the reference into a register."],["load_m128d","Loads the reference into a register."],["load_m128i","Loads the reference into a register."],["load_replace_high_m128d","Loads the reference into a register, replacing the high lane."],["load_replace_low_m128d","Loads the reference into a register, replacing the low lane."],["load_reverse_m128","Loads the reference into a register with reversed order."],["load_reverse_m128d","Loads the reference into a register with reversed order."],["load_unaligned_m128","Loads the reference into a register."],["load_unaligned_m128d","Loads the reference into a register."],["load_unaligned_m128i","Loads the reference into a register."],["max_i16_m128i","Lanewise `max(a, b)` with lanes as `i16`."],["max_m128","Lanewise `max(a, b)`."],["max_m128_s","Low lane `max(a, b)`, other lanes unchanged."],["max_m128d","Lanewise `max(a, b)`."],["max_m128d_s","Low lane `max(a, b)`, other lanes unchanged."],["max_u8_m128i","Lanewise `max(a, b)` with lanes as `u8`."],["min_i16_m128i","Lanewise `min(a, b)` with lanes as `i16`."],["min_m128","Lanewise `min(a, b)`."],["min_m128_s","Low lane `min(a, b)`, other lanes unchanged."],["min_m128d","Lanewise `min(a, b)`."],["min_m128d_s","Low lane `min(a, b)`, other lanes unchanged."],["min_u8_m128i","Lanewise `min(a, b)` with lanes as `u8`."],["move_high_low_m128","Move the high lanes of `b` to the low lanes of `a`, other lanes unchanged."],["move_low_high_m128","Move the low lanes of `b` to the high lanes of `a`, other lanes unchanged."],["move_m128_s","Move the low lane of `b` to `a`, other lanes unchanged."],["move_mask_i8_m128i","Gathers the `i8` sign bit of each lane."],["move_mask_m128","Gathers the sign bit of each lane."],["move_mask_m128d","Gathers the sign bit of each lane."],["mul_i16_horizontal_add_m128i","Multiply `i16` lanes producing `i32` values, horizontal add pairs of `i32` values to produce the final output."],["mul_i16_keep_high_m128i","Lanewise `a * b` with lanes as `i16`, keep the high bits of the `i32` intermediates."],["mul_i16_keep_low_m128i","Lanewise `a * b` with lanes as `i16`, keep the low bits of the `i32` intermediates."],["mul_m128","Lanewise `a * b`."],["mul_m128_s","Low lane `a * b`, other lanes unchanged."],["mul_m128d","Lanewise `a * b`."],["mul_m128d_s","Lowest lane `a * b`, high lane unchanged."],["mul_u16_keep_high_m128i","Lanewise `a * b` with lanes as `u16`, keep the high bits of the `u32` intermediates."],["mul_widen_u32_odd_m128i","Multiplies the odd `u32` lanes and gives the widened (`u64`) results."],["pack_i16_to_i8_m128i","Saturating convert `i16` to `i8`, and pack the values."],["pack_i16_to_u8_m128i","Saturating convert `i16` to `u8`, and pack the values."],["pack_i32_to_i16_m128i","Saturating convert `i32` to `i16`, and pack the values."],["read_timestamp_counter","Reads the CPU’s timestamp counter value."],["read_timestamp_counter_p","Reads the CPU’s timestamp counter value and store the processor signature."],["reciprocal_m128","Lanewise `1.0 / a` approximation."],["reciprocal_m128_s","Low lane `1.0 / a` approximation, other lanes unchanged."],["reciprocal_sqrt_m128","Lanewise `1.0 / sqrt(a)` approximation."],["reciprocal_sqrt_m128_s","Low lane `1.0 / sqrt(a)` approximation, other lanes unchanged."],["set_i16_m128i","Sets the args into an `m128i`, first arg is the high lane."],["set_i32_m128i","Sets the args into an `m128i`, first arg is the high lane."],["set_i32_m128i_s","Set an `i32` as the low 32-bit lane of an `m128i`, other lanes blank."],["set_i64_m128i","Sets the args into an `m128i`, first arg is the high lane."],["set_i64_m128i_s","Set an `i64` as the low 64-bit lane of an `m128i`, other lanes blank."],["set_i8_m128i","Sets the args into an `m128i`, first arg is the high lane."],["set_m128","Sets the args into an `m128`, first arg is the high lane."],["set_m128_s","Sets the args into an `m128`, first arg is the high lane."],["set_m128d","Sets the args into an `m128d`, first arg is the high lane."],["set_m128d_s","Sets the args into the low lane of a `m128d`."],["set_reversed_i16_m128i","Sets the args into an `m128i`, first arg is the low lane."],["set_reversed_i32_m128i","Sets the args into an `m128i`, first arg is the low lane."],["set_reversed_i8_m128i","Sets the args into an `m128i`, first arg is the low lane."],["set_reversed_m128","Sets the args into an `m128`, first arg is the low lane."],["set_reversed_m128d","Sets the args into an `m128d`, first arg is the low lane."],["set_splat_i16_m128i","Splats the `i16` to all lanes of the `m128i`."],["set_splat_i32_m128i","Splats the `i32` to all lanes of the `m128i`."],["set_splat_i64_m128i","Splats the `i64` to both lanes of the `m128i`."],["set_splat_i8_m128i","Splats the `i8` to all lanes of the `m128i`."],["set_splat_m128","Splats the value to all lanes."],["set_splat_m128d","Splats the args into both lanes of the `m128d`."],["shl_all_u16_m128i","Shift all `u16` lanes to the left by the `count` in the lower `u64` lane."],["shl_all_u32_m128i","Shift all `u32` lanes to the left by the `count` in the lower `u64` lane."],["shl_all_u64_m128i","Shift all `u64` lanes to the left by the `count` in the lower `u64` lane."],["shl_imm_u16_m128i","Shifts all `u16` lanes left by an immediate."],["shl_imm_u32_m128i","Shifts all `u32` lanes left by an immediate."],["shl_imm_u64_m128i","Shifts both `u64` lanes left by an immediate."],["shr_all_i16_m128i","Shift each `i16` lane to the right by the `count` in the lower `i64` lane."],["shr_all_i32_m128i","Shift each `i32` lane to the right by the `count` in the lower `i64` lane."],["shr_all_u16_m128i","Shift each `u16` lane to the right by the `count` in the lower `u64` lane."],["shr_all_u32_m128i","Shift each `u32` lane to the right by the `count` in the lower `u64` lane."],["shr_all_u64_m128i","Shift each `u64` lane to the right by the `count` in the lower `u64` lane."],["shr_imm_i16_m128i","Shifts all `i16` lanes right by an immediate."],["shr_imm_i32_m128i","Shifts all `i32` lanes right by an immediate."],["shr_imm_u16_m128i","Shifts all `u16` lanes right by an immediate."],["shr_imm_u32_m128i","Shifts all `u32` lanes right by an immediate."],["shr_imm_u64_m128i","Shifts both `u64` lanes right by an immediate."],["shuffle_abi_f32_all_m128","Shuffle the `f32` lanes from `$a` and `$b` together using an immediate control value."],["shuffle_abi_f64_all_m128d","Shuffle the `f64` lanes from `$a` and `$b` together using an immediate control value."],["shuffle_ai_f32_all_m128i","Shuffle the `i32` lanes in `$a` using an immediate control value."],["shuffle_ai_i16_h64all_m128i","Shuffle the high `i16` lanes in `$a` using an immediate control value."],["shuffle_ai_i16_l64all_m128i","Shuffle the low `i16` lanes in `$a` using an immediate control value."],["sqrt_m128","Lanewise `sqrt(a)`."],["sqrt_m128_s","Low lane `sqrt(a)`, other lanes unchanged."],["sqrt_m128d","Lanewise `sqrt(a)`."],["sqrt_m128d_s","Low lane `sqrt(b)`, upper lane is unchanged from `a`."],["store_high_m128d_s","Stores the high lane value to the reference given."],["store_i64_m128i_s","Stores the value to the reference given."],["store_m128","Stores the value to the reference given."],["store_m128_s","Stores the low lane value to the reference given."],["store_m128d","Stores the value to the reference given."],["store_m128d_s","Stores the low lane value to the reference given."],["store_m128i","Stores the value to the reference given."],["store_reverse_m128","Stores the value to the reference given in reverse order."],["store_reversed_m128d","Stores the value to the reference given."],["store_splat_m128","Stores the low lane value to all lanes of the reference given."],["store_splat_m128d","Stores the low lane value to all lanes of the reference given."],["store_unaligned_m128","Stores the value to the reference given."],["store_unaligned_m128d","Stores the value to the reference given."],["store_unaligned_m128i","Stores the value to the reference given."],["sub_i16_m128i","Lanewise `a - b` with lanes as `i16`."],["sub_i32_m128i","Lanewise `a - b` with lanes as `i32`."],["sub_i64_m128i","Lanewise `a - b` with lanes as `i64`."],["sub_i8_m128i","Lanewise `a - b` with lanes as `i8`."],["sub_m128","Lanewise `a - b`."],["sub_m128_s","Low lane `a - b`, other lanes unchanged."],["sub_m128d","Lanewise `a - b`."],["sub_m128d_s","Lowest lane `a - b`, high lane unchanged."],["sub_saturating_i16_m128i","Lanewise saturating `a - b` with lanes as `i16`."],["sub_saturating_i8_m128i","Lanewise saturating `a - b` with lanes as `i8`."],["sub_saturating_u16_m128i","Lanewise saturating `a - b` with lanes as `u16`."],["sub_saturating_u8_m128i","Lanewise saturating `a - b` with lanes as `u8`."],["sum_of_u8_abs_diff_m128i","Compute “sum of `u8` absolute differences”."],["transpose_four_m128","Transpose four `m128` as if they were a 4x4 matrix."],["truncate_m128_to_m128i","Truncate the `f32` lanes to `i32` lanes."],["truncate_m128d_to_m128i","Truncate the `f64` lanes to the lower `i32` lanes (upper `i32` lanes 0)."],["truncate_to_i32_m128d_s","Truncate the lower lane into an `i32`."],["truncate_to_i64_m128d_s","Truncate the lower lane into an `i64`."],["unpack_high_i16_m128i","Unpack and interleave high `i16` lanes of `a` and `b`."],["unpack_high_i32_m128i","Unpack and interleave high `i32` lanes of `a` and `b`."],["unpack_high_i64_m128i","Unpack and interleave high `i64` lanes of `a` and `b`."],["unpack_high_i8_m128i","Unpack and interleave high `i8` lanes of `a` and `b`."],["unpack_high_m128","Unpack and interleave high lanes of `a` and `b`."],["unpack_high_m128d","Unpack and interleave high lanes of `a` and `b`."],["unpack_low_i16_m128i","Unpack and interleave low `i16` lanes of `a` and `b`."],["unpack_low_i32_m128i","Unpack and interleave low `i32` lanes of `a` and `b`."],["unpack_low_i64_m128i","Unpack and interleave low `i64` lanes of `a` and `b`."],["unpack_low_i8_m128i","Unpack and interleave low `i8` lanes of `a` and `b`."],["unpack_low_m128","Unpack and interleave low lanes of `a` and `b`."],["unpack_low_m128d","Unpack and interleave low lanes of `a` and `b`."],["zeroed_m128","All lanes zero."],["zeroed_m128d","Both lanes zero."],["zeroed_m128i","All lanes zero."]],"macro":[["round_op","Turns a round operator token to the correct constant value."]],"mod":[["naming_conventions","An explanation of the crate’s naming conventions."]],"struct":[["m128","The data for a 128-bit SSE register of four `f32` lanes."],["m128d","The data for a 128-bit SSE register of two `f64` values."],["m128i","The data for a 128-bit SSE register of integer data."],["m256","The data for a 256-bit AVX register of eight `f32` lanes."],["m256d","The data for a 256-bit AVX register of four `f64` values."],["m256i","The data for a 256-bit AVX register of integer data."]]});