pub trait ButcherTableau {
    const ORDER: usize;
    const STAGES: usize;
    const A: [[f32; self::STAGES]; self::STAGES];
    const B: [f32; self::STAGES];
    const C: [f32; self::STAGES];
}

impl ButcherTableau for DormandPrince54 {
    const ORDER = 5;
    const STAGES = 4;
    const A = [[0.2,3. / 40.,44. / 45.,19372. / 6561,9017. / 3168.,35. / 384.],
[0,9. / 40.,-56. / 15.,-25360. / 2187.,-355. / 33.,0],
[0,0,32. / 9.,64448. / 6561.,46732. / 5247.,500. / 1113.],
[0,0,0,-212. / 729.,49. / 176.,125. / 192.],
[0,0,0,0,-5103. / 18656.,-2187. / 6784.]
[0,0,0,0,0,11. / 84.]
];
    const B = [35. / 384.,500. / 1113.,125. / 192.,-2187. / 6784.,11. / 84.];
    const C = [0.2,0.3,0.8,8. / 9.,1,1];
}
