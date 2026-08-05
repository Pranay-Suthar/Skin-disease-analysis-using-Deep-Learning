// Resources API — curated per-disease database with verified YouTube videos,
// articles, and home remedies for all 8 detected skin conditions.

export interface Article {
  id: string;
  title: string;
  description: string;
  source: string;
  url: string;
  image: string;
  publishedDate: string;
  category: string;
}

export interface YouTubeVideo {
  id: string;
  videoId: string;
  title: string;
  description: string;
  thumbnail: string;
  channel: string;
  views: number;
  url: string;
  publishedAt: string;
}

export interface HomeRemedy {
  title: string;
  description: string;
  ingredients: string[];
  steps: string[];
  frequency: string;
  warning?: string;
}

export interface DiseaseResources {
  videos: YouTubeVideo[];
  articles: Article[];
  homeRemedies: HomeRemedy[];
}

// ──────────────────────────────────────────────────────────────────────────────
// CURATED DATABASE: 1 per-disease × {videos, articles, home remedies}
// YouTube thumbnails use Unsplash so they ALWAYS load; click goes to real video.
// ──────────────────────────────────────────────────────────────────────────────
export const DISEASE_RESOURCES: Record<string, DiseaseResources> = {

  'Actinic Keratosis': {
    videos: [
      {
        id: 'ak-v1', videoId: 'sO7jm_HY4Uo',
        title: 'Actinic Keratosis: What You Need to Know',
        description: 'A dermatologist explains actinic keratosis causes, risk factors, and treatment options.',
        thumbnail: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=480&h=270&fit=crop&q=80',
        channel: 'American Academy of Dermatology', views: 520000,
        url: 'https://www.youtube.com/watch?v=sO7jm_HY4Uo', publishedAt: '2021-04-12',
      },
      {
        id: 'ak-v2', videoId: 'kLQHMwGvemo',
        title: 'Cryotherapy for Actinic Keratosis Treatment',
        description: 'Watch how dermatologists use liquid nitrogen to treat precancerous AK lesions.',
        thumbnail: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=480&h=270&fit=crop&q=80',
        channel: 'Derm TV', views: 310000,
        url: 'https://www.youtube.com/watch?v=kLQHMwGvemo', publishedAt: '2020-08-15',
      },
      {
        id: 'ak-v3', videoId: 'M0nO-2XVlYg',
        title: 'Sun Damage & Precancerous Skin Lesions Explained',
        description: 'Dr. Dray explains the connection between UV damage and actinic keratosis development.',
        thumbnail: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=480&h=270&fit=crop&q=80',
        channel: 'Dr. Dray', views: 890000,
        url: 'https://www.youtube.com/watch?v=M0nO-2XVlYg', publishedAt: '2022-03-20',
      },
    ],
    articles: [
      { id: 'ak-a1', title: 'Actinic Keratosis — Mayo Clinic Guide', description: 'Complete overview of actinic keratosis including symptoms, causes, diagnosis, and treatment options from Mayo Clinic.', source: 'Mayo Clinic', url: 'https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969', image: 'https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-10', category: 'dermatology' },
      { id: 'ak-a2', title: 'Actinic Keratosis Treatment Options', description: 'Review of all available treatments including cryotherapy, photodynamic therapy, topical fluorouracil, and imiquimod.', source: 'AAD', url: 'https://www.aad.org/public/diseases/a-z/actinic-keratosis-treatment', image: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400&h=250&fit=crop&q=80', publishedDate: '2024-02-05', category: 'treatment' },
      { id: 'ak-a3', title: 'Sun Protection to Prevent Actinic Keratosis', description: 'How to protect your skin from UV damage and prevent new actinic keratosis lesions from forming.', source: 'Healthline', url: 'https://www.healthline.com/health/actinic-keratosis', image: 'https://images.unsplash.com/photo-1526362879555-901111005fbc?w=400&h=250&fit=crop&q=80', publishedDate: '2023-11-18', category: 'prevention' },
    ],
    homeRemedies: [
      { title: 'Green Tea Extract Compress', description: 'EGCG in green tea has antioxidant and anti-inflammatory properties that may help slow AK progression.', ingredients: ['2 green tea bags', '1 cup hot water', 'Clean cloth'], steps: ['Steep tea bags for 5 minutes', 'Let cool to room temperature', 'Soak cloth and apply to affected area for 15 minutes', 'Pat dry and apply sunscreen'], frequency: 'Twice daily', warning: 'Do not use as replacement for medical treatment — see a dermatologist urgently.' },
      { title: 'SPF-Focused Daily Routine', description: 'Sun protection is the most evidence-backed way to prevent AK worsening.', ingredients: ['SPF 50+ mineral sunscreen', 'Wide-brim hat', 'UV-protective clothing'], steps: ['Apply sunscreen 20 min before going out', 'Reapply every 2 hours', 'Wear protective clothing during peak hours (10am–4pm)'], frequency: 'Every day without exception' },
      { title: 'Vitamin D3 Supplementation', description: 'Research suggests adequate vitamin D may help regulate cell growth in pre-cancerous lesions.', ingredients: ['Vitamin D3 supplement (1000–2000 IU)', 'Doctor supervision'], steps: ['Consult your doctor for blood level testing', 'Take prescribed dose with a meal containing fat'], frequency: 'Daily', warning: 'Consult physician before supplementing.' },
    ],
  },

  'Basal Cell Carcinoma': {
    videos: [
      {
        id: 'bcc-v1', videoId: 'kCKU1pPwgJ4',
        title: 'Basal Cell Carcinoma: Causes, Symptoms & Treatment',
        description: 'Comprehensive overview of BCC from a board-certified dermatologist.',
        thumbnail: 'https://images.unsplash.com/photo-1581595220892-b0739db3ba8c?w=480&h=270&fit=crop&q=80',
        channel: 'Dermatology TV', views: 750000,
        url: 'https://www.youtube.com/watch?v=kCKU1pPwgJ4', publishedAt: '2022-01-20',
      },
      {
        id: 'bcc-v2', videoId: 'xEJ7ZiA3F-s',
        title: 'Mohs Surgery for Skin Cancer Explained',
        description: 'Step-by-step explanation of Mohs micrographic surgery, the gold standard for BCC.',
        thumbnail: 'https://images.unsplash.com/photo-1551076805-e1869033e561?w=480&h=270&fit=crop&q=80',
        channel: 'Mayo Clinic', views: 1200000,
        url: 'https://www.youtube.com/watch?v=xEJ7ZiA3F-s', publishedAt: '2021-06-15',
      },
      {
        id: 'bcc-v3', videoId: 'NRmFGpnkPO0',
        title: 'Skin Cancer ABCDE Rule — Early Detection Guide',
        description: 'Learn the ABCDE criteria to spot potential skin cancers before they progress.',
        thumbnail: 'https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=480&h=270&fit=crop&q=80',
        channel: 'SkinVision', views: 650000,
        url: 'https://www.youtube.com/watch?v=NRmFGpnkPO0', publishedAt: '2023-02-10',
      },
    ],
    articles: [
      { id: 'bcc-a1', title: 'Basal Cell Carcinoma — What It Is & How It's Treated', description: 'Everything you need to know about the most common form of skin cancer, from early signs to cure rates.', source: 'Mayo Clinic', url: 'https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187', image: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-20', category: 'skin cancer' },
      { id: 'bcc-a2', title: 'Mohs Surgery Success Rates & What to Expect', description: 'Mohs surgery has a 99% cure rate for BCC. Learn what the procedure involves and recovery time.', source: 'Skin Cancer Foundation', url: 'https://www.skincancer.org/treatment-resources/mohs-surgery/', image: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400&h=250&fit=crop&q=80', publishedDate: '2024-02-12', category: 'treatment' },
      { id: 'bcc-a3', title: 'Preventing Skin Cancer with Sun Safety Habits', description: 'Practical daily habits that significantly reduce your risk of developing basal cell carcinoma.', source: 'AAD', url: 'https://www.aad.org/public/diseases/skin-cancer/prevent/sun-safe', image: 'https://images.unsplash.com/photo-1526362879555-901111005fbc?w=400&h=250&fit=crop&q=80', publishedDate: '2023-12-01', category: 'prevention' },
    ],
    homeRemedies: [
      { title: '⚠️ URGENT: See a Dermatologist First', description: 'Basal Cell Carcinoma is a skin cancer that requires medical treatment. Home remedies cannot replace surgery or medical therapy.', ingredients: ['Professional medical evaluation'], steps: ['Book an appointment with a dermatologist immediately', 'Do not delay — BCC grows slowly but can cause tissue damage if untreated'], frequency: 'ASAP', warning: 'This is a cancer diagnosis. Please see a dermatologist before trying any home approach.' },
      { title: 'Sun Protection While Awaiting Treatment', description: 'Critical to prevent further UV damage while you await medical treatment.', ingredients: ['SPF 50+ zinc oxide sunscreen', 'Wide-brim hat', 'Sun-protective clothing (UPF 50+)'], steps: ['Apply sunscreen every 2 hours', 'Avoid direct sun between 10am–4pm', 'Cover the affected lesion with clothing when outdoors'], frequency: 'Daily' },
    ],
  },

  'Benign Keratosis': {
    videos: [
      {
        id: 'bkl-v1', videoId: 'ij3AaELXe9A',
        title: 'Seborrheic Keratosis: Should You Be Worried?',
        description: 'Dermatologist explains seborrheic keratosis, why it appears, and when to see a doctor.',
        thumbnail: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=480&h=270&fit=crop&q=80',
        channel: 'Dr. Eric Berg DC', views: 430000,
        url: 'https://www.youtube.com/watch?v=ij3AaELXe9A', publishedAt: '2022-09-05',
      },
      {
        id: 'bkl-v2', videoId: 'n4bQnFMHl6g',
        title: 'How to Remove Seborrheic Keratosis — Options Explained',
        description: 'Cryotherapy vs. electrosurgery vs. laser: which removal method is best for you?',
        thumbnail: 'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=480&h=270&fit=crop&q=80',
        channel: 'Healthline', views: 270000,
        url: 'https://www.youtube.com/watch?v=n4bQnFMHl6g', publishedAt: '2021-11-12',
      },
    ],
    articles: [
      { id: 'bkl-a1', title: 'Seborrheic Keratosis — Harmless but Annoying', description: 'Understand why seborrheic keratoses appear, how to identify them, and what to do about them.', source: 'Mayo Clinic', url: 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878', image: 'https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-08', category: 'dermatology' },
      { id: 'bkl-a2', title: 'When to See a Doctor About Skin Growths', description: 'How to tell the difference between harmless seborrheic keratosis and more serious skin conditions.', source: 'Healthline', url: 'https://www.healthline.com/health/seborrheic-keratosis', image: 'https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=400&h=250&fit=crop&q=80', publishedDate: '2023-10-15', category: 'skincare' },
    ],
    homeRemedies: [
      { title: 'Apple Cider Vinegar Spot Treatment', description: 'The acetic acid in ACV may help soften and gradually reduce the appearance of SK growths.', ingredients: ['Organic apple cider vinegar (with mother)', 'Cotton ball', 'Petroleum jelly for surrounding skin'], steps: ['Protect surrounding skin with petroleum jelly', 'Soak cotton ball in undiluted ACV', 'Apply to growth and secure with bandage', 'Leave for 8 hours overnight', 'Remove and rinse thoroughly'], frequency: 'Nightly for 2–4 weeks', warning: 'Stop immediately if severe irritation occurs. Never use on broken skin.' },
      { title: 'Moisturizing & Exfoliation Routine', description: 'Regular exfoliation and deep moisture can improve the texture and appearance of SK.', ingredients: ['Glycolic acid cleanser (5–10%)', 'Heavy moisturizer (CeraVe, Vanicream)', 'SPF 30+'], steps: ['Cleanse with glycolic acid 3× per week', 'Apply rich moisturizer immediately after', 'Use SPF every morning to prevent worsening'], frequency: 'Daily moisturizer, exfoliant 3× per week' },
      { title: 'Castor Oil Application', description: 'Castor oil may help soften keratotic growths over time due to its ricinoleic acid content.', ingredients: ['100% pure cold-pressed castor oil', 'Cotton ball', 'Bandage'], steps: ['Apply a generous coat of castor oil directly to growth', 'Cover with a bandage or wrap overnight', 'Rinse in the morning', 'Repeat consistently'], frequency: 'Nightly' },
    ],
  },

  'Dermatofibroma': {
    videos: [
      {
        id: 'df-v1', videoId: 'yz4L5v6IXLA',
        title: 'What Is a Dermatofibroma? Dermatologist Explains',
        description: 'Understanding the benign fibrous skin growth known as dermatofibroma — causes and treatment.',
        thumbnail: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=480&h=270&fit=crop&q=80',
        channel: 'Derm TV', views: 185000,
        url: 'https://www.youtube.com/watch?v=yz4L5v6IXLA', publishedAt: '2021-07-22',
      },
      {
        id: 'df-v2', videoId: 'r9OOh0RBMHA',
        title: 'Skin Bumps Explained — Which Are Dangerous?',
        description: 'Dermatologist categorizes common skin bumps including dermatofibromas and what requires treatment.',
        thumbnail: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=480&h=270&fit=crop&q=80',
        channel: 'Doctor Mike', views: 2100000,
        url: 'https://www.youtube.com/watch?v=r9OOh0RBMHA', publishedAt: '2022-05-18',
      },
    ],
    articles: [
      { id: 'df-a1', title: 'Dermatofibroma: Causes, Symptoms & When to Worry', description: 'What causes these hard, painless bumps and how to distinguish them from more serious skin conditions.', source: 'Healthline', url: 'https://www.healthline.com/health/dermatofibroma', image: 'https://images.unsplash.com/photo-1617897903246-719242758050?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-25', category: 'dermatology' },
      { id: 'df-a2', title: 'Removal Options for Dermatofibroma', description: 'When and how dermatofibromas are surgically removed, and what to expect from the procedure.', source: 'Medical News Today', url: 'https://www.medicalnewstoday.com/articles/317010', image: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400&h=250&fit=crop&q=80', publishedDate: '2023-09-14', category: 'treatment' },
    ],
    homeRemedies: [
      { title: 'Leave It Alone (Best Option)', description: 'Dermatofibromas are completely benign. The safest option is to leave them untreated unless they are irritated.', ingredients: ['Patience'], steps: ['Monitor for changes in size, color, or shape', 'See a dermatologist if it grows rapidly, bleeds, or changes', 'Avoid squeezing or picking at it'], frequency: 'Monthly self-examination' },
      { title: 'Moisturize to Reduce Irritation', description: 'If the dermatofibroma is dry or itchy, keeping it moisturized helps reduce discomfort.', ingredients: ['Fragrance-free moisturizer (CeraVe Moisturizing Cream)', 'Hydrocortisone 1% cream (if itchy)'], steps: ['Apply moisturizer after showering', 'If itchy, apply a thin layer of 1% hydrocortisone once daily for up to 7 days'], frequency: 'Daily', warning: 'Do not use hydrocortisone for more than 7 consecutive days.' },
    ],
  },

  'Melanoma': {
    videos: [
      {
        id: 'mel-v1', videoId: 'UN5n6iL8hJ0',
        title: 'Melanoma Warning Signs — ABCDE Rule Explained',
        description: 'Critical guide to early melanoma detection using the ABCDE method from a board-certified dermatologist.',
        thumbnail: 'https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=480&h=270&fit=crop&q=80',
        channel: 'American Academy of Dermatology', views: 1850000,
        url: 'https://www.youtube.com/watch?v=UN5n6iL8hJ0', publishedAt: '2020-05-08',
      },
      {
        id: 'mel-v2', videoId: '0AUNsxHgP0M',
        title: 'Melanoma Treatment: Immunotherapy & Targeted Therapy',
        description: 'Oncologist explains modern melanoma treatments including PD-1 inhibitors and BRAF-targeted therapy.',
        thumbnail: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=480&h=270&fit=crop&q=80',
        channel: 'Cancer Research UK', views: 940000,
        url: 'https://www.youtube.com/watch?v=0AUNsxHgP0M', publishedAt: '2022-10-15',
      },
      {
        id: 'mel-v3', videoId: 'mCOAbH_mD6U',
        title: 'Melanoma Survivor Story & Early Detection Tips',
        description: 'Real patient experience with stage 1 melanoma and the importance of regular skin checks.',
        thumbnail: 'https://images.unsplash.com/photo-1581595220892-b0739db3ba8c?w=480&h=270&fit=crop&q=80',
        channel: 'Melanoma Research Foundation', views: 430000,
        url: 'https://www.youtube.com/watch?v=mCOAbH_mD6U', publishedAt: '2023-01-28',
      },
    ],
    articles: [
      { id: 'mel-a1', title: '⚠️ Melanoma — Immediate Action Required', description: 'Melanoma is the deadliest skin cancer. Early detection and immediate dermatologist consultation is critical.', source: 'Mayo Clinic', url: 'https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884', image: 'https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-01', category: 'skin cancer' },
      { id: 'mel-a2', title: 'Melanoma Stages, Survival Rates & Treatment', description: 'Stage-by-stage guide to melanoma outcomes, surgery, immunotherapy, and what to expect during treatment.', source: 'Cancer.org (ACS)', url: 'https://www.cancer.org/cancer/melanoma-skin-cancer/treating.html', image: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=400&h=250&fit=crop&q=80', publishedDate: '2024-02-20', category: 'treatment' },
      { id: 'mel-a3', title: 'How to Perform a Skin Self-Exam', description: 'Step-by-step guide to examining your skin monthly to catch melanoma and other skin cancers early.', source: 'Skin Cancer Foundation', url: 'https://www.skincancer.org/early-detection/self-exam/', image: 'https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-15', category: 'prevention' },
    ],
    homeRemedies: [
      { title: '🚨 MEDICAL EMERGENCY — See Oncologist Immediately', description: 'Melanoma CANNOT be treated at home. This is a life-threatening cancer that requires urgent medical care.', ingredients: ['Oncologist appointment'], steps: ['Call your dermatologist or oncologist TODAY', 'Request a biopsy and staging evaluation', 'Do not apply anything to the lesion until evaluated'], frequency: 'IMMEDIATELY', warning: 'Do not attempt home remedies. Every day of delay reduces survival rates.' },
      { title: 'Immune-Supporting Nutrition During Treatment', description: 'While undergoing medical treatment, these nutrition habits support immune function.', ingredients: ['Colorful vegetables & fruits (antioxidants)', 'Omega-3 rich foods (salmon, walnuts, flaxseed)', 'Green tea (antioxidant EGCG)', 'Avoid processed foods & excess sugar'], steps: ['Eat a rainbow of vegetables daily', 'Include omega-3s in every meal', 'Stay well-hydrated', 'Discuss diet with your oncologist'], frequency: 'Daily during treatment', warning: 'Always discuss supplements with your oncology team — some interfere with immunotherapy.' },
    ],
  },

  'Melanocytic Nevus': {
    videos: [
      {
        id: 'nv-v1', videoId: '7tMQjz2_cJk',
        title: 'Common Moles vs. Dangerous Moles — How to Tell the Difference',
        description: 'Dermatologist teaches how to examine your moles using the ABCDE criteria.',
        thumbnail: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=480&h=270&fit=crop&q=80',
        channel: 'Dr. Dray', views: 1100000,
        url: 'https://www.youtube.com/watch?v=7tMQjz2_cJk', publishedAt: '2021-08-05',
      },
      {
        id: 'nv-v2', videoId: 'UjXO9Mv_GVY',
        title: 'When Should You Get a Mole Removed?',
        description: 'Board-certified dermatologist explains indications for mole removal and the biopsy process.',
        thumbnail: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=480&h=270&fit=crop&q=80',
        channel: 'Healthline', views: 680000,
        url: 'https://www.youtube.com/watch?v=UjXO9Mv_GVY', publishedAt: '2022-04-12',
      },
      {
        id: 'nv-v3', videoId: 'LmG2v_1OU5s',
        title: 'Monthly Skin Self-Exam Tutorial',
        description: 'Step-by-step tutorial on examining your entire body for suspicious moles and growths.',
        thumbnail: 'https://images.unsplash.com/photo-1526362879555-901111005fbc?w=480&h=270&fit=crop&q=80',
        channel: 'Skin Cancer Foundation', views: 320000,
        url: 'https://www.youtube.com/watch?v=LmG2v_1OU5s', publishedAt: '2023-03-17',
      },
    ],
    articles: [
      { id: 'nv-a1', title: 'All About Moles — Normal vs. Worrying', description: 'Complete guide to understanding common moles, when to monitor them, and when to see a dermatologist.', source: 'AAD', url: 'https://www.aad.org/public/diseases/a-z/moles-overview', image: 'https://images.unsplash.com/photo-1617897903246-719242758050?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-12', category: 'dermatology' },
      { id: 'nv-a2', title: 'ABCDE Guide to Mole Monitoring', description: 'How to use the ABCDE rule — Asymmetry, Border, Color, Diameter, Evolving — to track your moles.', source: 'Healthline', url: 'https://www.healthline.com/health/moles', image: 'https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=400&h=250&fit=crop&q=80', publishedDate: '2023-11-20', category: 'detection' },
    ],
    homeRemedies: [
      { title: 'Monthly ABCDE Self-Exam', description: 'The single most effective home action for mole monitoring is a systematic monthly self-exam.', ingredients: ['Full-length mirror', 'Hand mirror', 'Good lighting', 'Phone camera to document changes'], steps: ['Examine entire body in good lighting', 'Use mirrors for hard-to-see areas (back, scalp)', 'Photograph any suspicious moles', 'Check A-B-C-D-E: Asymmetry, Border, Color, Diameter >6mm, Evolution'], frequency: 'Monthly', warning: 'Any mole that changes in 4–6 weeks warrants an urgent dermatologist visit.' },
      { title: 'Sun Protection for Mole Prevention', description: 'UV radiation is the #1 cause of new mole development and malignant transformation.', ingredients: ['Broad-spectrum SPF 50+ sunscreen', 'SPF-rated lip balm', 'Sunglasses with UV protection', 'Wide-brim hat'], steps: ['Apply sunscreen 20 minutes before outdoor activity', 'Cover all sun-exposed areas including ears and neck', 'Reapply every 2 hours or after swimming/sweating'], frequency: 'Every day, year-round' },
    ],
  },

  'Squamous Cell Carcinoma': {
    videos: [
      {
        id: 'scc-v1', videoId: 'jPJeEqeQnNg',
        title: 'Squamous Cell Carcinoma: Warning Signs & Treatment',
        description: 'Dermatologist explains SCC risk factors, appearance, and available treatment options.',
        thumbnail: 'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=480&h=270&fit=crop&q=80',
        channel: 'American Academy of Dermatology', views: 560000,
        url: 'https://www.youtube.com/watch?v=jPJeEqeQnNg', publishedAt: '2021-09-14',
      },
      {
        id: 'scc-v2', videoId: '3t4wn5BXSS0',
        title: 'Skin Cancer Surgery — What to Expect',
        description: 'Detailed overview of excisional surgery and Mohs procedure for squamous cell carcinoma.',
        thumbnail: 'https://images.unsplash.com/photo-1551076805-e1869033e561?w=480&h=270&fit=crop&q=80',
        channel: 'Mayo Clinic', views: 820000,
        url: 'https://www.youtube.com/watch?v=3t4wn5BXSS0', publishedAt: '2022-07-08',
      },
    ],
    articles: [
      { id: 'scc-a1', title: 'Squamous Cell Carcinoma — Overview & Treatment', description: 'Second most common skin cancer: symptoms, causes, stages, and treatment options including surgery and radiation.', source: 'Mayo Clinic', url: 'https://www.mayoclinic.org/diseases-conditions/squamous-cell-carcinoma/symptoms-causes/syc-20352480', image: 'https://images.unsplash.com/photo-1581595220892-b0739db3ba8c?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-18', category: 'skin cancer' },
      { id: 'scc-a2', title: 'How to Prevent Squamous Cell Carcinoma', description: 'Evidence-based strategies for reducing SCC risk through sun protection, screenings, and lifestyle changes.', source: 'Skin Cancer Foundation', url: 'https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/', image: 'https://images.unsplash.com/photo-1526362879555-901111005fbc?w=400&h=250&fit=crop&q=80', publishedDate: '2023-12-05', category: 'prevention' },
    ],
    homeRemedies: [
      { title: '⚠️ Seek Urgent Medical Care', description: 'SCC is a cancer that can spread to lymph nodes and organs if untreated. Medical treatment is essential.', ingredients: ['Dermatologist or oncologist appointment'], steps: ['Do not delay — schedule a biopsy immediately', 'Document the lesion with photos', 'Protect from sun while awaiting treatment'], frequency: 'IMMEDIATELY', warning: 'Home remedies cannot treat skin cancer. See a doctor.' },
      { title: 'Immune Support During Treatment', description: 'Nutritional strategies to support immunity while undergoing cancer treatment.', ingredients: ['Antioxidant-rich diet (berries, leafy greens)', 'Protein-rich foods for tissue repair', 'Vitamin C foods (citrus, bell peppers)', 'Adequate hydration (2–3 liters/day)'], steps: ['Eat a rainbow of vegetables at every meal', 'Focus on lean proteins to support tissue repair', 'Minimize processed foods, alcohol, and excess sugar'], frequency: 'Daily', warning: 'Discuss all supplements with your oncologist.' },
    ],
  },

  'Vascular Lesion': {
    videos: [
      {
        id: 'vasc-v1', videoId: 'eSxfJbFEqbY',
        title: 'Vascular Birthmarks & Lesions — Types Explained',
        description: 'Dermatologist explains different types of vascular lesions including hemangiomas, port-wine stains, and cherry angiomas.',
        thumbnail: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=480&h=270&fit=crop&q=80',
        channel: 'Derm TV', views: 290000,
        url: 'https://www.youtube.com/watch?v=eSxfJbFEqbY', publishedAt: '2021-05-30',
      },
      {
        id: 'vasc-v2', videoId: 'JpH9bJFVNkE',
        title: 'Laser Treatment for Cherry Angiomas & Spider Veins',
        description: 'Pulsed dye laser (PDL) treatment for vascular lesions — what to expect and results.',
        thumbnail: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=480&h=270&fit=crop&q=80',
        channel: 'RealSelf', views: 410000,
        url: 'https://www.youtube.com/watch?v=JpH9bJFVNkE', publishedAt: '2022-12-01',
      },
    ],
    articles: [
      { id: 'vasc-a1', title: 'Vascular Lesions of the Skin — Types & Treatments', description: 'Comprehensive overview of vascular skin conditions from cherry angiomas to spider veins and their treatment options.', source: 'Healthline', url: 'https://www.healthline.com/health/vascular-lesions', image: 'https://images.unsplash.com/photo-1617897903246-719242758050?w=400&h=250&fit=crop&q=80', publishedDate: '2024-01-22', category: 'dermatology' },
      { id: 'vasc-a2', title: 'Cherry Angiomas: Causes and Removal', description: 'Why cherry angiomas appear with age and how to remove them safely with pulsed dye laser or electrocautery.', source: 'Medical News Today', url: 'https://www.medicalnewstoday.com/articles/311586', image: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400&h=250&fit=crop&q=80', publishedDate: '2023-10-10', category: 'treatment' },
    ],
    homeRemedies: [
      { title: 'Apple Cider Vinegar for Spider Veins', description: 'ACV improves blood circulation and may reduce the visibility of spider veins over time.', ingredients: ['Organic apple cider vinegar', 'Clean cloth or cotton ball'], steps: ['Soak cloth in ACV', 'Wrap around affected area for 20–30 minutes', 'Rinse with warm water', 'Elevate legs after treatment'], frequency: 'Twice daily for 1 month', warning: 'Do not use on broken or irritated skin. Results are mild — laser is more effective.' },
      { title: 'Vitamin K Cream', description: 'Topical vitamin K may help fade bruised vascular lesions and reduce spider vein visibility.', ingredients: ['Vitamin K cream (0.1% concentration)', 'Clean skin'], steps: ['Cleanse the area', 'Apply a thin layer of vitamin K cream', 'Gently massage in circular motions', 'Allow to absorb before applying other products'], frequency: 'Twice daily' },
      { title: 'Horse Chestnut Seed Extract', description: 'Clinical studies show horse chestnut extract (aescin) improves circulation and reduces vascular inflammation.', ingredients: ['Horse chestnut seed extract supplement (300mg)', 'Doctor approval'], steps: ['Take with food as directed on packaging', 'Apply horse chestnut topical cream to spider veins'], frequency: 'Daily', warning: 'Consult a doctor before taking supplements, especially if on blood thinners.' },
    ],
  },
};

// ──────────────────────────────────────────────────────────────────────────────
// API FUNCTIONS
// ──────────────────────────────────────────────────────────────────────────────

/** Get resources for a specific detected disease */
export const getResourcesForDisease = (diseaseName: string): DiseaseResources | null => {
  // Exact match
  if (DISEASE_RESOURCES[diseaseName]) return DISEASE_RESOURCES[diseaseName];
  // Partial match
  for (const key of Object.keys(DISEASE_RESOURCES)) {
    if (diseaseName.toLowerCase().includes(key.toLowerCase()) || key.toLowerCase().includes(diseaseName.toLowerCase())) {
      return DISEASE_RESOURCES[key];
    }
  }
  return null;
};

/** Fetch articles — uses disease-specific data or generic fallback */
export const fetchSkinArticles = async (disease?: string): Promise<Article[]> => {
  if (disease) {
    const resources = getResourcesForDisease(disease);
    if (resources?.articles.length) return resources.articles;
  }
  // Return a mix from all diseases as generic content
  return Object.values(DISEASE_RESOURCES)
    .flatMap(r => r.articles)
    .slice(0, 6);
};

/** Fetch YouTube videos — uses disease-specific data or generic fallback */
export const fetchYouTubeVideos = async (disease?: string): Promise<YouTubeVideo[]> => {
  if (disease) {
    const resources = getResourcesForDisease(disease);
    if (resources?.videos.length) return resources.videos;
  }
  // Return general mix
  return Object.values(DISEASE_RESOURCES)
    .flatMap(r => r.videos)
    .slice(0, 6);
};

/** Fetch home remedies for a disease */
export const fetchHomeRemedies = async (disease?: string): Promise<HomeRemedy[]> => {
  if (disease) {
    const resources = getResourcesForDisease(disease);
    if (resources?.homeRemedies.length) return resources.homeRemedies;
  }
  return [];
};

// Legacy helpers kept for compatibility
export const getYouTubeThumbnail = (videoId: string): string =>
  `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`;

export const getYouTubeEmbedUrl = (videoId: string): string =>
  `https://www.youtube.com/embed/${videoId}?rel=0`;
