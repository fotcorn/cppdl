#pragma once

#include "tensor.h"

tensor<float> l1w({16, 2});
tensor<float> l1b({16, 1});

tensor<float> l2w({16, 16});
tensor<float> l2b({16, 1});

tensor<float> l3w({1, 16});
tensor<float> l3b({1, 1});

Neuron<2> LAYER0[16] = {
    {{-0.07300148988542916f, 1.1359772893875315}, -0.5621034413168672, true},
    {{-0.9184968896309132f, 0.37166880762758114}, -0.21715450580438525, true},
    {{0.7848369578765014f, -0.44020510733312473}, -0.6172867756094845, true},
    {{1.2519853895296171f, -0.7345578819562889}, -1.0546724724492367, true},
    {{0.9322719919911607f, 0.8439711942218322}, -0.2998738471213825, true},
    {{-1.0226085714467998f, -0.30339253176165865}, -0.019409979025763364, true},
    {{-0.5938781045294802f, -0.41216193368003756}, 0.001991125455726379, true},
    {{0.1360889448952135f, -0.04159379258786372}, -0.3797827423100108, true},
    {{-0.5555595388427368f, -0.7613102105434105}, 0.01965681709028249, true},
    {{-0.32223265849263494f, -0.07131424752077535}, -0.05114469522506509, true},
    {{-0.425688420872239f, -0.735572571961662}, -0.3609046387974498, true},
    {{-0.3512210603816492f, -1.0781450235562995}, -0.21982282282285984, true},
    {{-0.6177561426615421f, -0.7829836543245512}, 0.6943240411815667, true},
    {{-0.6614113744756367f, 0.12203045504292674}, -0.2759995148153068, true},
    {{-0.10695350092324434f, 0.03628832450627965}, -0.6026166067150676, true},
    {{0.3293045969659193f, 0.22231450023655533}, -0.37408025277912255, true},
};
Neuron<16> LAYER1[16] = {
    {{-0.4594134245302827f, -0.24889608684806655f, 0.7828994545353823f,
      -0.2803842833249532f, 0.33534594905671755f, -0.17506722408494835f,
      0.15333784755597032f, -0.506561452252456f, 0.9926448305262807f,
      -0.648036611644507f, 0.3564418713387753f, -0.5280094563088518f,
      -0.11678095594298954f, -0.46632709257841465f, 0.2674678434096645f,
      -0.2596218875281965},
     -0.23233825586272566,
     true},
    {{1.0788853780923333f, 0.2902007715461422f, 0.4629429042921407f,
      -0.9593438899153501f, 0.4326101622305363f, -0.45310015028812217f,
      0.8809462547139657f, 0.5894880570219987f, -0.8274829911965098f,
      0.5738060094155082f, -0.5675755557912614f, -0.9159064268584068f,
      -0.3158595261593728f, -0.7859557142183631f, 0.0956103146289978f,
      -0.49359947983194497},
     -0.03147441597629667,
     true},
    {{-0.03536308379929487f, 0.8605351448372405f, -0.24673183760970321f,
      -0.17068976342960054f, -0.9184529318718953f, -0.6182269334442946f,
      0.11175889919987235f, -0.9582831722518625f, -0.060677258865719924f,
      -0.7988141112811177f, 0.49137977107071273f, -0.01629607789965807f,
      -0.09128741319212783f, -0.37889463073161317f, -0.9594606718259371f,
      0.6478418559475245},
     -0.024918318756069,
     true},
    {{0.387110854189906f, -0.7853140578601336f, 0.280699545143892f,
      -0.2280036010504288f, 0.07243509956267083f, -0.5778185511588422f,
      -0.06076386331791133f, -0.15551745180715407f, 0.166435272188546f,
      -0.45573652838964174f, 0.8812969215389762f, 0.8824435478618449f,
      0.14965926519242823f, -0.059609303930116875f, -0.2847281315240958f,
      -0.5507969571101441},
     -0.0023033376933657957,
     true},
    {{-0.7846309693479374f, -0.38994238659728453f, -0.1640404427378965f,
      1.0785811102374712f, -0.3651656202918221f, -0.47612218608409107f,
      -0.5204611467369871f, 0.5700915587742817f, -0.8640779913357384f,
      -0.7559188177167417f, -0.8600673085839077f, -0.8876878220139209f,
      0.8123080804428632f, -0.6227866727693314f, -0.011657767769431754f,
      1.1647706136714568},
     0.31776542872245406,
     true},
    {{-0.2360778719727591f, 0.40443843364760584f, 0.5046011723382409f,
      0.6123310798667949f, 0.3600914516534681f, 0.7496907670899621f,
      -0.9303673805586091f, 0.5075591345536923f, -0.4891996404675676f,
      -0.3081399407413638f, -0.22192558526074357f, 0.35217424703196515f,
      -0.4385819097351341f, -0.13965737163121084f, 0.04913405212756727f,
      -0.726556154346322},
     -0.3059969840237853,
     true},
    {{-1.0168791557884078f, 0.7207194274742277f, -0.8447772641078349f,
      0.38510233151543183f, -0.06980854103291691f, -0.6198681879313822f,
      -0.1680208702609925f, -0.9208931721638536f, -0.8312336424396704f,
      0.31026422307770046f, -0.57127721744716f, 0.7961211754376706f,
      -0.013425143514702908f, 0.5774609321306213f, 0.4977046242333022f,
      -0.3973380566140537},
     -0.2406041428580013,
     true},
    {{-0.2267314044229651f, 0.2302821031138059f, -0.21806803403752625f,
      -0.010945868310563028f, -1.0258439059941593f, -0.7934633141103393f,
      -0.5959702304292354f, 0.7428312793693045f, 0.36594920903732514f,
      -0.7120021790915558f, -0.8688856702857876f, -0.9374425499257847f,
      0.823491631111517f, 0.4796986573512101f, -0.3230684695679463f,
      0.2645634008330233},
     -0.3463862161509954,
     true},
    {{0.3362165198580005f, -0.2198112651745276f, 0.7470137847877223f,
      0.6384795574977563f, 0.19193535791464253f, -0.7305133164044797f,
      -0.09446695447217696f, 0.43719624221656755f, 0.6691703300700682f,
      -0.5639290373244173f, -0.7469406752341783f, -0.1479655772608588f,
      -0.7146146098665642f, -0.7082021676629808f, 0.05147516132557065f,
      0.6465863898105879},
     -0.23252347561284817,
     true},
    {{-0.9383451440115661f, 0.4206068459309924f, 0.40523407560219865f,
      0.47413507696422474f, -0.6699587116241859f, 0.23471201855554638f,
      -0.43778322987136387f, 0.3990168327639217f, 0.34771045559367825f,
      -0.7241986131010089f, 0.20307067918259308f, 0.3165346677113996f,
      -0.5456260752133646f, -0.3200026908659159f, 0.61208687035459f,
      0.6808646312230937},
     -0.30019201507121773,
     true},
    {{0.882710616637309f, -0.5111231726257524f, 0.663595560105972f,
      -0.9635032513015033f, -0.6488463754082889f, -0.06251338850275678f,
      -0.020562185445693163f, 0.8078151871894353f, -0.5663285742184426f,
      0.6980105526254813f, -0.2829686770254754f, -0.8266564044541028f,
      -0.10213752723302448f, 0.16305638405335435f, -0.9449324614478657f,
      -0.4672238509595964},
     -0.1642717824155953,
     true},
    {{-0.5212852070386951f, -0.1427898602060981f, -0.6518047016359164f,
      -0.16300538808637183f, -0.6643912881165291f, -0.8250648509696353f,
      -0.3922755420140402f, 0.7189052406944784f, -0.10858806596983328f,
      0.7869132878791683f, 0.7317730953905097f, 0.020567140998265784f,
      0.47862623622178757f, 0.7632390329295683f, -0.4859129543265185f,
      0.2643310627666641},
     0.3329715356680971,
     true},
    {{-0.31355942698472583f, 0.2571923271397927f, -0.9740230486940366f,
      0.5347252911482782f, 0.8162575966644218f, -0.726102745293151f,
      0.9708871272034351f, -0.13903507431607273f, 0.3560173611109872f,
      -0.5621819457625995f, -0.448535030933449f, -0.9165415993374472f,
      -0.8861046813017688f, -0.4166828736840327f, 0.1304132302862829f,
      -0.5481283538226986},
     -0.11286609885112847,
     true},
    {{0.30310671948588314f, 0.930446447161041f, -1.1208592868037797f,
      -0.9961786730957051f, 0.8045943251547478f, 0.5305917938794823f,
      -0.5390032122200563f, -0.5445482223561083f, 0.9450851695009526f,
      0.2748785948054558f, 0.7125768182113058f, -0.22199208300063772f,
      -0.13581115499190707f, 0.35683429961411256f, 0.3917291063806325f,
      0.504492503541297},
     0.009097626288239317,
     true},
    {{0.6882357910732835f, 0.24517594182496816f, -0.9611539625759499f,
      -0.8338692684349545f, -0.3158404240850337f, -0.9053854131769479f,
      0.21592576118430795f, 0.9077877585783013f, 0.28437014930735727f,
      -0.9809087622203015f, -0.36366390335364357f, 0.15441028435205792f,
      -0.9838006177476292f, 0.5800928237555953f, -0.19599398865535111f,
      0.5471143774170266},
     -0.060546166305712024,
     true},
    {{-0.9177844927541001f, 0.11348127699377761f, -0.3284916164500222f,
      -0.6553212177229495f, -0.2629288960966895f, -0.7278235211229659f,
      -0.8863103233624317f, 0.04603812347292394f, 0.8377673710590503f,
      0.19812987118891892f, 0.7970507841794557f, 0.6602441263189734f,
      0.7119509318654991f, 0.022631262817546818f, -0.5888278862459624f,
      -0.4047599778565387},
     0.1609924381705681,
     true},
};
Neuron<16> LAYER2[1] = {
    {{0.30134911824250515f, -0.6628069446492874f, -0.4578082679602536f,
      0.08982502466935978f, 0.9788047931988489f, 0.37717063392447403f,
      0.6208936737527867f, -0.312952147375254f, 0.6362650149259922f,
      0.49098371554989795f, -0.39425165832551057f, 0.5766163353520937f,
      -0.0727937893449298f, -1.4540224284662608f, 0.2499096504008615f,
      1.1384087402432006},
     -0.12341459914540999,
     false},
};