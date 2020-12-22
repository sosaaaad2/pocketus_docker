package com.pingan.ultrasound.demo.data.ai;

import android.annotation.SuppressLint;
import android.content.Context;

import com.google.protobuf.ByteString;
import com.pingan.ultrasound.demo.data.ai.db.AIReportDbHelper;
import com.pingan.ultrasound.demo.data.ai.db.DetectionDbHelper;
import com.pingan.ultrasound.demo.data.handler.ResponseResultHandler;
import com.pingan.ultrasound.demo.data.patient.db.PatientDbHelper;
import com.pingan.ultrasound.demo.domain.feature.ai.AIRepository;
import com.pingan.ultrasound.demo.domain.feature.ai.request.ClassifyAndTiradsRequest;
import com.pingan.ultrasound.demo.domain.feature.ai.request.DeleteNoduleRequest;
import com.pingan.ultrasound.demo.domain.feature.ai.request.GenerateReportRequest;
import com.pingan.ultrasound.demo.domain.feature.ai.request.ImageDetectRequest;
import com.pingan.ultrasound.demo.domain.feature.ai.request.ModifyReportRequest;
import com.pingan.ultrasound.demo.domain.feature.patient.bean.AIReportInfo;
import com.pingan.ultrasound.demo.domain.feature.patient.bean.NoduleInfo;
import com.pingan.ultrasound.demo.domain.feature.patient.bean.PatientInfo;
import com.pingan.ultrasound.demo.net.GrpcComponent;
import com.pingan.ultrasound.demo.ui.ultrasound.bean.image.NotePath;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.UUID;

import javax.inject.Inject;

import io.reactivex.Observable;
import io.reactivex.ObservableEmitter;
import io.reactivex.ObservableOnSubscribe;
import io.reactivex.ObservableSource;
import io.reactivex.Observer;
import io.reactivex.functions.Consumer;
import io.reactivex.functions.Function;
import thyroidaiproto.ThyroidaiGrpcGrpc;
import thyroidaiproto.Thyroidrpc;

import static com.pingan.ultrasound.demo.BaseApplication.mCurrentDoctorInfo;

/**
 * @author Joy
 * @version 1.0.0
 * 2019-08-29 10:03
 */
public class AIDataRepository implements AIRepository {


    @Inject
    public AIDataRepository(Context context){
    }

    @Override
    public Observable<Thyroidrpc.Nodules> imageDetect(ImageDetectRequest.Params params) {


        return Observable.create(new ObservableOnSubscribe<ArrayList<Integer>>() {
            @Override
            public void subscribe(ObservableEmitter<ArrayList<Integer>> emitter) throws Exception {
                ArrayList<Integer> result = new ArrayList<>();
                for (int item : params.imageData) {
                    result.add(item);
                }
                emitter.onNext(result);
            }
        }).map(new Function<ArrayList<Integer>, Thyroidrpc.DetectRequest>() {
            @Override
            public Thyroidrpc.DetectRequest apply(ArrayList<Integer> integers) throws Exception {
                return Thyroidrpc.DetectRequest
                        .newBuilder()
                        .setHeight(params.height)
                        .setWidth(params.width)
                        .setIsRaw(params.isRaw)
                        .setImage(ByteString.copyFrom(params.imageData))
                        .build();
            }
        }).flatMap(new Function<Thyroidrpc.DetectRequest, ObservableSource<Thyroidrpc.ProtoResponse>>() {
            @Override
            public ObservableSource<Thyroidrpc.ProtoResponse> apply(Thyroidrpc.DetectRequest detectRequest) throws Exception {
                return Observable.create(new ObservableOnSubscribe<Thyroidrpc.ProtoResponse>() {
                    @Override
                    public void subscribe(ObservableEmitter<Thyroidrpc.ProtoResponse> emitter) throws Exception {
                        ThyroidaiGrpcGrpc.ThyroidaiGrpcBlockingStub stub = ThyroidaiGrpcGrpc.newBlockingStub(GrpcComponent.getChannel());
                        Thyroidrpc.ProtoResponse protoResponse = stub.detect(detectRequest);
                        emitter.onNext(protoResponse);
                    }
                });
            }
        }).compose(ResponseResultHandler.handleResult(Thyroidrpc.Nodules.class));
//                .flatMap(new Function<Thyroidrpc.Nodules, ObservableSource<Thyroidrpc.Nodules>>() {
//            @Override
//            public ObservableSource<Thyroidrpc.Nodules> apply(Thyroidrpc.Nodules nodules) throws Exception {
//
//                return Observable.create(new ObservableOnSubscribe<Thyroidrpc.Nodules>() {
//                    @Override
//                    public void subscribe(ObservableEmitter<Thyroidrpc.Nodules> emitter) throws Exception {
//                        Thyroidrpc.NoduleWithNum noduleWithNum = Thyroidrpc.NoduleWithNum.newBuilder()
//                                .setH(100)
//                                .setW(80)
//                                .setX(50)
//                                .setY(50)
//                                .setN(1)
//                                .build();
//                        Thyroidrpc.Nodules result = Thyroidrpc.Nodules
//                                .newBuilder().setNums(1).addNodule(noduleWithNum).build();
//                        emitter.onNext(result);
//                    }
//                });
//            }
//        });

    }

    @SuppressLint("CheckResult")
    @Override
    public Observable<AIReportInfo> classifyAndTirads(ClassifyAndTiradsRequest.Params params) {


        return Observable.create(new ObservableOnSubscribe<ArrayList<Thyroidrpc.NoduleWithNum>>() {
            @Override
            public void subscribe(ObservableEmitter<ArrayList<Thyroidrpc.NoduleWithNum>> emitter) throws Exception {
                ArrayList<Thyroidrpc.NoduleWithNum> result = new ArrayList<>();
                for (NotePath item : params.noduleInfos) {

                    int x = (int) (Math.min(item.getLastPoint().x, item.getCurrentPoint().x)/params.scale);
                    int y = (int) (Math.min(item.getLastPoint().y, item.getCurrentPoint().y)/params.scale);
                    int w = (int) (Math.abs(item.getLastPoint().x - item.getCurrentPoint().x)/params.scale);
                    int h = (int) (Math.abs(item.getLastPoint().y - item.getCurrentPoint().y)/params.scale);

                    Thyroidrpc.NoduleWithNum noduleWithNum = Thyroidrpc.NoduleWithNum.newBuilder()
                            .setN(item.number)
                            .setM(item.lastNumber)
                            .setW(w)
                            .setH(h)
                            .setX(x)
                            .setY(y)
                            .setPos(item.position)
                            .setS(params.cutType)
                            .build();
                    result.add(noduleWithNum);

                }
                emitter.onNext(result);
            }
        }).map(new Function<ArrayList<Thyroidrpc.NoduleWithNum>, Thyroidrpc.CTRequest>() {
            @Override
            public Thyroidrpc.CTRequest apply(ArrayList<Thyroidrpc.NoduleWithNum> noduleWithNums) throws Exception {
                return Thyroidrpc.CTRequest
                        .newBuilder()
                        .setUid(""+params.patientInfo.no)
                        .setScanID(params.scanID)
                        .setHeight(params.height)
                        .setWidth(params.width)
                        .setPpc(160)
                        .setIsRaw(params.isRaw)
                        .setImage(ByteString.copyFrom(params.image))
                        .addAllNodule(noduleWithNums)
                        .setImageID(params.imageNo)
                        .build();
            }
        }).flatMap(new Function<Thyroidrpc.CTRequest, ObservableSource<Thyroidrpc.ProtoResponse>>() {
            @Override
            public ObservableSource<Thyroidrpc.ProtoResponse> apply(Thyroidrpc.CTRequest detectRequest) throws Exception {
                return Observable.create(new ObservableOnSubscribe<Thyroidrpc.ProtoResponse>() {
                    @Override
                    public void subscribe(ObservableEmitter<Thyroidrpc.ProtoResponse> emitter) throws Exception {
                        ThyroidaiGrpcGrpc.ThyroidaiGrpcBlockingStub stub = ThyroidaiGrpcGrpc.newBlockingStub(GrpcComponent.getChannel());
                        Thyroidrpc.ProtoResponse protoResponse = stub.classifyAndTirads(detectRequest);
                        emitter.onNext(protoResponse);
                    }
                });
            }
        })
                .compose(ResponseResultHandler.handleResult(Thyroidrpc.CTResponse.class))
                .map(new Function<Thyroidrpc.CTResponse, AIReportInfo>() {
                    @Override
                    public AIReportInfo apply(Thyroidrpc.CTResponse ctResponse) throws Exception {

                        AIReportInfo reportInfo = new AIReportInfo();
                        if (ctResponse != null && ctResponse.getNums() > 0) {
                            reportInfo.id = UUID.randomUUID().toString();
                            reportInfo.patientNo = params.patientInfo.no;
                            reportInfo.doctorInfo = params.doctorInfo;
                            reportInfo.noduleInfos = new ArrayList<>();
                            for (int i = 0; i < ctResponse.getNums(); i++) {
                                Thyroidrpc.DetectionStatus status = ctResponse.getDs(i);
                                Thyroidrpc.NoduleWithNum noduleWithNum = ctResponse.getNodule(i);
                                Thyroidrpc.BenignAndProb prob = ctResponse.getBp(i);
                                Thyroidrpc.OneTiradsRes res = ctResponse.getTirads(i);
                                if (status.getStatus() == 0) {
                                    // 查找医生编辑的结节或者AI检测到的
                                    if (noduleWithNum.getN() > 0) {
                                        NoduleInfo noduleInfo = null;
                                        for (NotePath path : params.noduleInfos) {
                                            if (path.number == noduleWithNum.getN()) {
                                                noduleInfo = new NoduleInfo();
                                                noduleInfo.id = UUID.randomUUID().toString();
                                                noduleInfo.createTime = System.currentTimeMillis();
                                                noduleInfo.position = path.position;
                                                noduleInfo.cutType = path.cutType;
                                                noduleInfo.num = path.number;

                                                int x = (int) (Math.min(path.getLastPoint().x, path.getCurrentPoint().x));
                                                int y = (int) (Math.min(path.getLastPoint().y, path.getCurrentPoint().y));
                                                int w = (int) (Math.abs(path.getLastPoint().x - path.getCurrentPoint().x));
                                                int h = (int) (Math.abs(path.getLastPoint().y - path.getCurrentPoint().y));
                                                noduleInfo.x = x;
                                                noduleInfo.y = y;
                                                noduleInfo.h = h;
                                                noduleInfo.w = w;

                                                break;
                                            }
                                        }

                                        if (noduleInfo != null) {
                                            noduleInfo.patientNo = params.patientInfo.no;

                                            // 基本信息 - 暂时不用后台返回的,从参数中获取
//                                            noduleInfo.x = noduleWithNum.getX();
//                                            noduleInfo.y = noduleWithNum.getY();
//                                            noduleInfo.h = noduleWithNum.getH();
//                                            noduleInfo.w = noduleWithNum.getW();

                                            // 良恶性
                                            noduleInfo.bmType = prob.getBenign();
                                            noduleInfo.prob = prob.getProb();

                                            reportInfo.noduleInfos.add(noduleInfo);
                                        }
                                    }
                                }
                            }
                        }
                        return reportInfo;
                    }
                }).doOnNext(new Consumer<AIReportInfo>() {
                    @Override
                    public void accept(AIReportInfo info) throws Exception {
                        AIReportDbHelper.saveReportInfo(info);
                    }
                });
    }

    @Override
    public Observable<AIReportInfo> generateReport(GenerateReportRequest.Params params) {
        return Observable.create(new ObservableOnSubscribe<Thyroidrpc.UserID>() {
            @Override
            public void subscribe(ObservableEmitter<Thyroidrpc.UserID> emitter) throws Exception {
                emitter.onNext(Thyroidrpc.UserID.newBuilder().setUid(params.uid).setScanID(params.scanID).build());
            }
        }).flatMap(new Function<Thyroidrpc.UserID, ObservableSource<Thyroidrpc.ProtoResponse>>() {
            @Override
            public ObservableSource<Thyroidrpc.ProtoResponse> apply(Thyroidrpc.UserID detectRequest) throws Exception {
                return Observable.create(new ObservableOnSubscribe<Thyroidrpc.ProtoResponse>() {
                    @Override
                    public void subscribe(ObservableEmitter<Thyroidrpc.ProtoResponse> emitter) throws Exception {
                        ThyroidaiGrpcGrpc.ThyroidaiGrpcBlockingStub stub = ThyroidaiGrpcGrpc.newBlockingStub(GrpcComponent.getChannel());
                        Thyroidrpc.ProtoResponse protoResponse = stub.generateReport(detectRequest);
                        emitter.onNext(protoResponse);
                    }
                });
            }
        }).onErrorResumeNext(new Function<Throwable, ObservableSource<? extends Thyroidrpc.ProtoResponse>>() {
            @Override
            public ObservableSource<? extends Thyroidrpc.ProtoResponse> apply(Throwable throwable) throws Exception {
                return new Observable<Thyroidrpc.ProtoResponse>() {
                    @Override
                    protected void subscribeActual(Observer<? super Thyroidrpc.ProtoResponse> observer) {
                        observer.onNext(Thyroidrpc.ProtoResponse.newBuilder().build());
                    }
                };
            }
        })
                .compose(ResponseResultHandler.handleResult(Thyroidrpc.CTResponse.class))
                .map(new Function<Thyroidrpc.CTResponse, AIReportInfo>() {
                    @Override
                    public AIReportInfo apply(Thyroidrpc.CTResponse ctResponse) throws Exception {

                        AIReportInfo reportInfo = new AIReportInfo();
                        reportInfo.id = UUID.randomUUID().toString();
                        reportInfo.scanID = params.scanID;
                        reportInfo.patientNo = Integer.valueOf(params.uid);
                        reportInfo.doctorInfo = mCurrentDoctorInfo;
                        reportInfo.noduleInfos = new ArrayList<>();
                        if (ctResponse != null && ctResponse.getNums() > 0) {
                            for (int i = 0; i < ctResponse.getNums(); i++) {
                                Thyroidrpc.DetectionStatus status = ctResponse.getDs(i);
                                Thyroidrpc.NoduleWithNum noduleWithNum = ctResponse.getNodule(i);
                                Thyroidrpc.BenignAndProb prob = ctResponse.getBp(i);
                                Thyroidrpc.OneTiradsRes tirads = ctResponse.getTirads(i);
                                if (status.getStatus() == 0) {
                                    // 查找医生编辑的结节或者AI检测到的
                                    if (noduleWithNum.getN() > 0) {
                                        NoduleInfo noduleInfo = new NoduleInfo();
                                        noduleInfo.id = UUID.randomUUID().toString();
                                        noduleInfo.createTime = System.currentTimeMillis();
                                        noduleInfo.num = noduleWithNum.getN();
                                        noduleInfo.patientNo = Integer.valueOf(params.uid);
                                        noduleInfo.reportId = reportInfo.id;

                                        noduleInfo.imageInfos = params.noduleImages; // 图片信息

                                        // 基本信息
                                        noduleInfo.x = noduleWithNum.getX();
                                        noduleInfo.y = noduleWithNum.getY();
                                        noduleInfo.h = noduleWithNum.getH();
                                        noduleInfo.w = noduleWithNum.getW();

                                        // 详细信息
                                        noduleInfo.bmType = prob.getBenign();         // 良恶性
                                        noduleInfo.prob = prob.getProb();             // 概率
                                        noduleInfo.shape = tirads.getShape();         // 形状
                                        if (ctResponse.getModifyFlag(i)){
                                            noduleInfo.boundary = tirads.getBorderClear();// 边界
                                            noduleInfo.echoType = tirads.getEchoLevel();
                                        }else {
                                            noduleInfo.boundary = 3;// 边界 todo 先忽略后台写死的值
                                            noduleInfo.echoType = 5; // 回声类型 todo 先忽略后台写死的值
                                        }
                                        noduleInfo.growthMode = tirads.getRatio() > 1 ? 1 : 2; // 生长方式
                                        noduleInfo.internalStructure = tirads.getConstitute(); // 内部结构
                                        noduleInfo.position = noduleWithNum.getPos(); // 位置
                                        noduleInfo.cutType = noduleWithNum.getS();  // 横纵切
                                        // 内部强回声：无、微钙化、粗钙化、粗微钙化、边缘钙化、胶质凝集（伴彗星尾）
                                        //说明：钙化和胶质凝集（伴彗星尾）为两个模互斥，若有胶质凝集（伴彗星尾）则优先展示，若无胶质凝集（伴彗星尾）再展示钙化模型结果
                                        if (tirads.getComet() > 1){
                                            noduleInfo.strongEcho = 5;
                                        }else {
                                            int tiny = tirads.getCalcification(0); // 0：无微钙化 1： 有微钙化
                                            int curde = tirads.getCalcification(1); // 0：无粗钙化 1： 有粗钙化
                                            int ring = tirads.getCalcification(2);  // 0：无环钙化 1： 有环钙化

                                            if (ring == 1){
                                                noduleInfo.strongEcho = 4; // 边缘钙化
                                            }else if (tiny == 1 && curde == 1){
                                                noduleInfo.strongEcho = 3; // 粗微钙化
                                            }else if (tiny == 1){
                                                noduleInfo.strongEcho = 1; // 微钙化
                                            }else if (curde == 1){
                                                noduleInfo.strongEcho = 2; // 粗钙化
                                            }else if (curde == 0){
                                                noduleInfo.strongEcho = 0; // 待选
                                            }else {
                                                noduleInfo.strongEcho = 6; // 无钙化
                                            }
                                        }
                                        noduleInfo.lrsize = tirads.getHxlen(0); // 左右径
                                        noduleInfo.tbsize = tirads.getHxlen(1); // 上下径

                                        noduleInfo.fbsize = tirads.getVxlen();        // 前后径
                                        reportInfo.noduleInfos.add(noduleInfo);
                                    }
                                }
                            }
                        }
                        return reportInfo;
                    }
                }).doOnNext(new Consumer<AIReportInfo>() {
                    @Override
                    public void accept(AIReportInfo aiReportInfo) throws Exception {
                        // 按结节编号排序排序
                        if (aiReportInfo != null && aiReportInfo.noduleInfos != null && aiReportInfo.noduleInfos.size() > 0){
                            Collections.sort(aiReportInfo.noduleInfos, new Comparator<NoduleInfo>() {
                                @Override
                                public int compare(NoduleInfo o1, NoduleInfo o2) {
                                    return o1.num - o2.num;
                                }
                            });
                        }
                    }
                }).doOnNext(new Consumer<AIReportInfo>() {
                    @Override
                    public void accept(AIReportInfo info) throws Exception {
                        // insert to db
                        AIReportDbHelper.saveReportInfo(info);
                    }
                }).doOnNext(new Consumer<AIReportInfo>() {
                    @Override
                    public void accept(AIReportInfo info) throws Exception {
                        // 更新Detection 信息
                        PatientInfo patientInfo = PatientDbHelper.getPatientInfo(info.patientNo);
                        DetectionDbHelper.updateReportId(patientInfo.latestDetectInfo.id, info.id);
                    }
                });
    }

    @Override
    public Observable<AIReportInfo> modifyReport(ModifyReportRequest.Params params) {
        // 修改报告
        return Observable.create(new ObservableOnSubscribe<ModifyReportRequest.Params>() {
            @Override
            public void subscribe(ObservableEmitter<ModifyReportRequest.Params> emitter) throws Exception {

                // AIReport 转换
                ArrayList<ModifyReportRequest.ModifyItem> modifyItems = new ArrayList<>();

                for (NoduleInfo noduleInfo : params.reportInfo.noduleInfos) {

                    ModifyReportRequest.ModifyItem item = new ModifyReportRequest.ModifyItem();
                    // 前后左后上下径暂时不支持修改
                    item.hxlen = new ArrayList<>();
                    item.vxlen = new ArrayList<>();

                    // 良恶性
                    item.benign = new ArrayList<>();
                    item.benign.add(noduleInfo.bmType);
                    item.prob = new ArrayList<>();
                    if (noduleInfo.prob <= 0){
                        item.prob.add(0.8f);
                    }

                    // 内部结构
                    item.constitute = new ArrayList<>();
                    item.constitute.add(noduleInfo.internalStructure);

                    // 内部强回声
                    item.comet = new ArrayList<>();
                    item.calcification = new ArrayList<>();
                    if (noduleInfo.strongEcho == 5){
                        item.comet.add(2);
                    }else {

//                        int tiny = tirads.getCalcification(0); // 0：无微钙化 1： 有微钙化
//                        int curde = tirads.getCalcification(1); // 0：无粗钙化 1： 有粗钙化
//                        int ring = tirads.getCalcification(2);  // 0：无环钙化 1： 有环钙化
//
//                        if (ring == 1){
//                            noduleInfo.strongEcho = 4; // 边缘钙化
//                        }else if (tiny == 1 && curde == 1){
//                            noduleInfo.strongEcho = 3; // 粗微钙化
//                        }else if (tiny == 1){
//                            noduleInfo.strongEcho = 1; // 微钙化
//                        }else if (curde == 1){
//                            noduleInfo.strongEcho = 2; // 粗钙化
//                        }else if (curde == 0){
//                            noduleInfo.strongEcho = 0; // 待选
//                        }else {
//                            noduleInfo.strongEcho = 6; // 无钙化
//                        }

                        if (noduleInfo.strongEcho > 0){
                            item.comet.add(0);
                            if (noduleInfo.strongEcho == 1){
                                item.calcification.add(1);
                                item.calcification.add(0);
                                item.calcification.add(0);
                            }else if (noduleInfo.strongEcho == 2){
                                item.calcification.add(0);
                                item.calcification.add(1);
                                item.calcification.add(0);
                            }else if (noduleInfo.strongEcho == 3){
                                item.calcification.add(1);
                                item.calcification.add(1);
                                item.calcification.add(0);
                            }else if (noduleInfo.strongEcho == 4){
                                item.calcification.add(0);
                                item.calcification.add(0);
                                item.calcification.add(1);
                            }else if (noduleInfo.strongEcho == 6){
                                // 由于存在待选的情况，这里做区分
                                item.calcification.add(2);
                                item.calcification.add(2);
                                item.calcification.add(2);
                            }
                        }
                    }

                    // 形状
                    item.shape = new ArrayList<>();
                    item.shape.add(noduleInfo.shape);

                    // 生长方式
                    item.ratio = new ArrayList<>();
                    if (noduleInfo.growthMode == 1){
                        item.ratio.add(2f);
                    }else {
                        item.ratio.add(0.5f);
                    }
                    // 回声类型
                    item.echo_level = new ArrayList<>();
                    item.echo_level.add(noduleInfo.echoType);
                    // 边界
                    item.border_clear = new ArrayList<>();
                    item.border_clear.add(noduleInfo.boundary);

                    // 请求
                    Thyroidrpc.ModifyCT modifyCt = Thyroidrpc.ModifyCT
                            .newBuilder().addAllBenign(item.benign)
                            .addAllProb(item.prob)
                            .addAllBorderClear(item.border_clear)
                            .addAllCalcification(item.calcification)
                            .addAllComet(item.comet)
                            .addAllRatio(item.ratio)
                            .addAllShape(item.shape)
                            .addAllHxlen(item.hxlen)
                            .addAllVxlen(item.vxlen)
                            .addAllEchoLevel(item.echo_level)
                            .addAllConstitute(item.constitute)
                            .setUid(params.uid)
                            .setScanID(params.scanID)
                            .setNindex(noduleInfo.num)
                            .build();

                    ThyroidaiGrpcGrpc.ThyroidaiGrpcBlockingStub stub = ThyroidaiGrpcGrpc.newBlockingStub(GrpcComponent.getChannel());
                    Thyroidrpc.ProtoResponse protoResponse = stub.modifyReport(modifyCt);
                }

                emitter.onNext(params);
                emitter.onComplete();
            }
        }).flatMap(new Function<ModifyReportRequest.Params, ObservableSource<Thyroidrpc.ProtoResponse>>() {
            @Override
            public ObservableSource<Thyroidrpc.ProtoResponse> apply(ModifyReportRequest.Params params) throws Exception {

                return Observable.create(new ObservableOnSubscribe<Thyroidrpc.ProtoResponse>() {
                    @Override
                    public void subscribe(ObservableEmitter<Thyroidrpc.ProtoResponse> emitter) throws Exception {
                        // 读取报告
                        Thyroidrpc.UserID userID = Thyroidrpc.UserID
                                .newBuilder()
                                .setScanID(params.scanID)
                                .setUid(params.uid)
                                .build();

                        ThyroidaiGrpcGrpc.ThyroidaiGrpcBlockingStub stub = ThyroidaiGrpcGrpc.newBlockingStub(GrpcComponent.getChannel());
                        Thyroidrpc.ProtoResponse protoResponse = stub.generateReport(userID);

                        emitter.onNext(protoResponse);
                        emitter.onComplete();
                    }
                });
            }
        }).map(new Function<Thyroidrpc.ProtoResponse, AIReportInfo>() {
            @Override
            public AIReportInfo apply(Thyroidrpc.ProtoResponse protoResponse) throws Exception {
                // 暂时不用读取的报告信息，太麻烦
                return params.reportInfo;
            }
        }).doOnNext(new Consumer<AIReportInfo>() {
            @Override
            public void accept(AIReportInfo info) throws Exception {
                // 更新数据库
                AIReportDbHelper.saveReportInfo(info);
            }
        });
    }

    @Override
    public Observable<Thyroidrpc.ProtoResponse> deleteNodule(DeleteNoduleRequest.Params params) {
        return Observable.create(new ObservableOnSubscribe<Thyroidrpc.DeleteNoduleRequest>() {
            @Override
            public void subscribe(ObservableEmitter<Thyroidrpc.DeleteNoduleRequest> emitter) throws Exception {
                emitter.onNext(Thyroidrpc
                        .DeleteNoduleRequest
                        .newBuilder()
                        .addAllNoduleID(params.noduleNos)
                        .setImageID(params.imageId)
                        .setScanID(params.scanID)
                        .setUid(params.uid).build());
                emitter.onComplete();
            }
        }).flatMap(new Function<Thyroidrpc.DeleteNoduleRequest, ObservableSource<Thyroidrpc.ProtoResponse>>() {
            @Override
            public ObservableSource<Thyroidrpc.ProtoResponse> apply(Thyroidrpc.DeleteNoduleRequest deleteNoduleRequest) throws Exception {
                return Observable.create(new ObservableOnSubscribe<Thyroidrpc.ProtoResponse>() {
                    @Override
                    public void subscribe(ObservableEmitter<Thyroidrpc.ProtoResponse> emitter) throws Exception {
                        ThyroidaiGrpcGrpc.ThyroidaiGrpcBlockingStub stub = ThyroidaiGrpcGrpc.newBlockingStub(GrpcComponent.getChannel());
                        Thyroidrpc.ProtoResponse protoResponse = stub.deleteNodule(deleteNoduleRequest);
                        emitter.onNext(protoResponse);
                    }
                });
            }
        });
    }
}
