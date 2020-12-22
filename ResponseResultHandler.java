package com.pingan.ultrasound.demo.data.handler;


import com.google.protobuf.Any;
import com.google.protobuf.GeneratedMessageLite;
import com.google.protobuf.Message;
import com.google.protobuf.MessageLite;
import com.pingan.ultrasound.demo.net.exception.DefaultException;

import io.reactivex.Observable;
import io.reactivex.ObservableEmitter;
import io.reactivex.ObservableOnSubscribe;
import io.reactivex.ObservableSource;
import io.reactivex.ObservableTransformer;
import io.reactivex.functions.Function;
import thyroidaiproto.Thyroidrpc;

/**
 * 通信结果处理类
 * @author Joy
 * @version 2016/8/8
 *          CopyRight www.eku001.com
 */
public class ResponseResultHandler {

    /**
     * 对通信请求结果进行预处理
     */
    public static <T extends Message> ObservableTransformer<Thyroidrpc.ProtoResponse, T> handleResult(final Class<T> clzz) {

        return new ObservableTransformer<Thyroidrpc.ProtoResponse, T>() {
            @Override
            public ObservableSource<T> apply(Observable<Thyroidrpc.ProtoResponse> upstream) {

                return upstream.flatMap(new Function<Thyroidrpc.ProtoResponse, Observable<T>>() {
                    @Override
                    public Observable<T> apply(Thyroidrpc.ProtoResponse result) throws Exception {
                        if (result.getCode() == 0) {
                            // 获取data
                            final Any anyData = result.getData();
                            return Observable.create(new ObservableOnSubscribe<T>() {
                                @Override
                                public void subscribe(ObservableEmitter<T> e) throws Exception {

                                    try {
                                        e.onNext(anyData.unpack(clzz));
                                        e.onComplete();
                                    } catch (Exception exception) {
                                        exception.printStackTrace();
                                        e.onError(exception);
                                    }
                                }
                            });
                        } else{ // 特殊错误需要处理
                            return Observable.error(new DefaultException(result.getCode(), result.getMsg()));
                        }
                    }
                });
            }
        };
    }
}
